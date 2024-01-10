from safetensors.torch import safe_open
from safetensors import SafetensorError
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import scripts.merging.operators as oper
import scripts.common as cmn
import torch
from modules import devices

SKIP_KEYS = [
    "alphas_cumprod",
    "alphas_cumprod_prev",
    "betas",
    "log_one_minus_alphas_cumprod",
    "posterior_log_variance_clipped",
    "posterior_mean_coef1",
    "posterior_mean_coef2",
    "posterior_variance",
    "sqrt_alphas_cumprod",
    "sqrt_one_minus_alphas_cumprod",
    "sqrt_recip_alphas_cumprod",
    "sqrt_recipm1_alphas_cumprod"
]

#Hashable merge info that is used as keys for the cache
class TaskInfo:
    def __init__(self,key, task, args, sources):
        self.key = key
        self.task = task
        self.args_keys = tuple(args.keys())
        self.args_values = tuple(args.values())
        self.sources = tuple(sources)

    def __hash__(self):
        return hash((self.key, self.task, self.args_keys, self.args_values, self.sources))

    def __eq__(self, other):
        return self.key == other.key and self.task == other.task and self.args_keys == other.args_keys and self.args_values == other.args_values and self.sources == other.sources
    
    def __getitem__(self,key):
        i = self.args_keys.index(key)
        return self.args_values[i]


def create_task(key, task_str, args):
    readied_sources = []
    
    if task_str.startswith("checkpoint"):
        task = oper.load_tensor
        args = {'filename':args}
        readied_sources = []
    else:
        task = cmn.operators[task_str]
        args_copy = args.copy()
        for source_task,source_args in args['sources'].items():
            readied_sources.append(create_task(key,source_task,source_args))
        del args_copy['sources']
        args = args_copy

    return TaskInfo(
        key = key,
        task = task,
        args = args,
        sources = readied_sources
        )

#Items are added at the end of the dict and removed at the beginning 
#High overhead, so is only worth using for computationally demanding operations
class tCache:
    def __init__(self,size):
        self.cache = OrderedDict()
        self.size = size
        self.footprint = 0

    def append(self,key: TaskInfo,tensor):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            tensor = tensor.detach().cpu()
            self.cache.update({key:tensor})
            self.footprint += tensor_size(tensor)
            self.clear()

    def retrieve(self,key: TaskInfo):
        tensor = self.cache[key]
        self.cache.move_to_end(key)
        return tensor.detach().clone().to(cmn.device).type(cmn.precision)

    def clear(self):
        while self.footprint > self.size:
            tensor = self.cache.popitem(last=False)
            self.footprint -= tensor_size(tensor)

def tensor_size(t):
    return t.element_size() * t.nelement()

cmn.tensor_cache = tCache(cmn.cache_size)

def parse_recipe(recipe,keys,primary):
    tasks = []
    for key in keys:
        try:
            if key in SKIP_KEYS or 'model_ema' in key:
                tasks.append(create_task(key,'checkpoint',primary))
            elif key in recipe:
                tasks.append(create_task(key,*list(recipe[key].items())[0]))
            elif 'ALL.' in recipe:
                tasks.append(create_task(key,*list(recipe['ALL.'].items())[0]))
            else: 
                tasks.append(create_task(key,'checkpoint',primary))
        except SafetensorError:
            tasks.append(create_task(key,'checkpoint',primary))
    return tasks
        

def prepare_merge(recipe):
    checkpoints = recipe['checkpoints']
    tasks_recipe = recipe['targets']
    primary = recipe['primary_checkpoint']
    print(primary)
    with safe_open_multiple(checkpoints,device=cmn.device) as cmn.loaded_checkpoints:
        state_dict_keys = cmn.loaded_checkpoints[primary].keys()

        tasks = parse_recipe(tasks_recipe,state_dict_keys,primary)

        with ThreadPoolExecutor(max_workers=cmn.threads) as executor:
            results = executor.map(initialize_merge,tasks)
    jumbled_dict = dict(results)
    state_dict = OrderedDict({key:jumbled_dict[key] for key in state_dict_keys})
            
    return state_dict


def initialize_merge(taskinfo):
    tensor = merge(taskinfo)

    if cmn.low_vram:
        tensor = tensor.detach().cpu()
    devices.torch_gc()
    return (taskinfo.key,tensor)


def merge(taskinfo:TaskInfo):
    try:
        return cmn.tensor_cache.retrieve(taskinfo)
    except KeyError:pass

    source_tensors = []
    #adding multithreading here would probably help performance, need to implement a combined thread limit to keep it from using too many threads
    for source in taskinfo.sources:
        source_tensors.append(merge(source))

    result = taskinfo.task(*source_tensors,taskinfo)
    if taskinfo.task != oper.load_tensor and taskinfo.task != oper.add:
        cmn.tensor_cache.append(taskinfo,result)

    return result


#Tensors are loaded lazily throughout the merge, both to save memory and reduce code complexity. Pickletensors are not supported due to their high overhead.
class safe_open_multiple(object):
    def __init__(self,checkpoints,device):
        self.checkpoints = checkpoints
        self.device = device
        self.open_files = {}
     
    def __enter__(self):
        for alias,filename in self.checkpoints.items():
            self.open_files[alias] = safe_open(filename,framework='pt',device=self.device)
        return self.open_files

    def __exit__(self,*args):
        for file in self.open_files.values():
            file.__exit__(*args)



    
