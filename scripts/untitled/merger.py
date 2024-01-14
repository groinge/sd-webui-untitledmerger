from safetensors.torch import safe_open
from safetensors import SafetensorError
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import scripts.untitled.operators as oper
import scripts.common as cmn
import torch,os,re
from tqdm import tqdm
from copy import copy
from modules import devices,shared,script_loading,paths,sd_hijack

networks = script_loading.load_module(os.path.join(paths.extensions_builtin_dir,'Lora','networks.py'))

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

#Operator names can only have letters and "- in them
OPERATORS = {'add': oper.add,
             'sub': oper.sub,
             'weight-sum': oper.weight_sum,
             'traindiff': oper.traindiff,
             'extract': oper.extract_super
             }

#Hashable merge info that is used as keys for the cache
class TaskInfo:
    def __init__(self,key, task, args, sources):
        self.key = key
        self.task = task
        self.args_keys = tuple(args.keys())
        self.args_values = tuple(args.values())
        self.sources = tuple(sources)
        self.hash = hash((self.key, self.task, self.args_keys, self.args_values, self.sources))

    def update_hash(self):
        self.hash = hash((self.key, self.task, self.args_keys, self.args_values, self.sources))

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return self.key == other.key and self.task == other.task and self.args_keys == other.args_keys and self.args_values == other.args_values and self.sources == other.sources
    
    def __call__(self):
        return self.task(self)
    
    def __getitem__(self,key):
        i = self.args_keys.index(key)
        return self.args_values[i]


def create_task(key, task_input, args) -> TaskInfo: 
    readied_sources = []
    task_name = re.match(r"[a-z\-]*",task_input).group()
    
    if task_name == 'checkpoint':
        task = oper.load_tensor
        args = {'filename':args}
    else:
        task = OPERATORS[task_name]
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
            tensor = self.cache.popitem(last=False)[1]
            self.footprint -= tensor_size(tensor)

def tensor_size(t: torch.Tensor) -> int:
    return t.element_size() * t.nelement()

cmn.tensor_cache = tCache(cmn.cache_size)


def prepare_merge(recipe,timer) -> dict:
    checkpoints = recipe['checkpoints']
    tasks_recipe = recipe['targets']
    cmn.primary = os.path.basename(recipe['primary_checkpoint'])
    
    with safe_open_multiple(checkpoints,device=cmn.device) as cmn.loaded_checkpoints:
        state_dict_keys = cmn.loaded_checkpoints[cmn.primary].keys()

        tasks = parse_recipe(tasks_recipe,state_dict_keys,cmn.primary)
        tasks_copy = copy(tasks)

        state_dict = {}
        #Reuse merged tensors from the last merge's loaded model, if availible
        if shared.sd_model.sd_checkpoint_info.name_for_extra == str(hash(cmn.last_merge_tasks)):
            state_dict,tasks = get_tensors_from_loaded_model(state_dict,tasks)
        
        timer.record('Prepare merge')

        with ThreadPoolExecutor(max_workers=cmn.threads) as executor:
            results = executor.map(initialize_merge,tasks)
    
    cmn.last_merge_tasks = tuple(tasks_copy)
    state_dict.update(dict(results))
    
    timer.record('Merge')
    return state_dict


def parse_recipe(recipe,keys,primary) -> list:
    tasks = []

    assigned_keys = assign_keys_to_targets(list(recipe.keys()),keys)

    for key in keys:
        if key in SKIP_KEYS or 'model_ema' in key:
            tasks.append(create_task(key,'checkpoint',primary))
        elif key in assigned_keys.keys():
            tasks.append(create_task(key,*list(recipe[assigned_keys[key]].items())[0]))
        else: 
            tasks.append(create_task(key,'checkpoint',primary))
    return tasks


def initialize_merge(taskinfo) -> tuple:
    try:
        tensor = taskinfo()
    except SafetensorError:
        tensor = cmn.loaded_checkpoints[cmn.primary].get_tensor(taskinfo.key)

    if cmn.low_vram:
        tensor = tensor.detach().cpu()
    devices.torch_gc()
    #torch.cuda.empty_cache()
    return (taskinfo.key, tensor)

BASE_SELECTORS = {
    "all":  "",
    "clip": "cond",
    "unet": "model\\.diffusion_model",
    "in":   "model\\.diffusion_model\\.input_blocks",
    "out":  "model\\.diffusion_model\\.output_blocks",
    "mid":  "model\\.diffusion_model\\.middle_block"
}

BASE_SELECTORS_PRIORITY = {
    "all":  0,
    "clip": 1,
    "unet": 1,
    "in":   2,
    "out":  2,
    "mid":  2
}
def assign_keys_to_targets(targets,keys) -> dict:
    target_assigners = []
    keystext = "\n".join(keys)+"\n"

    for target_name in targets:
        target = re.split("\.|:",target_name.lower())

        regex = "^"
        priority = 0
        if target[0] in BASE_SELECTORS:
            priority += BASE_SELECTORS_PRIORITY[target[0]]
            regex += BASE_SELECTORS[target.pop(0)]

        priority += len(target)

        for selector in target:
            #Check if the selector qualifies as a number
            if re.search(r'^[\d,-]*$',selector):
                
                #Turns number inputs like "2-5,10" in to "2|3|4|5|10"
                splitnumeric = set(selector.split(','))
                for segment in splitnumeric.copy():
                    if '-' in segment:
                        a, b = segment.split('-')
                        valuerange = [str(i) for i in range(int(a),int(b)+1)]
                        splitnumeric.remove(segment)
                        splitnumeric.update(valuerange)
                formattedselector = '|'.join(splitnumeric)

                regex += f"\\D*\\.(?:{formattedselector})\\."
            else:
                regex += f".*{selector}"

        regex += ".*$"
        target_assigners.append((priority,target_name,regex))
    
    #Sorts the selectors according to the priority of the base selector + the number of segments in the target. 
    #This is to give more specific targets priority when assigning keys, unfortunately this system is kinda imperfect
    target_assigners.sort(key=lambda x:x[0],reverse=True)
    
    assigned_keys = dict()

    for _, target, regex in target_assigners:
        target_keys = re.findall(regex,keystext,re.M)
        target_dict = dict.fromkeys(target_keys,target)

        for key in target_keys:
            keystext = keystext.replace(key+"\n","")

        assigned_keys.update(target_dict)
    return assigned_keys


def get_tensors_from_loaded_model(state_dict,tasks) -> dict:
        intersected = set(cmn.last_merge_tasks).intersection(set(tasks))
        if intersected:
            #clear loras from model
            with torch.no_grad():
                for module in shared.sd_model.modules():
                    networks.network_restore_weights_from_backup(module)
            old_state_dict = shared.sd_model.state_dict()

            for task in intersected:
                try:
                    state_dict[task.key] = old_state_dict[task.key]
                except:pass
                tasks.remove(task)
            
        return state_dict,tasks

#Tensors are loaded lazily throughout the merge, both to save memory and reduce code complexity.
class safe_open_multiple(object):
    def __init__(self,checkpoints,device):
        self.checkpoints = checkpoints
        self.device = device
        self.open_files = {}
     
    def __enter__(self):
        for filename in self.checkpoints:
            self.open_files[os.path.basename(filename)] = safe_open(filename,framework='pt',device=self.device)
        return self.open_files

    def __exit__(self,*args):
        for file in self.open_files.values():
            file.__exit__(*args)

