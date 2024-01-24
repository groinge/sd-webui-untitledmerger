from safetensors.torch import safe_open
from safetensors import SafetensorError
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import scripts.untitled.operators as oper
import scripts.untitled.misc_util as mutil
import scripts.untitled.common as cmn
import torch,os,re,gc
from tqdm import tqdm
from copy import copy,deepcopy
from modules import devices,shared,script_loading,paths,paths_internal

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

cmn.tensor_cache = oper.Cache(cmn.cache_size)

def prepare_merge(calcmode,targets,checkpoints,discard_targets,cludes,timer,finetune) -> dict:
    cmn.primary = checkpoints[0]
    
    with safe_open_multiple(checkpoints,device=cmn.device) as cmn.loaded_checkpoints:
        state_dict_keys = cmn.loaded_checkpoints[cmn.primary].keys()

        tasks = parse_recipe(calcmode,targets,state_dict_keys,cmn.primary,discard_targets,cludes,checkpoints)
        tasks_copy = copy(tasks)

        state_dict = {}
        #Reuse merged tensors from the last merge's loaded model, if availible
        if shared.sd_model.sd_checkpoint_info.short_title == hash(cmn.last_merge_tasks):
            state_dict,tasks = get_tensors_from_loaded_model(state_dict,tasks)
        
        if cmn.trash_model:
            shared.sd_model.to(device='meta')
            devices.torch_gc()

        timer.record('Prepare merge')
        try:
            with ThreadPoolExecutor(max_workers=cmn.threads) as executor:
                results = executor.map(initialize_merge,tasks)
                executor.shutdown()
        except:
            clear_cache()
            raise
    
    state_dict.update(dict(results))

    fine = fineman(finetune, False)
    if finetune:
        for key in FINETUNES:
            state_dict.get(key)
            if key:
                index = FINETUNES.index(key)
                if 5 > index : 
                    state_dict[key] = state_dict[key]* fine[index] 
                else :state_dict[key] = state_dict[key] + torch.tensor(fine[5]).to(state_dict[key].device)
                for task in tasks_copy:
                    if task.key == key:
                        tasks_copy.remove(task)


    cmn.last_merge_tasks = tuple(tasks_copy)
            
    timer.record('Merge')
    return state_dict


def parse_recipe(calcmode,targets,keys,primary,discard,clude,checkpoints) -> list:
    cludemode = clude.pop(0)
    tasks = []
    discard_regex = re.compile(mutil.target_to_regex(discard))
    discard_keys = list(filter(lambda x: re.search(discard_regex,x),keys))

    desired_keys = keys
    if clude:
        clude_regex = re.compile(mutil.target_to_regex(clude))
        if cludemode == 'exclude':
            desired_keys = list(filter(lambda x: not re.search(clude_regex,x),keys))
        else:
            desired_keys = list(filter(lambda x: re.search(clude_regex,x),keys))

    assigned_keys = assign_weights_to_keys(targets,desired_keys)

    for key in keys:
        if key in discard_keys:continue
        elif key in SKIP_KEYS or 'model_ema' in key:
            tasks.append(oper.LoadTensor(key,primary))
        elif key in assigned_keys.keys():
            tasks.append(calcmode.create_recipe(key,*checkpoints,**assigned_keys[key]))
        else: 
            tasks.append(oper.LoadTensor(key,primary))
    return tasks


def initialize_merge(task) -> tuple:
    try:
        tensor = task.merge()
    except SafetensorError: #Fallback in case one of the secondary models lack a key present in the primary model
        tensor = cmn.loaded_checkpoints[cmn.primary].get_tensor(task.key)

    #tensor = tensor.detach().cpu()
    devices.torch_gc()
    #torch.cuda.empty_cache()
    return (task.key, tensor)


def assign_weights_to_keys(targets,keys) -> dict:
    weight_assigners = []
    keystext = "\n".join(keys)

    for target_name,weights in targets.items():
        regex = mutil.target_to_regex(target_name)

        weight_assigners.append((weights, regex))
    
    keys_n_weights = list()

    for weights, regex in weight_assigners:
        target_keys = re.findall(regex,keystext,re.M)
        keys_n_weights.append((target_keys,weights))
    
    keys_n_weights.sort(key=lambda x: len(x[0]))
    keys_n_weights.reverse()

    assigned_keys = defaultdict()
    assigned_keys.default_factory = dict
    
    for keys, weights in keys_n_weights:
        for key in keys:
            assigned_keys[key].update(weights)

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


class safe_open_multiple(object):
    def __init__(self,checkpoints,device):
        self.checkpoints = checkpoints
        self.device = device
        self.open_files = {}
     
    def __enter__(self):
        for name in self.checkpoints:
            if name:
                filename = os.path.join(paths_internal.models_path,'Stable-diffusion',name)
                self.open_files[name] = safe_open(filename,framework='pt',device=self.device)
        return self.open_files

    def __exit__(self,*args):
        for file in self.open_files.values():
            file.__exit__(*args)


def clear_cache():
    cmn.tensor_cache.__init__(cmn.cache_size)
    gc.collect()
    devices.torch_gc()
    torch.cuda.empty_cache()
    return "All caches cleared"


#From https://github.com/hako-mikan/sd-webui-supermerger
def fineman(fine,isxl):
    if fine.find(",") != -1:
        tmp = [t.strip() for t in fine.split(",")]
        fines = [0.0]*8
        for i,f in enumerate(tmp[0:8]):
            try:
                f = float(f)
                fines[i] = f
            except Exception:
                pass

        fine = fines
    else:
        return None

    fine = [
        1 - fine[0] * 0.01,
        1+ fine[0] * 0.02,
        1 - fine[1] * 0.01,
        1+ fine[1] * 0.02,
        1 - fine[2] * 0.01,
        [fine[3]*0.02] + colorcalc(fine[4:8],isxl)
        ]
    return fine

def colorcalc(cols,isxl):
    colors = COLSXL if isxl else COLS
    outs = [[y * cols[i] * 0.02 for y in x] for i,x in enumerate(colors)]
    return [sum(x) for x in zip(*outs)]

COLS = [[-1,1/3,2/3],[1,1,0],[0,-1,-1],[1,0,1]]
COLSXL = [[0,0,1],[1,0,0],[-1,-1,0],[-1,1,0]]

def weighttoxl(weight):
    weight = weight[:9] + weight[12:22] +[0]
    return weight

FINETUNES = [
"model.diffusion_model.input_blocks.0.0.weight",
"model.diffusion_model.input_blocks.0.0.bias",
"model.diffusion_model.out.0.weight",
"model.diffusion_model.out.0.bias",
"model.diffusion_model.out.2.weight",
"model.diffusion_model.out.2.bias",
]