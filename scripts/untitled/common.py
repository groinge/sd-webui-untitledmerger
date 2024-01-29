import torch

blocks = None
opts = None
last_seed = 5318008

stop = False
interrupted = False

loaded_checkpoints = None
checkpoints_types = None
primary = ""

last_merge_tasks = tuple()

def device():
    device,dtype = opts['device'].split('/')
    return device 

def dtype():
    device,dtype = opts['device'].split('/')
    if dtype == 'float16': return torch.float16
    elif dtype == 'float8': return torch.float8_e4m3fn
    else: return torch.float32

