from torch import float16
from modules import devices

threads = 8

device = devices.get_optimal_device_name()
precision = float16

#Size in bytes
cache_size = 1024*1024*1024*4
tensor_cache = None

#moves finished tensors to CPU during merge to save video memory (slower)
low_vram = False

loaded_checkpoints = None
primary = ""

last_merge_tasks_hash = ""

#Components

slidervalues = {}