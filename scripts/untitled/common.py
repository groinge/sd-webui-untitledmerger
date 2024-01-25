from torch import float16
from modules import devices

threads = 12

device = devices.get_optimal_device_name()
precision = float16

#Size in bytes
cache_size = 1024*1024*1024*4

#Removes loaded model from memory at the start of the merge, requiring a new one to be initialized before loading
trash_model = False

interrupted = False
loaded_checkpoints = None
primary = ""

last_merge_tasks = tuple()