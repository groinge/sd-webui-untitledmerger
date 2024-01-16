import torch,scipy
import scripts.common as cmn
from copy import deepcopy


###DECORATORS####

def recursion(func):
    def inner(taskinfo):
        source_tensors = []
        for source_taskinfo in taskinfo.sources:
            source_tensors.append(source_taskinfo())

        return func(*source_tensors,taskinfo)
    return inner


def cache_operation(func):
    def inner(taskinfo):
        try:
            return cmn.tensor_cache.retrieve(taskinfo)
        except KeyError:pass

        result = func(taskinfo)

        cmn.tensor_cache.append(taskinfo,result)
        return result
    return inner


###OPERATORS####

def load_tensor(taskinfo) -> torch.Tensor:
    key = taskinfo.key
    fname = taskinfo['filename']
    return cmn.loaded_checkpoints[fname].get_tensor(key).to(cmn.device)


@recursion
def weight_sum(a,b,taskinfo) -> torch.Tensor:
    mult_b = taskinfo['alpha']
    mult_a = 1-mult_b

    return a*mult_a + b*mult_b


@recursion
def add(a,b,taskinfo) -> torch.Tensor:
    mult = taskinfo['alpha']
    return a + b * mult


@cache_operation
@recursion
def sub(a,b,taskinfo) -> torch.Tensor:
    return a - b


#Bespoke caching and recursion logic to have the difference calculation cached independantly of the adding step
#Allows the weight to be adjusted without having to redo the entire calculation.
def traindiff(taskinfo) -> torch.Tensor:
    source_a,source_b,source_c = taskinfo.sources
    a = source_a()

    new_taskinfo = deepcopy(taskinfo)
    new_taskinfo.args_values = tuple()
    new_taskinfo.update_hash()

    try:
        result = cmn.tensor_cache.retrieve(new_taskinfo)
        return a+result*taskinfo['alpha']
    except KeyError:
        pass

    b = source_b()
    c = source_c()

    ###From https://github.com/hako-mikan/sd-webui-supermerger
    if torch.allclose(b.float(), c.float(), rtol=0, atol=0):
        return a

    diff_AB = b.float() - c.float()

    distance_A0 = torch.abs(b.float() - c.float())
    distance_A1 = torch.abs(b.float() - a.float())

    sum_distances = distance_A0 + distance_A1

    scale = torch.where(sum_distances != 0, distance_A1 / sum_distances, torch.tensor(0.).float())
    sign_scale = torch.sign(b.float() - c.float())
    scale = sign_scale * torch.abs(scale)

    new_diff = scale * torch.abs(diff_AB)
    result = new_diff.to(cmn.precision)  *1.8
    ####

    cmn.tensor_cache.append(new_taskinfo,result)

    return a + result * taskinfo['alpha']


#From https://github.com/hako-mikan/sd-webui-supermerger

@cache_operation
@recursion
def similarity(a: torch.Tensor, b: torch.Tensor, taskinfo):
    return extract_super(None,a,b,taskinfo)

def extract_super(base: torch.Tensor|None,a: torch.Tensor, b: torch.Tensor, taskinfo) -> torch.Tensor:
    alpha = taskinfo['alpha']
    beta = taskinfo['beta']
    gamma = taskinfo['gamma']
    assert base is None or base.shape == a.shape
    assert a.shape == b.shape
    assert 0 <= alpha <= 1
    assert 0 <= beta <= 1
    assert 0 <= gamma
    dtype = base.dtype if base is not None else a.dtype
    base = base.float() if base is not None else 0
    a = a.float() - base
    b = b.float() - base
    c = torch.cosine_similarity(a, b, -1).clamp(-1, 1).unsqueeze(-1)
    d = ((c + 1) / 2) ** (gamma * 25)
    result = base + torch.lerp(a, b, alpha) * torch.lerp(d, 1 - d, beta)
    return result.to(dtype)

extract = recursion(extract_super)

#From https://github.com/hako-mikan/sd-webui-supermerger
@cache_operation
@recursion
def smooth(a,taskinfo):
    # Apply median filter to the differences
    filtered_diff = scipy.ndimage.median_filter(a.detach().cpu().to(torch.float32).numpy(), size=3)
    # Apply Gaussian filter to the filtered differences
    filtered_diff = scipy.ndimage.gaussian_filter(filtered_diff, sigma=1)
    return torch.tensor(filtered_diff,dtype=cmn.precision,device=cmn.device)    
    