import torch,scipy,cachetools
import scripts.untitled.common as cmn
import torch.nn.functional as F
import numpy as np


#Wrappers

def recurse(operation):
    source_tensors = []
    for source_oper in operation.sources:
        source_tensor = source_oper.merge()
        source_tensors.append(source_tensor)

    return operation.oper(*source_tensors)

def cache_operation(func):
    def inner(operation):
        try:
            return weights_cache[operation]
        except KeyError:pass

        result = func(operation)

        weights_cache[operation] = result
        return result
    return inner


###OPERATORS####

class Operation:
    def __init__(self,key,*sources):
        self.key = key
        self.sources = tuple(sources)
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.delta = None
        self.seed = None
        self.merge_func = recurse

    def __eq__(self, other):
        return (self.key, self.alpha, self.beta, self.gamma, self.delta, self.seed, self.sources) == (other.key, other.alpha, other.beta, other.gamma, other.delta, other.seed, other.sources)
    
    def __hash__(self):
        return hash((self.key, self.alpha, self.beta, self.gamma, self.delta, self.seed, self.sources))
    
    def oper(self,*args) -> torch.Tensor:
        raise NotImplementedError

    def merge(self):
        return self.merge_func(self)
    
    def cache(self):
        if cmn.opts['cache_size'] > 512:
            self.merge_func = cache_operation(recurse)
        return self
        

class LoadTensor(Operation):
    def __init__(self,key,alpha):
        super().__init__(key,*tuple())
        self.alpha = alpha

    #loadtensor uses merge instead of oper as it has no model inputs, use oper everywhere else 
    def merge(self) -> torch.Tensor:
        return cmn.loaded_checkpoints[self.alpha].get_tensor(self.key).to(cmn.device())


class Multiply(Operation):
    def __init__(self,key,alpha,*sources):
        super().__init__(key,*sources)
        self.alpha = alpha

    def oper(self,a) -> torch.Tensor:
        return a * self.alpha


class Add(Operation):
    def __init__(self,*args):
        super().__init__(*args)

    def oper(self,a,b) -> torch.Tensor:
        return a + b


class Sub(Operation):
    def __init__(self,*args):
        super().__init__(*args)

    def oper(self,a,b) -> torch.Tensor:
        return a - b


class Smooth(Operation):
    def __init__(self,*args):
        super().__init__(*args)

    ###From https://github.com/hako-mikan/sd-webui-supermerger
    def oper(self,a) -> torch.Tensor:
        # Apply median filter to the differences
        filtered_diff = scipy.ndimage.median_filter(a.detach().cpu().to(torch.float32).numpy(), size=3)
        # Apply Gaussian filter to the filtered differences
        filtered_diff = scipy.ndimage.gaussian_filter(filtered_diff, sigma=1)
        return torch.tensor(filtered_diff,dtype=cmn.dtype(),device=cmn.device())
    

class TrainDiff(Operation):
    def __init__(self,*args):
        super().__init__(*args)

    ###From https://github.com/hako-mikan/sd-webui-supermerger
    def oper(self, a, b, c) -> torch.Tensor:
        if torch.allclose(b.float(), c.float(), rtol=0, atol=0):
            return torch.zeros_like(a)

        diff_AB = b.float() - c.float()

        distance_A0 = torch.abs(b.float() - c.float())
        distance_A1 = torch.abs(b.float() - a.float())

        sum_distances = distance_A0 + distance_A1

        scale = torch.where(sum_distances != 0, distance_A1 / sum_distances, torch.tensor(0.).float())
        sign_scale = torch.sign(b.float() - c.float())
        scale = sign_scale * torch.abs(scale)

        new_diff = scale * torch.abs(diff_AB)
        return new_diff.to(cmn.dtype())  *1.8
        

class Extract(Operation):
    def __init__(self,key,alpha,beta,gamma,*args):
        super().__init__(key,*args)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    #From https://github.com/hako-mikan/sd-webui-supermerger
    def oper(self, base: torch.Tensor|None,a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert base is None or base.shape == a.shape
        assert a.shape == b.shape
        assert 0 <= self.alpha <= 1
        assert 0 <= self.beta <= 1
        assert 0 <= self.gamma
        dtype = base.dtype if base is not None else a.dtype
        base = base.float() if base is not None else 0
        a = a.float() - base
        b = b.float() - base
        c = torch.cosine_similarity(a, b, -1).clamp(-1, 1).unsqueeze(-1)
        d = ((c + 1) / 2) ** self.gamma
        result = torch.lerp(a, b, self.alpha) * torch.lerp(d, 1 - d, self.beta)
        return result.to(dtype)
    

class Similarities(Extract):
    def __init__(self,*args):
        super().__init__(*args)

    def oper(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return super().oper(None,a,b)


class PowerUp(Operation):
    def __init__(self,key,alpha,*sources):
        super().__init__(key,*sources)
        self.alpha = alpha

    #https://github.com/martyn/safetensors-merge-supermario/blob/main/merge.py
    #https://arxiv.org/pdf/2311.03099.pdf
    #https://github.com/yule-BUAA/MergeLM/tree/main/model_merging_methods
    def oper(self, a, b):
        # Calculate the delta of the weights
        a, b = resize_tensors(a, b)
        delta = b - a
        # Generate the mask m^t from Bernoulli distribution
        m = torch.empty_like(delta,dtype=cmn.dtype()).uniform_(0,1) < self.alpha
        # Apply the mask to the delta to get δ̃^t
        delta_tilde = m * delta
        # Scale the masked delta by the dropout rate to get δ̂^t
        delta_hat = delta_tilde / (1 - self.alpha)
        return delta_hat
    

def resize_tensors(tensor1, tensor2):
    if len(tensor1.shape) not in [1, 2]:
        return tensor1, tensor2

    # Pad along the last dimension (width)
    if tensor1.shape[-1] < tensor2.shape[-1]:
        padding_size = tensor2.shape[-1] - tensor1.shape[-1]
        tensor1 = F.pad(tensor1, (0, padding_size, 0, 0))
    elif tensor2.shape[-1] < tensor1.shape[-1]:
        padding_size = tensor1.shape[-1] - tensor2.shape[-1]
        tensor2 = F.pad(tensor2, (0, padding_size, 0, 0))

    # Pad along the first dimension (height)
    if tensor1.shape[0] < tensor2.shape[0]:
        padding_size = tensor2.shape[0] - tensor1.shape[0]
        tensor1 = F.pad(tensor1, (0, 0, 0, padding_size))
    elif tensor2.shape[0] < tensor1.shape[0]:
        padding_size = tensor1.shape[0] - tensor2.shape[0]
        tensor2 = F.pad(tensor2, (0, 0, 0, padding_size))

    return tensor1, tensor2


class ShuffleTensor(Operation):
    def __init__(self,key,alpha,*sources):
        super().__init__(key,*sources)
        self.alpha = alpha

    def oper(self, a, b):
        bitmask = torch.empty(a,dtype=cmn.dtype()).uniform_(0,1) > self.alpha
        res = a * bitmask + b * ~bitmask
        return res


class InterpolateDifference(Operation):
    def __init__(self,key,alpha,beta,gamma,seed,*sources):
        super().__init__(key,*sources)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seed = seed

    def oper(self, a, b):
        delta = torch.abs(a - b)

        if self.beta != 1:
            diff = (delta / torch.max(delta)) ** self.alpha
        else:
            diff = (1 - delta / torch.max(delta)) ** self.alpha

        rngenerator = torch.Generator(device=diff.device)
        rngenerator.manual_seed(self.seed)
        bitmask = torch.bernoulli(torch.clamp(diff,0,1),out=torch.empty_like(diff),generator=rngenerator)

        interpolated_mask = torch.lerp(bitmask, diff, self.gamma).to(a.dtype)

        res = a * interpolated_mask  + b * (1 - interpolated_mask)
        return res

#The cache
class WeightsCache(cachetools.LRUCache):
    def __init__(self, size):
        capped = min(size, 8192)
        super().__init__(capped*1024*1024,lambda x: x.element_size() * x.nelement())

    def __setitem__(self, key, value):
        value = value.detach().cpu()
        super().__setitem__(key,value)

    def __getitem__(self, key: Operation) -> torch.Tensor:
        res = super().__getitem__(key)
        return res.clone().to(cmn.device()).type(cmn.dtype())
    
    
weights_cache = WeightsCache(4096)


