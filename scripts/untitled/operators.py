import torch,scipy,cachetools
import scripts.untitled.common as cmn


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
        self.merge_func = recurse

    def __eq__(self, other):
        return (self.key, self.alpha, self.beta, self.gamma, self.sources) == (other.key, other.alpha, other.beta, other.gamma, other.sources)
    
    def __hash__(self):
        return hash((self.key, self.alpha, self.beta, self.gamma, self.sources))
    
    def oper(self,*args):
        raise NotImplementedError

    def merge(self):
        return self.merge_func(self)
    
    def cache(self):
        self.merge_func = cache_operation(recurse)
        return self
        

class LoadTensor(Operation):
    def __init__(self,key,alpha):
        super().__init__(key,*tuple())
        self.alpha = alpha

    #loadtensor uses merge instead of oper as it has no model inputs, use oper everywhere else 
    def merge(self) -> torch.Tensor:
        return cmn.loaded_checkpoints[self.alpha].get_tensor(self.key).to(cmn.device)


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
        return torch.tensor(filtered_diff,dtype=cmn.precision,device=cmn.device)
    

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
        return new_diff.to(cmn.precision)  *1.8
        

class ExtractRelative(Operation):
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
    

class Extract(ExtractRelative):
    def __init__(self,*args):
        super().__init__(*args)

    def oper(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return super().oper(None,a,b)
    

#The cache
class WeightsCache(cachetools.LRUCache):
    def __init__(self, size):
        super().__init__(size,lambda x: x.element_size() * x.nelement())

    def __setitem__(self, key, value):
        value = value.detach().cpu()
        super().__setitem__(key,value)

    def __getitem__(self, key: Operation) -> torch.Tensor:
        res = super().__getitem__(key)
        return res.clone().to(cmn.device).type(cmn.precision)

weights_cache = WeightsCache(cmn.cache_size)