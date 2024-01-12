import torch
import scripts.common as cmn

def load_tensor(taskinfo):
    key = taskinfo.key
    fname = taskinfo['filename']
    return cmn.loaded_checkpoints[fname].get_tensor(key)


def weight_sum(a,b,taskinfo):
    mult_b = taskinfo['alpha']
    mult_a = 1-mult_b

    return a*mult_a + b*mult_b


def add(a,b,taskinfo):
    mult = taskinfo['alpha']
    return a + b * mult


def sub(a,b,taskinfo):
    return a - b


#From https://github.com/hako-mikan/sd-webui-supermerger
def traindiff(a,b,c,taskinfo):
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
    return new_diff *1.8


def extract_super(a: torch.Tensor, b: torch.Tensor, taskinfo) -> torch.Tensor:
    base = None
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
    d = ((c + 1) / 2) ** gamma
    result = base + torch.lerp(a, b, alpha) * torch.lerp(d, 1 - d, beta)
    return result.to(dtype)