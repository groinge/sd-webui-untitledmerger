import torch
import scripts.common as cmn

def load_tensor(taskinfo):
    key = taskinfo.key
    fname = taskinfo['filename']
    return cmn.loaded_checkpoints[fname].get_tensor(key).type(cmn.precision)


def weight_sum(a,b,taskinfo):
    mult_b = taskinfo['multiplier']
    mult_a = 1-mult_b

    result = a*mult_a + b*mult_b
    return result

cmn.operators['weight_sum'] = weight_sum


def add(a,b,taskinfo):
    mult = taskinfo['multiplier']
    return a + b * mult

cmn.operators['add'] = add


def sub(a,b,taskinfo):
    return a - b

cmn.operators['sub'] = sub


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

cmn.operators['traindiff'] = traindiff

