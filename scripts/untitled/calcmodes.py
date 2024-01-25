from typing import Any
import scripts.untitled.operators as opr

CALCMODES_LIST = []

class CalcMode:
    name = 'calcmode'
    description = 'description'
    input_models = 3
    input_sliders = 3

    slid_a_info = '-'
    slid_a_config = (-1, 2, 0.01) #minimum,maximum,step

    slid_b_info = '-'
    slid_b_config = (-1, 2, 0.01)

    slid_c_info = '-'
    slid_c_config = (-1, 2, 0.01)

    slid_d_info = '-'
    slid_d_config = (-1, 2, 0.01)

    def create_recipe(self, key, model_a, model_b, model_c, smooth=False, alpha=0, beta=0, gamma=0, delta=0) -> opr.Operation:
        raise NotImplementedError


class WeightSum(CalcMode):
    name = 'Weight-Sum'
    description = 'model_a * (1 - alpha) + model_b * alpha'
    input_models = 2
    input_sliders = 1
    slid_a_info = "model_a - model_b"
    slid_a_config = (0, 1, 0.01)

    def create_recipe(key, model_a, model_b, model_c, alpha=0, **kwargs):
        #This is used when constructing the recipe for the merge, tensors are not handled here.
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)

        if alpha >= 1:
            return b
        elif alpha <= 0:
            return a

        c = opr.Multiply(key, 1-alpha, a)
        d = opr.Multiply(key, alpha, b)
        
        res = opr.Add(key, c, d)
        return res
    
CALCMODES_LIST.append(WeightSum)


class AddDifference(CalcMode):
    name = 'Add Difference'
    description = 'model_a + (model_b - model_c) * alpha'
    input_models = 3
    input_sliders = 1
    slid_a_info = "alpha"
    slid_a_config = (-1, 2, 0.01)

    def create_recipe(key, model_a, model_b, model_c, alpha=0, smooth=False, **kwargs):
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)
        c = opr.LoadTensor(key,model_c)

        diff = opr.Sub(key, b, c)
        if smooth:
            diff = opr.Smooth(key,diff)
        diff.cache()

        diffm = opr.Multiply(key, alpha, diff)

        res = opr.Add(key, a, diffm)
        return res
    
CALCMODES_LIST.append(AddDifference)


class TrainDifference(CalcMode):
    name = 'Train Difference'
    description = 'model_a + (model_b - model_c) * alpha'
    input_models = 3
    input_sliders = 1
    slid_a_info = "alpha"
    slid_a_config = (-1, 2, 0.01)

    def create_recipe(key, model_a, model_b, model_c, alpha=0, smooth=False, **kwargs):
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)
        c = opr.LoadTensor(key,model_c)

        diff = opr.TrainDiff(key,a, b, c)
        if smooth:
            diff = opr.Smooth(key,diff)
        diff.cache()

        diffm = opr.Multiply(key, alpha, diff)

        res = opr.Add(key, a, diffm)
        return res
    
CALCMODES_LIST.append(TrainDifference)


class Extract(CalcMode):
    name = 'Extract'
    description = 'Adds (dis)similar features between (model_b - model_a) and (model_c - model_a) to model_a'
    input_models = 3
    input_sliders = 4
    
    slid_a_info = 'model_b - model_c'
    slid_a_config = (0, 1, 0.01)

    slid_b_info = 'similarity - dissimilarity'
    slid_b_config = (0, 1, 0.01)

    slid_c_info = 'similarity bias'
    slid_c_config = (0, 2, 0.01)

    slid_d_info = 'addition multiplier'
    slid_d_config = (-1, 4, 0.01)

    def create_recipe(key, model_a, model_b, model_c, alpha=0, beta=0, gamma=0, delta=1, smooth=False):
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)
        c = opr.LoadTensor(key,model_c)

        extracted = opr.ExtractRelative(key, alpha, beta, gamma*15, a, b, c)
        if smooth:
            extracted = opr.Smooth(key, extracted)
        extracted.cache()

        multiplied = opr.Multiply(key, delta, extracted)

        res = opr.Add(key, a, multiplied)
        return res

CALCMODES_LIST.append(Extract)


class AddDisimilarity(CalcMode):
    name = 'Add Dissimilarites'
    description = 'Adds dissimalar features between model_b and model_c to model_a'
    input_models = 3
    input_sliders = 3
    
    slid_a_info = 'model_b - model_c'
    slid_a_config = (0, 1, 0.01)

    slid_b_info = 'addition multiplier'
    slid_b_config = (-1, 4, 0.01)

    slid_c_info = 'similarity bias'
    slid_c_config = (0, 2, 0.01)

    def create_recipe(key, model_a, model_b, model_c, alpha=0, beta=0, gamma=0, delta=1, smooth=False):
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)
        c = opr.LoadTensor(key,model_c)

        extracted = opr.Extract(key, alpha, 1, gamma*15, b, c)
        if smooth:
            extracted = opr.Smooth(key, extracted)
        extracted.cache()

        multiplied = opr.Multiply(key, beta, extracted)

        res = opr.Add(key, a, multiplied)
        return res
    
CALCMODES_LIST.append(AddDisimilarity)
        


