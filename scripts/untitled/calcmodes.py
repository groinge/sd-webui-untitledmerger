from typing import Any
import scripts.untitled.operators as opr

CALCMODES_LIST = []

class CalcMode:
    name = 'calcmode'
    description = None
    input_models = 3
    input_sliders = 3

    slid_a_info = ''
    slid_a_config = (-1, 2, 0.01) #minimum,maximum,step

    slid_b_info = ''
    slid_b_config = (-1, 2, 0.01)

    slid_c_info = ''
    slid_c_config = (-1, 2, 0.01)

    def create_recipe(self, key, model_a, model_b, model_c, alpha=0, beta=0, gamma=0) -> opr.Operation:
        pass


class WeightSum(CalcMode):
    name = 'Weight-Sum'
    description = 'model_a * (1 - alpha) + model_b * alpha'
    input_models = 2
    input_sliders = 1
    slid_a_info = "model_a | model_b"
    slid_a_config = (0, 1, 0.01)

    def create_recipe(key, model_a, model_b, model_c, alpha=0, beta=0, gamma=0):
        #This is used when constructing the recipe for the merge, tensors are not handled here.
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)

        c = opr.Multiply(key, 1-alpha, a)
        d = opr.Multiply(key, alpha, b).cache()
        
        res = opr.Add(key, c, d)
        return res
    
CALCMODES_LIST.append(WeightSum)


class AddDifference(CalcMode):
    name = 'Add difference'
    description = 'model_a + (mode_b - model_c) * alpha'
    input_models = 3
    input_sliders = 1
    slid_a_info = "alpha"
    slid_a_config = (-1, 2, 0.01)

    def create_recipe(key, model_a, model_b, model_c, alpha=0, beta=0, gamma=0):
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)
        c = opr.LoadTensor(key,model_c)

        diff = opr.Sub(key, b, c).cache()
        diffm = opr.Multiply(key, alpha, diff)

        res = opr.Add(key, a, diffm)
        return res
    
CALCMODES_LIST.append(AddDifference)


class SmoothAdd(CalcMode):
    name = 'Smooth add'
    description = 'model_a + (mode_b - model_c) * alpha'
    input_models = 3
    input_sliders = 1
    slid_a_info = "alpha"
    slid_a_config = (-1, 4, 0.01)

    def create_recipe(key, model_a, model_b, model_c, alpha=0, beta=0, gamma=0):
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)
        c = opr.LoadTensor(key,model_c)

        diff = opr.Sub(key, b, c)
        diffsmooth = opr.Smooth(key,diff).cache()
        diffm = opr.Multiply(key, alpha, diffsmooth)

        res = opr.Add(key, a, diffm)
        return res
    
CALCMODES_LIST.append(SmoothAdd)

        


