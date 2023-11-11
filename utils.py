import torch 
import numpy as np 
import pandas as pd 

def print_number_of_trainable_params(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, params in model.named_parameters():
        all_model_params += params.numel()
        if params.requires_grad:
            trainable_model_params += params.numel()
    
    ret = f'trainable_model_params: {trainable_model_params}\n'
    ret += f'all model params: {all_model_params}\n'
    ret += f'\% trainable: {100 * trainable_model_params / all_model_params}%'
    return ret 

