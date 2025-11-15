import torch
import torch.nn as nn
import torch.nn.functional as F

class LGRA():  
    def __init__(self, args, inputs_dim):
        self.args = args
        self.inputs_dim = inputs_dim

        
