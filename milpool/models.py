import torch

from .pool import XYMILPool

class SimpleMIL(torch.nn.Module):
    def __init__(self, n_input=1, pooling=XYMILPool()):
        super().__init__()
        self.linear = torch.nn.Linear(n_input, 1)
        self.pooling = pooling

    def forward(self, X):
        I = self.linear(X)
        return self.pooling(I)

# class MIL(torch.nn.Module):
#     def __init__(self, n_input=1, instance_model, pooling=XYMILPool()):
#         super().__init__()
#         self.instance_model = instance_model
#         self.pooling = pooling
# 
#     def forward(self, X):
#         I = self.linear(X)
#         return self.pooling(I)
    
