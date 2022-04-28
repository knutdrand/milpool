import torch

from .pool import XYMILPool


class Quadratic(torch.nn.Module):

     def __init__(self, n_input=1):
         super().__init__()
         self.linear = torch.nn.Linear(n_input, 1)
         self.quadratic = torch.nn.Linear(n_input, 1, bias=False)

     def forward(self, X):
          return self.linear(X)+self.quadratic(X**2)

class MIL(torch.nn.Module):
     def __init__(self, n_input, instance_model, pooling=XYMILPool()):
         super().__init__()
         self.instance_model = instance_model
         self.pooling = pooling
 
     def forward(self, X):
         I = self.instance_model(X)
         return self.pooling(I)
    
class SimpleMIL(MIL):
    def __init__(self, n_input=1, pooling=XYMILPool()):
        super().__init__(n_input, torch.nn.Linear(n_input, 1), pooling)

class QuadraticMIL(MIL):
    def __init__(self, n_input=1, pooling=XYMILPool()):
        super().__init__(n_input, Quadratic(n_input), pooling)

class ALinearMIL(MIL):
    def __init__(self, n_input, pooling=XYMILPool()):
        alinear = torch.nn.ReLU
        n_hidden=5
        instance_model = torch.nn.Sequential(
            torch.nn.Linear(n_input, n_hidden),
            alinear(),
            torch.nn.Linear(n_hidden, n_hidden),
            alinear(),
            torch.nn.Linear(n_hidden, n_hidden),
            alinear(),
            torch.nn.Linear(n_hidden, n_hidden),
            alinear(),
            torch.nn.Linear(n_hidden, 1))
        super().__init__(n_input, instance_model, pooling)

class AdjustedALinearMIL(ALinearMIL):
     def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self.adjuster = torch.nn.Linear(1, 1)
          
     def forward(self, X):
          logodds = super().forward(X)
          return self.adjuster(logodds)
