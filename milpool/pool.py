from torch.autograd import Function

import torch
class SetPooling(torch.nn.Module):
    def _pre_calc(self, I):
        pass

    def _post_calc(self, S):
        pass
    
    def forward(self, X):
        psi = self._pre_calc(X)
        S = psi.sum(axis=-2)
        return self._post_calc(S)

class XYPsi(Function):
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.logaddexp(input, torch.zeros(1))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output*torch.sigmoid(input)#(torch.logaddexp(-input, torch.zeros(1)))

class XYEta(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        m = torch.clamp(input, min=0)
        return m+torch.log(torch.exp(input-m)-torch.exp(-m))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors 
        return grad_output/(1-torch.exp(-input))


class XYMILPool(SetPooling):
    def _pre_calc(self, I):
        return XYPsi.apply(I)
        # return torch.log(1+torch.exp(I))

    def _post_calc(self, S):
        return XYEta.apply(S)
        # return torch.log(torch.exp(S)-1)


class YXPsi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, logow, q=torch.tensor(0.5)):
        w = torch.sigmoid(logow)
        ctx.save_for_backward(input, w, q)
        A = torch.log(1-w)
        B = torch.log(1-q*w)-torch.log(q)
        return torch.logaddexp(A, input+B)

    @staticmethod
    def backward(ctx, grad_output):
        input, w, q = ctx.saved_tensors
        d_input = grad_output/((1-w)*q/(1-q*w)*torch.exp(-input)+1)
        # print(d_input.max(), d_input.min())
        d_w = -grad_output/(1-(q-1)/q*torch.sigmoid(input)+w)
        d_logow = d_w*(1-w)*w
        return d_input, d_logow.sum(0), None

class SimpleYXPsi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, logow):
        w = torch.sigmoid(logow)
        ctx.save_for_backward(input, w)
        A = torch.log(1-w)
        B = torch.log(w)
        return torch.logaddexp(A, input+B)

    @staticmethod
    def backward(ctx, grad_output):
        input, w= ctx.saved_tensors
        d_input = grad_output/((1-w)/w*torch.exp(-input)+1)
        # d_w = -grad_output/(1-(q-1)/q*torch.sigmoid(input)+w)
        d_w = grad_output/(1/(torch.exp(input)-1)+w)
        d_logow = d_w*(1-w)*w
        return d_input, d_logow.sum(0), None

class YXMILPool(SetPooling):
    def __init__(self, w=None):
        super().__init__()
        self.q = torch.tensor([0.5])
        rg = w is None
        if w is None:
           w = 0.1 
        self.logOw = torch.nn.parameter.Parameter(torch.logit(torch.tensor([w])), requires_grad=rg)

    def _pre_calc(self, I):
        return YXPsi().apply(I, self.logOw, self.q)
        # return SimpleYXPsi().apply(I, self.logOw)

    def _post_calc(self, S):
        return torch.log(self.q/(1-self.q))+S

# def max_pool(n):
#     mp = torch.nn.MaxPool1d(n)
#     return lambda X: mp(torch.squeeze(X, -1))
# 

class MaxPool(torch.nn.Module):
    def forward(self, X):
        return torch.max(X, axis=-2)[0]

class SoftMaxPool(torch.nn.Module):
    def forward(self, X):
        return (torch.nn.Softmax(dim=-2)(X)*X).sum(axis=-2)
class AvgPool(torch.nn.Module):
    def forward(self, X):
        return torch.mean(X, axis=-2)
