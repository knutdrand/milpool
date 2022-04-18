import torch
import numpy as np
from dataclasses import dataclass
import logging
np.set_printoptions(suppress=True)

#logging.basicConfig(level="DEBUG")
logging.basicConfig(level="INFO")

seed = 13456
def loglik_normal(mu, sigma, x):
    return torch.log(1/torch.sqrt(2*np.pi*sigma**2)) -(x-mu)**2/(2*sigma**2)

def loglik_normal_exp(mu, sigma, x):
    return torch.sum(torch.log(1/torch.sqrt(2*np.pi*sigma**2)) -(x-mu)**2/(2*sigma**2))

def deriv(mu, sigma, x):
    dmu = 2*(x-mu)/(2*sigma**2)
    dsigma = -1/sigma+(x-mu)**2/(sigma**3)
    return dmu, dsigma

def deriv2(mu, sigma, x):
    dmu2 = -1/sigma**2#   2*(x-mu)/(2*sigma**2)
    dsigma2 = 1/sigma**2-3*(x-mu)**2/(sigma**4)
    return dmu2, dsigma2


@dataclass
class Distribution:
    def sample(self, n=100):
        return NotImplemented

    def get_func(self, *x):
        return lambda *params: torch.mean(self.log_likelihood(*x, *params))

    def estimate_fisher_information(self, n=1000000):
        x = self.sample(n)
        f = self.get_func(*x)
        return -np.array(torch.autograd.functional.hessian(f, self.params))# (self.mu, self.sigma)))/n
        

    def l(self, *x):
        return self.log_likelihood(*x, *self.params)

class NormalDistribution(Distribution):
    mu: float=torch.tensor(0.)
    sigma: float=torch.tensor(1.)
    params = (mu, sigma)

    def sample(self, n=1):
        return [torch.normal(self.mu, self.sigma, (n,))]

    def log_likelihood(self, x, mu, sigma):
        return torch.log(1/torch.sqrt(2*np.pi*sigma**2)) -(x-mu)**2/(2*sigma**2)
    

class MixtureXY(Distribution):
    mu_1: float=torch.tensor(0.)
    mu_2: float=torch.tensor(1.)
    sigma: float=torch.tensor(1.)
    w: float=torch.tensor(0.333)
    params = (mu_1, mu_2, sigma, w)

    def sample(self, n=1):
        y = torch.bernoulli(self.w*torch.ones(n))
        mu = self.mu_1*y+self.mu_2*(1-y)
        return torch.normal(mu, self.sigma), y
    
    def l1(self, x):
        mu_1, mu_2, sigma, w = (self.mu_1, self.mu_2, self.sigma, self.w)
        return torch.log(w)+torch.log(1/torch.sqrt(2*np.pi*sigma**2)) -(x-mu_1)**2/(2*sigma**2)

    def l2(self, x):
        mu_1, mu_2, sigma, w = (self.mu_1, self.mu_2, self.sigma, self.w)
        return torch.log(w)+torch.log(1/torch.sqrt(2*np.pi*sigma**2)) -(x-mu_2)**2/(2*sigma**2)

    def log_likelihood(self, x, y, mu_1, mu_2, sigma, w):
        l1 = torch.log(w)+np.log(1/np.sqrt(2*np.pi))-torch.log(sigma) -(x-mu_1)**2/(2*sigma**2)
        l2 = torch.log(1-w)+np.log(1/np.sqrt(2*np.pi))-torch.log(sigma) -(x-mu_2)**2/(2*sigma**2)
        return y*l1+(1-y)*l2


@dataclass
class MixtureX(MixtureXY):
    def sample(self, n=1):
        return super().sample(n)[:1]
        
    def log_likelihood(self, x, mu_1, mu_2, sigma, w):
        L1 = w*(1/np.sqrt(2*np.pi)/sigma*torch.exp(-(x-mu_1)**2/(2*sigma**2)))
        L2 = (1-w)*(1/np.sqrt(2*np.pi)/sigma*torch.exp(-(x-mu_2)**2/(2*sigma**2)))
        return torch.log(L1+L2)

    
class MixtureConditional(MixtureXY):
    def log_likelihood(self, x, y, mu_1, mu_2, sigma, w):
        alpha = (mu_2**2-mu_1**2)/(2*sigma**2)+torch.log(w/(1-w))
        beta = (mu_1-mu_2)/sigma**2
        eta = alpha+beta*x
        return y*torch.log(torch.sigmoid(eta))+(1-y)*torch.log(torch.sigmoid(-eta))

class MixtureY(MixtureXY):
    def sample(self, n=1):
        return super().sample(n)[1:]

    def log_likelihood(self, y, mu_1, mu_2, sigma, w):
        return y*w+(1-y)*(1-w)

class MixtureXgivenY(MixtureXY):
    def log_likelihood(self, y, mu_1, mu_2, sigma, w):
        pass



# print(NormalDistribution().estimate_fisher_information())
x = torch.arange(-10, 10)
y = torch.tensor(5)
print(MixtureX().l(x)+MixtureConditional().l(x, y)/MixtureXY().l(x, y))
# print(MixtureXY().l(x, y))

I_X = MixtureX().estimate_fisher_information()
I_Y = MixtureX().estimate_fisher_information()
I_XY = MixtureXY().estimate_fisher_information()
I_X_g_Y = MixtureConditional().estimate_fisher_information()
# print(I_XY)
print("________")
print("X")
print(I_X)
print("Y|X")
print(I_X_g_Y)
print("SUM")
print(I_X+I_X_g_Y)
print("XY")
print(I_XY)
