import torch
import numpy as np
from dataclasses import dataclass
from .reparametrization import Reparametrization, reparametrize
from sklearn.linear_model import LogisticRegression


@dataclass
class Distribution:
    def sample(self, n=100):
        return NotImplemented

    def get_func(self, *x):
        return lambda *params: torch.mean(self.log_likelihood(*x, *params))

    def estimate_fisher_information(self, n=10000000):
        x = self.sample(n)
        f = self.get_func(*x)
        return -np.array(torch.autograd.functional.hessian(f, self.params))# (self.mu, self.sigma)))/n

    def l(self, *x):
        return self.log_likelihood(*x, *self.params)


class NormalDistribution(Distribution):
    mu: float = torch.tensor(0.)
    sigma: float = torch.tensor(1.)
    params = (mu, sigma)

    def sample(self, n=1):
        return [torch.normal(self.mu, self.sigma, (n,))]

    def log_likelihood(self, x, mu, sigma):
        return torch.log(1/torch.sqrt(2*np.pi*sigma**2)) -(x-mu)**2/(2*sigma**2)


class MixtureXY(Distribution):
    mu_1: float=torch.tensor(0.)
    mu_2: float=torch.tensor(1.)
    sigma: float=torch.tensor(3.)
    w: float=torch.tensor(0.66666)
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

    def get_square_errors(self, n_samples=1000, n_iterations=1000):
        estimates = np.array([self.estimate_parameters(n_samples) for _ in range(n_iterations)])
        true_params = np.array(self.params)
        return ((estimates-true_params)**2).sum(axis=0)/n_iterations

    def estimate_parameters(self, n=1000):
        x, y = self.sample(n)
        x = np.array(x)
        y = np.array(y)

        group_2 = x[y == 0]
        group_1 = x[y == 1]
        mu_1 = np.mean(group_1)
        mu_2 = np.mean(group_2)
        sigma = np.sqrt((np.sum((group_1-mu_1)**2) + np.sum((group_2-mu_2)**2))/x.size)
        w = group_1.size/y.size
        return (mu_1, mu_2, sigma, w)


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


class PureMixtureConditional(MixtureConditional):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mu_1, mu_2, sigma, w = self.params
        self.params = ((mu_2**2-mu_1**2)/(2*sigma**2)+torch.log(w/(1-w)),
                       (mu_1-mu_2)/sigma**2)

    def log_likelihood(self, x, y, alpha, beta):
        eta = alpha+beta*x
        return y*torch.log(torch.sigmoid(eta))+(1-y)*torch.log(torch.sigmoid(-eta))

    def estimate_parameters(self, n_samples=1000):
        x, y = self.sample(n_samples)
        lr = LogisticRegression(penalty='none')
        lr.fit(x[:, None], y)
        return np.array([lr.intercept_[0], lr.coef_[0,0]])
        return np.array(lr.intercept_, lr.coef_[0])


class MixtureY(MixtureXY):
    def sample(self, n=1):
        return super().sample(n)[1:]

    def log_likelihood(self, y, mu_1, mu_2, sigma, w):
        return y*w+(1-y)*(1-w)


class MixtureXgivenY(MixtureXY):
    def log_likelihood(self, y, mu_1, mu_2, sigma, w):
        pass


full_triplet = (MixtureX, MixtureXY, MixtureConditional)
MidPointReparam = Reparametrization(
    old_to_new=(lambda mu_0, mu_1, sigma, w: (mu_0+mu_1)/2-sigma**2*torch.log(w/(1-w))/(2*(mu_0-mu_1)),
                lambda mu_0, mu_1, _, __: (mu_0-mu_1),
                lambda _, __, sigma, w: sigma**2*torch.log(w/(1-w)),
                lambda _, __, sigma, w: sigma**2),
    new_to_old=(lambda eta_0, eta_1, eta_2, eta_3: eta_0 + eta_2/(2*eta_1) + eta_1/2,
                lambda eta_0, eta_1, eta_2, eta_3: eta_0 + eta_2/(2*eta_1) -eta_1/2,
                lambda eta_0, eta_1, eta_2, eta_3: torch.sqrt(eta_3),
                lambda eta_0, eta_1, eta_2, eta_3: torch.sigmoid(eta_2/eta_3)))
                
rp_full_triplet = tuple(reparametrize(cls, MidPointReparam) for cls in full_triplet)

PureMidPointReparam = Reparametrization(
    old_to_new=(lambda alpha, beta: -alpha/beta,
                lambda alpha, beta: beta),
    new_to_old=(lambda eta_0, eta_1: -eta_0*eta_1,
                lambda eta_0, eta_1: eta_1))
PureMixtureConditionalRP = reparametrize(PureMixtureConditional, PureMidPointReparam)
