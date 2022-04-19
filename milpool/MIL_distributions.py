from .reparametrization import Reparametrization, reparametrize
from .distributions import MixtureXY
import numpy as np
from scipy.special import logsumexp
from numpy import logaddexp
import torch


class MILXY(MixtureXY):
    q: float = torch.tensor(0.5)
    group_size: float = 6

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = tuple(list(self.params) + [self.q])

    def sample(self, n=1):
        y = torch.bernoulli(self.q*torch.ones(n)[:, None])
        z = torch.bernoulli(y*torch.ones(self.group_size)*self.w)
        mu = self.mu_1*z+self.mu_2*(1-z)
        return torch.normal(mu, self.sigma), y

    def log_likelihood(self, x, y, mu_1, mu_2, sigma, w, q):
        L1 = w*(1/np.sqrt(2*np.pi)/sigma*torch.exp(-(x-mu_1)**2/(2*sigma**2)))
        L2 = (1-w)*(1/np.sqrt(2*np.pi)/sigma*torch.exp(-(x-mu_2)**2/(2*sigma**2)))
        l_posY = torch.log(L1+L2).sum(axis=-1)
        l_negY = (np.log(1/np.sqrt(2*np.pi))-torch.log(sigma) - (x-mu_2)**2/(2*sigma**2)).sum(axis=-1)
        y = y.ravel()
        return y*(torch.log(q) + l_posY) + (1-y)*(l_negY+torch.log(1-q))


class MILX(MILXY):
    def sample(self, n=1):
        return super().sample(n)[:1]

    def log_likelihood(self, x, mu_1, mu_2, sigma, w, q):
        L1 = w*(1/np.sqrt(2*np.pi)/sigma*torch.exp(-(x-mu_1)**2/(2*sigma**2)))
        L2 = (1-w)*(1/np.sqrt(2*np.pi)/sigma*torch.exp(-(x-mu_2)**2/(2*sigma**2)))
        l_posY = torch.log(L1+L2).sum(axis=-1)
        l_negY = (np.log(1/np.sqrt(2*np.pi))-torch.log(sigma) - (x-mu_2)**2/(2*sigma**2)).sum(axis=-1)
        return torch.logaddexp(torch.log(q) + l_posY, l_negY+torch.log(1-q))


class MILConditional(MILXY):
    def log_likelihood(self, x, y, mu_1, mu_2, sigma, w, q):
        return MILXY().log_likelihood(x, y, mu_1, mu_2, sigma, w, q)-MILX().log_likelihood(x, mu_1, mu_2, sigma, w, q)


class PureConditional(MILConditional):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mu_1, mu_2, sigma, w, q = self.params
        self.params = ((mu_2**2-mu_1**2)/(2*sigma**2)+torch.log(w/(1-w)),
                       (mu_1-mu_2)/sigma**2, w, q)

    def log_likelihood(self, x, y, alpha, beta, w, q):
        eta = alpha+beta*x
        p = torch.sigmoid(eta)
        # l_pos = torch.log(q) + torch.sum(
        #     torch.log(torch.sigmoid(eta)*w/(q*w)+torch.sigmoid(-eta)*(1-w)/(1-q*w)), axis=-1)
        # l_pos = torch.log(q) + torch.sum(
        #     torch.log(torch.sigmoid(eta)*w/(q*w)+torch.sigmoid(-eta)*(1-w)/(1-q*w)), axis=-1)
        inside_the_sum = torch.logaddexp(torch.log(1-w), torch.log(1/w*q-2-w) + torch.log(p))
        inside_the_sum = torch.logaddexp(torch.log(1-p)+torch.log(1-w), torch.log(p)+torch.log(1/q-w)-1)
        l_pos = torch.log(q) + torch.sum(inside_the_sum, axis=-1)
        l_neg = torch.log(1-q) + torch.sum(torch.log(1-p), axis=-1)
        print(l_pos.mean(axis=-1), l_neg.mean(axis=-1))
        #l_neg = torch.log(1-q) + torch.sum(torch.log(torch.sigmoid(-eta)/(1-q*w)), axis=-1)
        l = torch.logaddexp(l_pos, l_neg)
        print(l.mean(axis=-1))
        y = y.ravel()
        total =  y*l_pos+(1-y)*l_neg-l
        print(total.mean())
        return total

dists = (MILX, MILXY, MILConditional)

MidPointReparam = Reparametrization(
    old_to_new=(lambda mu_0, mu_1, sigma, w, q: (mu_0+mu_1)/2-sigma**2*torch.log(w/(1-w))/(2*(mu_0-mu_1)),
                lambda mu_0, mu_1, _, __, q: (mu_0-mu_1),
                lambda _, __, sigma, w, q: sigma**2*torch.log(w/(1-w)),
                lambda _, __, sigma, w, q: sigma**2,
                lambda _, __, sigma, w, q: q),
    new_to_old=(lambda eta_0, eta_1, eta_2, eta_3, q: eta_0 + eta_2/(2*eta_1) + eta_1/2,
                lambda eta_0, eta_1, eta_2, eta_3, q: eta_0 + eta_2/(2*eta_1) -eta_1/2,
                lambda eta_0, eta_1, eta_2, eta_3, q: torch.sqrt(eta_3),
                lambda eta_0, eta_1, eta_2, eta_3, q: torch.sigmoid(eta_2/eta_3),
                lambda eta_0, eta_1, eta_2, eta_3, q: q))

PureMidPointReparam = Reparametrization(
    old_to_new=(lambda alpha, beta, w, q: -alpha/beta,
                lambda alpha, beta, w, q: beta,
                lambda alpha, beta, w, q: w,
                lambda alpha, beta, w, q: q),
    new_to_old=(lambda eta_0, eta_1, w, q: -eta_0*eta_1,
                lambda eta_0, eta_1, w, q: eta_1,
                lambda eta_0, eta_1, w, q: w,
                lambda eta_0, eta_1, w, q: q))


reparam_dists = tuple(reparametrize(cls, MidPointReparam) for cls in dists)
RPPureConditional = reparametrize(PureConditional, PureMidPointReparam)
