import torch
import numpy as np
from .distributions import Distribution, MixtureDistribution
from .poissonmixture import PoissonMixture, PoissonMIL
# from scipy.stats import poisson


class PoissonDistribution(Distribution):
    mu: torch.tensor = 1
    log_factorial = torch.cumsum(torch.log(torch.arange(1, 1000)), dim=0)

    def __init__(self, mu=1):
        self.mu = torch.as_tensor(mu)
        self.params = (self.mu, )

    def sample(self, n):
        mu = torch.tile(self.mu, (n, 1))
        return [torch.poisson(mu)]

    def log_likelihood(self, k, mu):
        k = k.long()
        v = (k*torch.log(mu)-mu)
        u = -self.log_factorial[k]
        return (v+u).sum(axis=-1)
    # return v# +u

    def estimate_parameters(self, n_samples=1000):
        x = self.sample(n_samples)[0]
        mu = torch.mean(x, axis=0)
        return (mu, )

    def _get_x_for_plotting(self):
        m = (self.mu*+2*torch.sqrt(self.mu)).long()
        return torch.arange(int(m+1))


class PoissonMixtureDistribution(MixtureDistribution):
    def estimate_parameters(self, n=1000):
        s = self.sample(n)
        x = s[0]
        model = PoissonMixture(n_components=2)
        model.fit(x)
        return (np.concatenate([np.sort(model.means_, axis=0).ravel(),
                                np.sort(model.weights_)]),)

    def _get_x_for_plotting(self):
        return max((d._get_x_for_plotting() for d in self._distributions), key=len)


class MILDistribution:
    def __init__(self, pos_dist, neg_dist, w, q, group_size):
        self._pos_dist = pos_dist
        self._neg_dist = neg_dist
        self._w = torch.as_tensor(w)
        self._q = torch.as_tensor(q)
        self.params = (torch.hstack([pos_dist.params[0], neg_dist.params[0], self._w, self._q]))
        self._group_size = group_size

    def sample(self, n=1):
        y = torch.bernoulli(self._q*torch.ones(n)[:, None])
        z = torch.bernoulli(y*torch.ones(self._group_size)*self._w)
        n_pos = int(z.sum())
        X_pos = self._pos_dist.sample(n_pos)[0]
        X_neg = self._neg_dist.sample(self._group_size*n-n_pos)[0]
        X = torch.empty((n, self._group_size, X_pos.shape[-1]))
        X[z.bool()] = X_pos
        X[~z.bool()] = X_neg
        return X, y

    def estimate_parameters(self, n=100):
        X, y = self.sample(n)
        model = PoissonMIL(n_components=2)
        model.fit(np.array(X), np.array(y))
        return model.means_, model.weights_
