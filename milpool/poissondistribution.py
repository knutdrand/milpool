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


class MILDistribution(Distribution):
    def __init__(self, pos_dist, neg_dist, w, q, group_size):
        self._pos_dist = pos_dist
        self._neg_dist = neg_dist
        self._w = torch.as_tensor(w)
        self._q = torch.as_tensor(q)
        self.params = (torch.hstack([pos_dist.params[0], neg_dist.params[0], self._w, self._q]),)
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

    def log_likelihood(self, X, y, *params):
        params = params[0]
        w, q = params[-2:]
        n_p = int((len(params)-2)/2)
        neg_lik = self._neg_dist.log_likelihood(X, *(params[n_p:2*n_p], ))
        pos_lik = self._pos_dist.log_likelihood(X, params[:n_p])
        comb_lik = torch.logaddexp(torch.log(w) + pos_lik,
                                   torch.log(1-w) + neg_lik)
        return y.ravel()*(torch.log(q) + comb_lik.sum(axis=-1))+(1-y.ravel())*(torch.log(1-q) + neg_lik.sum(axis=-1))

    def estimate_parameters(self, n=100):
        X, y = (np.array(d) for d in self.sample(n))
        model = PoissonMIL(n_components=2)
        model.fit(X, y)
        return (np.concatenate((
            model.means_.ravel(), model.weights_.ravel()[[0]], [y.sum()/y.size])),)
