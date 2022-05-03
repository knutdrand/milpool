import torch
import numpy as np
from .distributions import Distribution
# from scipy.stats import poisson


class PoissonDistribution(Distribution):
    mu: torch.tensor = 1
    log_factorial = np.cumsum(np.log(np.arange(1, 1000)))

    def __init__(self, mu=1):
        self.mu = torch.as_tensor(mu)
        self.params = (self.mu, )

    def sample(self, n):
        return [torch.poisson(torch.full((n, ), self.mu))]

    def log_likelihood(self, k, mu):
        k = k.long()
        v = (k*torch.log(mu)-mu)
        u = -self.log_factorial[k]
        print(v, u)
        return v# +u

    def estimate_parameters(self, n_samples=1000):
        x = self.sample(n_samples)[0]
        mu = torch.mean(x, axis=0)
        return (mu, )

    def _get_x_for_plotting(self):
        m = (self.mu*+5*torch.sqrt(self.mu)).long()
        return torch.arange(m+1)
