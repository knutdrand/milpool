import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class Reparametrization:
    old_to_new: tuple
    new_to_old: tuple


class FReparam(Reparametrization):
    def __init__(self, w, sigma):
        logodds_w = torch.log(w/(1-w))
        self.old_to_new = (
            lambda mu_1, mu_2: (mu_2**2-mu_1**2)/(2*sigma**2)+torch.log(w/(1-w)),
            lambda mu_1, mu_2: (mu_1-mu_2)/sigma**2)
        self.new_to_old = (
            lambda alpha, beta: -(alpha-logodds_w)/beta+beta*sigma**2/2,
            lambda alpha, beta: -(alpha-logodds_w)/beta-beta*sigma**2/2)


def reparametrize(cls, R):
    class NewClass(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.params = tuple(f(*self.params) for f in R.old_to_new)

        def log_likelihood(self, *args):
            n_params = len(self.params)
            param_args = args[-n_params:]
            new_args = [f(*param_args) for f in R.new_to_old]
            return super().log_likelihood(*args[:-n_params], *new_args)

        def estimate_parameters(self, n_samples=1000):
            estimates = torch.tensor(super().estimate_parameters(n_samples))
            new_estimates = np.array([f(*estimates) for f in R.old_to_new])
            # print(estimates, new_estimates, self.params)
            return new_estimates

    NewClass.__name__ = cls.__name__+"RP"
    NewClass.__qualname__ = cls.__qualname__+"RP"
    return NewClass

