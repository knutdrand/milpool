import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class Reparametrization:
    old_to_new: tuple
    new_to_old: tuple


class NpReparametrization:
    def old_to_new(self, old_params):
        return torch.stack(self._old_to_new(old_params))

    def new_to_old(self, new_params):
        return torch.stack(self._new_to_old(new_params))

    def _get_jacobian(self, params):
        print(params)
        return torch.autograd.functional.jacobian(self.new_to_old, params)

    def old_fisher_to_new(self, I, params):
        J = self._get_jacobian(params)
        print(J)
        return J.T @ I @ J


class NpFReparam(NpReparametrization):
    def __init__(self, w, sigma):
        self.logodds_w = torch.log(w/(1-w))
        self.sigma = sigma

    def old_to_new(self, old_params):
        mu_1, mu_2 = old_params
        return torch.stack([(mu_2**2-mu_1**2)/(2*self.sigma**2)+self.logodds_w,
                            (mu_1-mu_2)/self.sigma**2])

    def new_to_old(self, new_params):
        alpha, beta = new_params
        return torch.stack([-(alpha-self.logodds_w)/beta+beta*self.sigma**2/2,
                            -(alpha-self.logodds_w)/beta-beta*self.sigma**2/2])


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
            return new_estimates

    NewClass.__name__ = cls.__name__+"RP"
    NewClass.__qualname__ = cls.__qualname__+"RP"
    return NewClass


def np_reparametrize(cls, R):
    class NewClass(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            print(self.params)
            self.params = tuple(R.old_to_new(self.params))

        def log_likelihood(self, *args):
            n_params = len(self.params)
            param_args = args[-n_params:]
            new_args = R.new_to_old(param_args)
            return super().log_likelihood(*args[:-n_params], *new_args)

        def estimate_parameters(self, n_samples=1000):
            estimates = torch.tensor(super().estimate_parameters(n_samples))
            new_estimates = np.array(R.new_to_old(estimates))
            return new_estimates

    NewClass.__name__ = cls.__name__+"npRP"
    NewClass.__qualname__ = cls.__qualname__+"npRP"
    return NewClass
