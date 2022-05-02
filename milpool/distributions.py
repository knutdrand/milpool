import torch
import numpy as np
from dataclasses import dataclass
from .reparametrization import Reparametrization, reparametrize, NpReparametrization
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


@dataclass
class Distribution:
    def sample(self, n=100):
        return NotImplemented

    def get_func(self, *x):
        return lambda *params: torch.mean(self.log_likelihood(*x, *params))

    def estimate_fisher_information(self, n=10000000):
        x = self.sample(n)
        f = self.get_func(*x)
        return -np.array(torch.autograd.functional.hessian(f, self.params))  # (self.mu, self.sigma)))/n

    def prob(self, *x):
        return np.exp(self.log_likelihood(*x, *self.params))


class NormalDistribution(Distribution):
    mu: float = torch.tensor(0.)
    sigma: float = torch.tensor(1.)
    params = (mu, sigma)

    def __init__(self, mu, sigma):
        self.mu = torch.tensor(mu)
        self.sigma = torch.tensor(sigma)
        self.params = (mu, sigma)

    def sample(self, n=1):
        return [torch.normal(self.mu, self.sigma, (n,))]

    def log_likelihood(self, x, mu, sigma):
        return torch.log(1/torch.sqrt(2*np.pi*sigma**2)) -(x-mu)**2/(2*sigma**2)


class NormalNaturalReparam(NpReparametrization):
    def _old_to_new(self, theta):
        mu, sigma = theta
        return [mu/sigma**2, -1/(2*sigma**2)]

    def _new_to_old(self, theta):
        return [-theta[0]/(2*theta[1]), torch.sqrt(-1/(2*theta[1]))]


class MixtureXY(Distribution):
    # mu_1: float=torch.tensor(-1.)
    # mu_2: float=torch.tensor(1.)
    # sigma: float=torch.tensor(0.5)
    # w: float=torch.tensor(0.5)
    # params = (mu_1, mu_2, sigma, w)

    def __init__(self, mu_1=torch.tensor(0.), mu_2=torch.tensor(1.), sigma=torch.tensor(1.), w=torch.tensor(0.3333)):
        self.mu_1 = torch.tensor(mu_1)
        self.mu_2 = torch.tensor(mu_2)
        self.sigma = torch.tensor(sigma)
        self.w = torch.tensor(w)
        self.params = (self.mu_1, self.mu_2, self.sigma, self.w)

    def sample(self, n=1):
        y = torch.bernoulli(self.w*torch.ones(n))
        mu = self.mu_1*y+self.mu_2*(1-y)
        return torch.normal(mu, self.sigma), y
    
    def l1(self, x, mu_1, mu_2, sigma, w):
        # mu_1, mu_2, sigma, w = (self.mu_1, self.mu_2, self.sigma, self.w)
        return torch.log(w)+np.log(1/np.sqrt(2*np.pi))-torch.log(sigma) -(x-mu_1)**2/(2*sigma**2)

    def l2(self, x, mu_1, mu_2, sigma, w):
        # mu_1, mu_2, sigma, w = (self.mu_1, self.mu_2, self.sigma, self.w)
        return torch.log(1-w)+np.log(1/np.sqrt(2*np.pi))-torch.log(sigma) -(x-mu_2)**2/(2*sigma**2)
        # torch.log(1-w)+torch.log(1/torch.sqrt(2*np.pi*sigma**2)) -(x-mu_2)**2/(2*sigma**2)

    def log_likelihood(self, x, y, mu_1, mu_2, sigma, w):
        l1 = self.l1(x, mu_1, mu_2, sigma, w)
        # torch.log(w)+np.log(1/np.sqrt(2*np.pi))-torch.log(sigma) -(x-mu_1)**2/(2*sigma**2)
        l2 = self.l2(x, mu_1, mu_2, sigma, w)
        # torch.log(1-w)+np.log(1/np.sqrt(2*np.pi))-torch.log(sigma) -(x-mu_2)**2/(2*sigma**2)
        return y*l1+(1-y)*l2

    def get_square_errors(self, n_samples=1000, n_iterations=1000, do_plot=False):
        estimates = np.array([self.estimate_parameters(n_samples) for _ in range(n_iterations)])
        true_params = np.array(self.params)
        if do_plot:
            for i, param in enumerate(true_params):
                plt.hist(estimates[:, i])
                plt.axvline(x=param)
                plt.title(f"n={n_samples}")
                plt.show()
        
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


class MixtureX(MixtureXY):
    def sample(self, n=1):
        return super().sample(n)[:1]

    def log_likelihood(self, x, mu_1, mu_2, sigma, w):
        l1 = self.l1(x, mu_1, mu_2, sigma, w)
        l2 = self.l2(x, mu_1, mu_2, sigma, w)
        # torch.log(w)+np.log(1/np.sqrt(2*np.pi))-torch.log(sigma) -(x-mu_1)**2/(2*sigma**2)
        # l2 = torch.log(1-w)+np.log(1/np.sqrt(2*np.pi))-torch.log(sigma) -(x-mu_2)**2/(2*sigma**2)
        return torch.logaddexp(l1, l2)
        L1 = w*(1/np.sqrt(2*np.pi)/sigma*torch.exp(-(x-mu_1)**2/(2*sigma**2)))
        L2 = (1-w)*(1/np.sqrt(2*np.pi)/sigma*torch.exp(-(x-mu_2)**2/(2*sigma**2)))
        return torch.log(L1+L2)

    def estimate_parameters_sk(self, n=1000):
        X = np.array(self.sample(n)[0])
        model = GaussianMixture(n_components=2, covariance_type="tied")
        model.fit(X[:, None])
        return (model.means_[0, 0], model.means_[1, 0],
                np.sqrt(model.covariances_[0, 0]), model.weights_[0])

    def estimate_parameters(self, n=1000):
        return self.estimate_parameters_sk(n=n)
        n_iterations = 200
        s = self.sample(n)
        x = s[0]
        weights = torch.ones(n)*0.4
        mu_1, mu_2 = torch.rand(2)*10-5
        w = torch.rand(1)
        sigma = torch.rand(1)*10
        params = tuple(torch.tensor(p) for p in (mu_1, mu_2, sigma, w))
        for _ in range(n_iterations):
            l1, l2 = (self.l1(x, *params), self.l2(x, *params))
            weights = torch.exp(l1-torch.logaddexp(l1, l2))
            # print(weights)
            # super().log_likelihood(x, 1, *params)-self.log_likelihood(x, *params))
            m = weights > 1
            assert torch.all(~m), (weights[m], super().log_likelihood(x, 1, *params)[m], self.log_likelihood(x, *params)[m])
            mu_0 = torch.sum(weights*x)/torch.sum(weights)
            mu_1 = torch.sum((1-weights)*x)/torch.sum(1-weights)
            sigma = torch.sqrt(torch.sum((weights*(x-mu_0)**2 + (1-weights)*(x-mu_1)**2))/(torch.sum(weights)+torch.sum(1-weights)))
            w = torch.mean(weights)
            params = (mu_0, mu_1, sigma, w)
        # print(params)
        return params


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
        return np.array([lr.intercept_[0], lr.coef_[0, 0]])
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
                lambda eta_0, eta_1, eta_2, eta_3: eta_0 + eta_2/(2*eta_1) - eta_1/2,
                lambda eta_0, eta_1, eta_2, eta_3: torch.sqrt(eta_3),
                lambda eta_0, eta_1, eta_2, eta_3: torch.sigmoid(eta_2/eta_3)))


ABReparam = Reparametrization(
    old_to_new=(lambda mu_0, mu_1, sigma, w: (mu_0-mu_1)/sigma**2,
                lambda mu_0, mu_1, sigma, w: (mu_1**2-mu_0**2)/(2*sigma**2)+torch.log(w/(1-w)),
                lambda _, __, sigma, w: sigma,
                lambda _, __, sigma, w: w),
    new_to_old=(lambda alpha, beta, sigma, w: -(beta-torch.log(w/(1-w)))/alpha+sigma**2*alpha/2,
                lambda alpha, beta, sigma, w: -(beta-torch.log(w/(1-w)))/alpha-sigma**2*alpha/2,
                lambda alpha, beta, sigma, w: sigma,
                lambda alpha, beta, sigma, w: w))


MidPointReparam = Reparametrization(
    old_to_new=(lambda mu_0, mu_1, sigma, w: (mu_0+mu_1)/2-sigma**2*torch.log(w/(1-w))/(2*(mu_0-mu_1)),
                lambda mu_0, mu_1, _, __: (mu_0-mu_1),
                lambda _, __, sigma, w: sigma**2*torch.log(w/(1-w)),
                lambda _, __, sigma, w: sigma**2),
    new_to_old=(lambda eta_0, eta_1, eta_2, eta_3: eta_0 + eta_2/(2*eta_1) + eta_1/2,
                lambda eta_0, eta_1, eta_2, eta_3: eta_0 + eta_2/(2*eta_1) - eta_1/2,
                lambda eta_0, eta_1, eta_2, eta_3: torch.sqrt(eta_3),
                lambda eta_0, eta_1, eta_2, eta_3: torch.sigmoid(eta_2/eta_3)))

rp_full_triplet = tuple(reparametrize(cls, MidPointReparam) for cls in full_triplet)

ab_full_triplet = tuple(reparametrize(cls, ABReparam) for cls in full_triplet)


PureMidPointReparam = Reparametrization(
    old_to_new=(lambda alpha, beta: -alpha/beta,
                lambda alpha, beta: beta),
    new_to_old=(lambda eta_0, eta_1: -eta_0*eta_1,
                lambda eta_0, eta_1: eta_1))
PureMixtureConditionalRP = reparametrize(PureMixtureConditional, PureMidPointReparam)
