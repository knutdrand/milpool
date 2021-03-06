import torch
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

from .reparametrization import Reparametrization, reparametrize, NpReparametrization, np_reparametrize

def get_var(information):
    print(information)
    return np.linalg.inv(information)


@dataclass
class Distribution:
    def sample(self, n=100):
        return NotImplemented

    def get_func(self, *x):
        return lambda *params: torch.mean(self.log_likelihood(*x, *params))

    def estimate_fisher_information(self, n=10000000):
        x = self.sample(n)
        f = self.get_func(*x)
        print("P", self.params)
        H = torch.autograd.functional.hessian(f, self.params)
        print(H)
        H = np.array([[np.array(h) for h in row] for row in H])
        H = H.reshape(H.shape[-1], -1)
        return -np.array(H)  # (self.mu, self.sigma)))/n

    def prob(self, *x):
        return np.exp(self.log_likelihood(*x, *self.params))

    def plot(self):
        x = self._get_x_for_plotting()
        params = self.params
        y = np.exp(self.log_likelihood(x[:, None], *params))
        plt.plot(x, y)

    def plot_all_errors(self, color="red", n_params=None):
        if n_params is None:
            n_params = len(self.params)
        name = self.__class__.__name__
        I = self.estimate_fisher_information()
        I = I[:n_params, :n_params]
        all_var = get_var(I)
        n_samples = [200*i for i in range(1, 10)]
        errors = [self.get_square_errors(n_samples=n, n_iterations=200, do_plot=False) for n in n_samples]
        print(errors)
        fig, axes = plt.subplots((n_params+1)//2, 2)
        if (n_params+1)//2 == 1:
            axes = [axes]
        if len(self.params) == 1:
            params = self.params[0]
        else:
            params = self.params
        for i, param in enumerate(params[:n_params]):
            var = all_var[i, i]
            ax = axes[i//2][i % 2]
            ax.axline((0, 0), slope=1/var, color=color, label=name+" CRLB")
            ax.plot(n_samples, 1/np.array(errors)[:, i], color=color, label=name+" errors")
            ax.set_ylabel("1/sigma**2")
            ax.set_xlabel("n_samples")

    def get_square_errors(self, n_samples=1000, n_iterations=1000, do_plot=False):
        estimates = [self.estimate_parameters(n_samples)
                     for _ in range(n_iterations)]
        if len(self.params) == 1:
            true_params = np.array(self.params[0])
            estimates = np.array([np.array(row[0]) for row in estimates])
        else:
            true_params = np.array(self.params)
            estimates = np.array(estimates)
        if do_plot:
            for i, param in enumerate(true_params):
                plt.hist(estimates[:, i])
                plt.axvline(x=param)
                plt.title(f"n={n_samples}")
                plt.show()
        print("E", estimates.mean(axis=0))
        print("T", true_params)
        return ((estimates-true_params)**2).sum(axis=0)/n_iterations


class MixtureDistribution(Distribution):
    def __init__(self, distributions, weights):
        self._distributions = distributions
        self._weights = torch.as_tensor(weights)
        self.params = (torch.hstack([p for d in distributions for p in d.params]+[self._weights]),)#  for d in distributions]+weights)
        self._param_numbers = [len(d.params[0]) for d in distributions]
        self._param_offsets = np.cumsum(self._param_numbers)
        self._n_components = len(self._distributions)

    def log_likelihood(self, x, *params):
        weights = params[0][-self._n_components:]
        ps = [params[0][offset-n:offset] for n, offset
              in zip(self._param_numbers, self._param_offsets)]
        l = [torch.log(w) + d.log_likelihood(x, p)
             for w, d, p in zip(weights, self._distributions, ps)]
        return torch.logsumexp(torch.vstack(l), axis=0)

    def sample(self, n_samples):
        z = torch.multinomial(self._weights, n_samples, replacement=True)
        counts = torch.bincount(z, minlength=len(self._distributions))
        l = [dist.sample(n)[0] for dist, n in zip(self._distributions, counts)]
        return [torch.vstack(l)]


class NormalDistribution(Distribution):
    mu: float = torch.tensor(0.)
    sigma: float = torch.tensor(1.)
    params = (mu, sigma)

    def __init__(self, mu, sigma):
        self.mu = torch.as_tensor(mu)
        self.sigma = torch.as_tensor(sigma)
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
        self.mu_1 = torch.as_tensor(mu_1)
        self.mu_2 = torch.as_tensor(mu_2)
        self.sigma = torch.as_tensor(sigma)
        self.w = torch.as_tensor(w)
        self.params = (self.mu_1, self.mu_2, self.sigma, self.w)

    def sample(self, n=1):
        y = torch.bernoulli(self.w*torch.ones(n))
        mu = self.mu_1*y+self.mu_2*(1-y)
        return torch.normal(mu, self.sigma), y
    
    def l1(self, x, mu_1, mu_2, sigma, w):
        # mu_1, mu_2, sigma, w = (self.mu_1, self.mu_2, self.sigma, self.w)
        return torch.log(w)+np.log(1/np.sqrt(2*np.pi))-torch.log(sigma) - (x-mu_1)**2/(2*sigma**2)

    def l2(self, x, mu_1, mu_2, sigma, w):
        # mu_1, mu_2, sigma, w = (self.mu_1, self.mu_2, self.sigma, self.w)
        return torch.log(1-w)+np.log(1/np.sqrt(2*np.pi))-torch.log(sigma) - (x-mu_2)**2/(2*sigma**2)
        # torch.log(1-w)+torch.log(1/torch.sqrt(2*np.pi*sigma**2)) -(x-mu_2)**2/(2*sigma**2)

    def log_likelihood(self, x, y, mu_1, mu_2, sigma, w):
        l1 = self.l1(x, mu_1, mu_2, sigma, w)
        # torch.log(w)+np.log(1/np.sqrt(2*np.pi))-torch.log(sigma) -(x-mu_1)**2/(2*sigma**2)
        l2 = self.l2(x, mu_1, mu_2, sigma, w)
        # torch.log(1-w)+np.log(1/np.sqrt(2*np.pi))-torch.log(sigma) -(x-mu_2)**2/(2*sigma**2)
        return y*l1+(1-y)*l2

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

    def _get_x_for_plotting(self):
        d = torch.abs(self.mu_1- self.mu_2)
        m = min(self.mu_1, self.mu_2)-d/2
        return torch.linspace(m, m+2*d, 100)
    
    def log_likelihood(self, x, mu_1, mu_2, sigma, w):
        l1 = self.l1(x, mu_1, mu_2, sigma, w)
        l2 = self.l2(x, mu_1, mu_2, sigma, w)
        # torch.log(w)+np.log(1/np.sqrt(2*np.pi))-torch.log(sigma) -(x-mu_1)**2/(2*sigma**2)
        # l2 = torch.log(1-w)+np.log(1/np.sqrt(2*np.pi))-torch.log(sigma) -(x-mu_2)**2/(2*sigma**2)
        return torch.logaddexp(l1, l2)
        L1 = w*(1/np.sqrt(2*np.pi)/sigma*torch.exp(-(x-mu_1)**2/(2*sigma**2)))
        L2 = (1-w)*(1/np.sqrt(2*np.pi)/sigma*torch.exp(-(x-mu_2)**2/(2*sigma**2)))
        return torch.log(L1+L2)

    def estimate_parameters_sk(self, X):
        model = GaussianMixture(n_components=2, covariance_type="tied", reg_covar=0)
        model.fit(X[:, None])
        if False:
            print(model.means_[0, 0], model.means_[1, 0],
                  np.sqrt(model.covariances_[0, 0]), model.weights_)
        return (model.means_[0, 0], model.means_[1, 0],
                np.sqrt(model.covariances_[0, 0]), model.weights_[0])

    def estimate_parameters(self, n=1000):
        
        s = self.sample(n)
        x = s[0]
        # return self.estimate_parameters_sk(x)
        return self._estimate(x)

    def _estimate(self, x):
        epoch = 100
        n_iterations = epoch*5
        # weights = torch.ones(x.shape[0])*0.4
        mu_0 = torch.min(x)
        mu_1 = torch.max(x) # , mu_1 = torch.rand(2)*10-5
        w = torch.as_tensor(0.5)# torch.rand(1)
        sigma = torch.rand(1)*10
        params = tuple(torch.as_tensor(p) for p in (mu_0, mu_1, sigma, w))

        #model = GaussianMixture(n_components=2, covariance_type="tied", reg_covar=0,
        #weights_init=np.array([w, (1-w)]),
        #                         means_init=np.array([[mu_0], [mu_1]]),
        #                         precisions_init=np.array([[1/sigma]]))
        # model.fit(x[:, None])
        # plt.hist(np.array(x))
        # plt.show()
        for i in range(n_iterations):
            l1, l2 = (self.l1(x, *params), self.l2(x, *params))
            weights = torch.exp(l1-torch.logaddexp(l1, l2))
            m = weights > 1
            if i % epoch == -1:
                plt.scatter(x, weights)
                plt.show()
            assert torch.all(~m)
            N_0 = weights.sum()
            N_1 = (1-weights).sum()
            mu_0 = torch.sum(weights*x)/N_0
            mu_1 = torch.sum((1-weights)*x)/N_1
            sigma = torch.sqrt(torch.sum((weights*(x-mu_0)**2 + (1-weights)*(x-mu_1)**2))/(N_0+N_1))
            w = N_0/(N_0+N_1)# torch.mean(weights)
            params = (mu_0, mu_1, sigma, w)
        # print(model.means_[0, 0], model.means_[1, 0],
        #       np.sqrt(model.covariances_[0, 0]), model.weights_)
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


class MarginalReparam(NpReparametrization):
    def _old_to_new(self, params):
        mu_0, mu_1, sigma, w = params
        return [(mu_0+mu_1)/2,
                2*torch.log(w/(1-w))/(mu_0-mu_1),
                # (mu_0-mu_1)/2,
                sigma,
                torch.log(w/(1-w))]

    def _new_to_old(self, theta):
        return [(theta[0] + theta[3]/theta[1]),
                (theta[0] - theta[3]/theta[1]),
                theta[2],
                torch.sigmoid(theta[3])]


class MarginalReparamOneWay(NpReparametrization):
    def _old_to_new(self, params):
        mu_0, mu_1, sigma, w = params
        return [(mu_0+mu_1)/2,
                (mu_0-mu_1)**2,
                #2*torch.log(w/(1-w))/(mu_0-mu_1),
                # (mu_0-mu_1)/2,
                sigma,
                torch.log(w/(1-w))**2]

    def _new_to_old(self, theta):
        return [(theta[0] + theta[3]/theta[1]),
                (theta[0] - theta[3]/theta[1]),
                theta[2],
                torch.sigmoid(theta[3])]


rp_full_triplet = tuple(reparametrize(cls, MidPointReparam) for cls in full_triplet)
marginal_triplet = tuple(np_reparametrize(cls, MarginalReparamOneWay()) for cls in full_triplet)
ab_full_triplet = tuple(reparametrize(cls, ABReparam) for cls in full_triplet)


PureMidPointReparam = Reparametrization(
    old_to_new=(lambda alpha, beta: -alpha/beta,
                lambda alpha, beta: beta),
    new_to_old=(lambda eta_0, eta_1: -eta_0*eta_1,
                lambda eta_0, eta_1: eta_1))
PureMixtureConditionalRP = reparametrize(PureMixtureConditional, PureMidPointReparam)
