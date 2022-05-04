import torch
import numpy as np
from .distributions import Distribution, MixtureDistribution
from sklearn.mixture._base import BaseMixture
from sklearn.mixture._gaussian_mixture import _check_means, _check_weights
# from scipy.stats import poisson
log_factorial = np.concatenate(([0], np.cumsum(np.log(np.arange(1, 1000)))))


def log_likelihood(k, mu):
    k = np.asanyarray(k[..., None, :], dtype="int")
    v = (k*np.log(mu)-mu)
    u = -log_factorial[k]
    return (v+u).sum(axis=-1)


def _estimate_poisson_parameters(X, resp):
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    return nk, means


class PoissonMixture(BaseMixture):
    def __init__(
        self,
        n_components=1,
        *,
        tol=1e-3,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        super().__init__(
            n_components=n_components,
            tol=tol,
            reg_covar=0,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )
        self.weights_init = weights_init
        self.means_init = means_init

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _check_parameters(self, X):
        """Check the Gaussian mixture parameters are well defined."""
        _, n_features = X.shape
        if self.weights_init is not None:
            self.weights_init = _check_weights(self.weights_init, self.n_components)

        if self.means_init is not None:
            self.means_init = _check_means(
                self.means_init, self.n_components, n_features
            )

    def _get_parameters(self):
        return (
            self.weights_,
            self.means_,
        )

    def _set_parameters(self, params):
        (
            self.weights_,
            self.means_,
        ) = params

    def _initialize(self, X, resp):
        """Initialization of the Gaussian mixture parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        resp : array-like of shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        weights, means = _estimate_poisson_parameters(X, resp)
        weights /= n_samples

        self.weights_ = weights if self.weights_init is None else self.weights_init
        self.means_ = means if self.means_init is None else self.means_init

    def _estimate_log_prob(self, X):
        return log_likelihood(X, self.means_)

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        self.weights_, self.means_ = _estimate_poisson_parameters(X, np.exp(log_resp))
        self.weights_ /= self.weights_.sum()
        # print(self.weights_, self.means_)


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
