import numpy as np

from sklearn.mixture._base import BaseMixture
from sklearn.mixture._gaussian_mixture import _check_means, _check_weights


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


def _estimate_poisson_parameters_mil(X, resp, neg_sum, n_neg):
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    numerator = np.dot(resp.T, X)
    denominator = nk[:, np.newaxis].copy()
    numerator[1] += neg_sum
    denominator[1] += n_neg
    means = numerator/denominator
    # means = (np.dot(resp.T, X)+neg_sum]) / (nk[:, np.newaxis] + [n_neg)
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


class PoissonMIL(PoissonMixture):
    def fit(self, X, y):
        pos_X = X[y.ravel() == 1].reshape(-1, X.shape[-1])
        neg_X = X[y.ravel() == 0].reshape(-1, X.shape[-1])
        self._neg_sum = neg_X.sum(axis=0)
        self._n_neg = len(neg_X)
        super().fit(pos_X)

    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        resp = np.exp(log_resp)
        self.weights_, self.means_ = _estimate_poisson_parameters_mil(
            X, resp, self._neg_sum, self._n_neg)
        print(self.weights_.sum())
        self.weights_ /= self.weights_.sum()

    def _initialize_parameters(self, X, random_state):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        """
        n_samples, _ = X.shape
        resp = np.full((n_samples, 2), 0.5)
        self._initialize(X, resp)

    def _initialize(self, X, resp):
        """Initialization of the Gaussian mixture parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        resp : array-like of shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        weights, means = _estimate_poisson_parameters_mil(
            X, resp, self._neg_sum, self._n_neg)
        weights /= n_samples

        self.weights_ = weights if self.weights_init is None else self.weights_init
        self.means_ = means if self.means_init is None else self.means_init
