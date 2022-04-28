from dataclasses import dataclass
from sklearn.naive_bayes import MultinomialNB
from .distributions import Distribution
import numpy as np
import torch
"""
P(s | f, class) = ps[s, f, class]

"""

@dataclass
class ConditionalKmerModel(Distribution):
    ps_0: np.ndarray
    ps_1: np.ndarray
    w: np.ndarray
    seq_len: torch.tensor = torch.tensor(6)
    # params = (ps_0, ps_1, w)

    def __post_init__(self):
        self.params = (self.ps_0, self.ps_1, self.w)
    # def __init__(self, k, alphabet_size=4):
        #self.models = [MultinomialNB() for _ in range(alphabet_size**(k-1))]
        # self.parameters = np.zeros(alphabet_size**(k-1), alphabet_size)

    def fit(self, X, y, w):
        for i, model in enumerate(self.models):
            model.fit(X[:, i, :], y, w)

    def predict_log_proba(self, X):
        return sum(model.predict_log_proba(X[:, i, :]) for i, model in enumerate(self.models))

    def log_likelihood(self, x, y, ps_0, ps_1, w):
        x = x.swapaxes(-1, -2)
        l_0 = torch.log(w) + torch.distributions.categorical.Categorical(probs=ps_0).log_prob(x)
        l_1 = torch.log(1-w) + torch.distributions.categorical.Categorical(probs=ps_1).log_prob(x)
        return l_0**y + l_1**(1-y)

    def sample(self, n=100):
        """
        X n_samples x n_flank x alphabet_size
        """
        y = torch.bernoulli(self.w*torch.ones(n))[:, None]
        xs = []
        for row_0, row_1 in zip(self.ps_0, self.ps_1):
            x_0 = torch.multinomial(row_0, n*self.seq_len, replacement=True).reshape(n, self.seq_len)
            x_1 = torch.multinomial(row_1, n*self.seq_len, replacement=True).reshape(n, self.seq_len)
            x = x_0**y*x_1**(1-y)
            xs.append(np.array(x))
        tmp = np.array(xs)
        X = torch.tensor(tmp)
        return X.swapaxes(0, 1), y[..., None]


            

class ConditionalKmerModelX(ConditionalKmerModel):
    def sample(self, n=1):
        return super().sample(n)[:1]

    def log_likelihood(self, x, p_0, p_1, w):
        x = x.swapaxes(-1, -2)
        l_0 = torch.log(w) + torch.distributions.categorical.Categorical(probs=ps_0).log_prob(x)
        l_1 = torch.log(1-w) + torch.distributions.categorical.Categorical(probs=ps_1).log_prob(x)
        return torch.logaddexp(l_0, l_1)

    def estimate_parameters(self, n=1000):
        n_iterations = 50
        x, y = self.sample(n)
        pi = torch.ones(n)*0.5
        params = (0.5*torch.ones_like(self.p_0), 0.5*torch.ones_like(self.p_1), torch.tensor(0.5))
        for _ in range(n_iterations):
            pi = self._get_probs(params, x)
            params = self._update_params(params, pi)

    def _get_probs(self, params, x):
        a = self.log_likelihood(x, torch.tensor(1), *params)
        b = self.log_likelihood(x, torch.tensor(0), *params)
        return torch.exp(a-torch.logaddexp(a, b))

    def _estimate_probs(self, x, weights):
        """X n_samples x n_flank x alphabet_size"""
        weights = weights[:, None, None]
        totals = weights

        a = self.log_likelihood(x, torch.tensor(1), *params)
        b = self.log_likelihood(x, torch.tensor(0), *params)
        return torch.exp(a-torch.logaddexp(a, b))

    def _update_params(self, pi, x):
        p_0 = self._estimate_probs(x, pi)
        p_1 = self._estimate_probs(x, 1-pi)
        w = pi.mean()
        return (p_0, p_1, w)



if __name__ == "__main__":
    pass

    rng = np.random.RandomState(1)
    X = rng.randint(5, size=(6, 100))
    y = np.array([1, 2, 3, 4, 5, 6])
    
    clf = MultinomialNB()
    clf.fit(X, y)
    MultinomialNB()
    
    print(clf.predict(X[2:3]))
