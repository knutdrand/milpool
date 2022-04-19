import numpy as np
import logging as log
import torch
from scipy.special import logsumexp, logit
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

class GaussModel:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __repr__(self):
        return f"N({self.mu}, {self.sigma**2})"
    
    def fit(self, X, weights):
        self.mu = np.sum(weights[:, None]*X, axis=0)/np.sum(weights)
        self.sigma = np.sqrt(np.sum(weights[:, None]*X**2, axis=0)/np.sum(weights)-self.mu**2)


    def logpdf(self, X):
        return multivariate_normal(self.mu, np.diag(self.sigma**2)).logpdf(X)

    def pdf(self, X):
        return multivariate_normal(self.mu, np.diag(self.sigma**2)).pdf(X)
        

class YXModel:
    def __init__(self, negative_model, positive_model):
        self.negative_model = negative_model
        self.positive_model = positive_model
        self.w = 0.2
        self.q = 0.5

    def __repr__(self):
        return f"MIL({self.q}, {self.w}, {self.negative_model}, {self.positive_model})"

    def get_posterior(self, X, prior=None):
        if prior is None:
            prior = self.w
        if prior in (0, 1):
            return prior

        log_w = np.log(prior)
        log_not_w = np.log(1-prior)
        p_positive = self.positive_model.logpdf(X)
        p_negtaive = self.negative_model.logpdf(X)
        posterior_pos = log_w+p_positive-np.logaddexp(log_w+p_positive, log_not_w+p_negtaive)
        return np.exp(posterior_pos)

    def instance_model(self, X):
        posterior = self.get_posterior(X, self.q*self.w)
        return torch.tensor(logit(posterior)[:, None], dtype=torch.float32)

    def __call__(self, X):
        if self.w == 1:
            pos_posterior = np.log(self.q)+self.positive_model.logpdf(X).sum(axis=-1)
        else:
            pos_posterior = np.log(self.q)+np.logaddexp(
                np.log(self.w)+self.positive_model.logpdf(X),
                np.log(1-self.w)+self.negative_model.logpdf(X)).sum(axis=-1)

        neg_posterior = np.log(1-self.q)+self.negative_model.logpdf(X).sum(axis=-1)
        l = logit(np.exp(pos_posterior-np.logaddexp(pos_posterior, neg_posterior)))
        return torch.tensor(l[:, None], dtype=torch.float32)

    def train(self, X, y, n_iter=10, n_epochs=5, do_fit_negative=False):
        X = X.detach().numpy()
        y = y.detach().numpy().flatten()
        negative_bags = X[y==0]
        positive_bags = X[y==1]
        negative_instances = negative_bags.reshape(-1, X.shape[-1])
        possible_positive_instances = positive_bags.reshape(-1, X.shape[-1])

        all_instances = np.vstack((possible_positive_instances, negative_instances))
        n_pos = len(possible_positive_instances)
        self.q = len(positive_bags)/len(X)
        all_weights = np.zeros(len(all_instances))
        all_weights[:n_pos] = 0.4
        self.negative_model.fit(negative_instances, np.ones(len(negative_instances)))
        self.positive_model.fit(possible_positive_instances, np.ones(len(possible_positive_instances)))
        for i in range(n_iter*n_epochs):
            if (i+1) % n_iter == 0:
                log.info(f"w: {self.w}, +: {self.positive_model}, -: {self.negative_model}")
            all_weights[:n_pos] = self.get_posterior(possible_positive_instances)
            self.positive_model.fit(all_instances, all_weights)
            if do_fit_negative:
                self.negative_model.fit(all_instances, 1-all_weights)
            self.w = np.sum(all_weights[:n_pos])/n_pos
