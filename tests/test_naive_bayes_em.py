import pytest
import torch
from milpool.naive_bayes_em import ConditionalKmerModel


def test_shape():
    alphabet_size = 2
    n_samples = 4
    k = 3
    ps_0 = torch.linspace(0, 1, alphabet_size**k).reshape(-1, alphabet_size)
    ps_1 = torch.linspace(0, 1, alphabet_size**k).reshape(-1, alphabet_size)
    w = torch.tensor(0.5)

    model = ConditionalKmerModel(ps_0, ps_1, w)
    
    X, y = model.sample(n_samples)
    print(y)
    model.log_likelihood(X, y, ps_0, ps_1, w)
    print([[s.sum() for s in r] for r in model.estimate_fisher_information(n=10)])
    print([[s.shape for s in r] for r in model.estimate_fisher_information(n=10)])
    # assert False
