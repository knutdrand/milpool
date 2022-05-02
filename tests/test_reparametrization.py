import pytest
import numpy as np
import torch


from milpool.distributions import (
    MidPointReparam, PureMidPointReparam, ABReparam, NormalDistribution, NormalNaturalReparam)
from milpool.reparametrization import np_reparametrize



@pytest.mark.parametrize("params", [[0, 1, 1, 0.5], [0, 1, 2, 0.2], [0.2, 1.3, 0.5, 0.7]])
def test_midpoint_reparam(params):
    params = torch.tensor(params)
    new_params = [f(*params) for f in MidPointReparam.old_to_new]
    back_params = [f(*new_params) for f in MidPointReparam.new_to_old]
    assert np.allclose(np.array(back_params), np.array(params))

@pytest.mark.parametrize("params", [[0, 1, 1, 0.5], [0, 1, 2, 0.2], [0.2, 1.3, 0.5, 0.7]])
def test_ab_reparam(params):
    params = torch.tensor(params)
    new_params = [f(*params) for f in ABReparam.old_to_new]
    back_params = [f(*new_params) for f in ABReparam.new_to_old]
    print(np.array(back_params), np.array(params))
    assert np.allclose(np.array(back_params), np.array(params))


@pytest.mark.parametrize("params", [[0, 1], [1, 2], [-1, 1.5], [-1.2, -1.3]])
def test_pure_midpoint_reparam(params):
    params = torch.tensor(params)
    new_params = [f(*params) for f in PureMidPointReparam.old_to_new]
    back_params = [f(*new_params) for f in PureMidPointReparam.new_to_old]
    assert np.allclose(np.array(back_params), np.array(params))


def test_fisher_reparam():
    mu = torch.tensor(3.)
    sigma = torch.tensor(2.)
    dist = NormalDistribution(mu, sigma)
    rp_dist = np_reparametrize(NormalDistribution, NormalNaturalReparam())(mu, sigma)
    I = dist.estimate_fisher_information()
    I2 = rp_dist.estimate_fisher_information()
    new_I = np.array(NormalNaturalReparam().old_fisher_to_new(
        I, torch.stack(rp_dist.params)))
    assert np.allclose(new_I, I2, rtol=0.01)
