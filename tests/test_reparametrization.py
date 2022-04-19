import pytest
import numpy as np
import torch


from milpool.distributions import MidPointReparam, PureMidPointReparam


@pytest.mark.parametrize("params", [[0, 1, 1, 0.5], [0, 1, 2, 0.2], [0.2, 1.3, 0.5, 0.7]])
def test_midpoint_reparam(params):
    params = torch.tensor(params)
    new_params = [f(*params) for f in MidPointReparam.old_to_new]
    back_params = [f(*new_params) for f in MidPointReparam.new_to_old]
    assert np.allclose(np.array(back_params), np.array(params))


@pytest.mark.parametrize("params", [[0, 1], [1, 2], [-1, 1.5], [-1.2, -1.3]])
def test_pure_midpoint_reparam(params):
    params = torch.tensor(params)
    new_params = [f(*params) for f in PureMidPointReparam.old_to_new]
    back_params = [f(*new_params) for f in PureMidPointReparam.new_to_old]
    assert np.allclose(np.array(back_params), np.array(params))
