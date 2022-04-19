import pytest
import numpy as np
from milpool.MIL_distributions import MILXY
np.set_printoptions(suppress=True)


def test_sample():
    x, y = MILXY().sample(10)
    print(x.sum(axis=-1), y)
    assert False
