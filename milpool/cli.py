"""Console script for milpool."""
import sys
import click
import torch
import numpy as np
from .simulation import get_data_set, get_yx_bag_simulator, get_xy_bag_simulator
from .training import train_mil
from .evaluation import test
from .pool import XYMILPool, max_pool, YXMILPool
@click.command()
@click.option("-n", '--n_iter',  default=1000)
@click.option('-s', '--n_samples', default=1000)
@click.option('-b', '--bag_size', default=1000)
@click.option('-d', '--direction', default="yx")
@click.option('-w', '--witness_rate', default=0.01)
def main(n_iter, n_samples, bag_size, direction, witness_rate):
    """Console script for milpool."""
    torch.manual_seed(42)
    np.random.seed(1)
    bag_sim = get_yx_bag_simulator(bag_size, witness_rate) if direction=="yx" else get_xy_bag_simulator(bag_size)
    X, y, z = get_data_set(bag_sim, n_samples)
    tX, ty, tz = get_data_set(10000, bag_size)
    print(y.sum(), y.shape)
    for func in (YXMILPool(), max_pool, XYMILPool()):
        m = train_mil(X, y, func, n_iterations=n_iter)
        # m = train_flat(X, z, func)
        test(m.linear, tX.reshape(-1, 1), tz.reshape(-1, 1))
        test(m, tX, ty)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
