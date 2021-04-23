"""Console script for milpool."""
import sys
import click
import torch
import numpy as np
from .simulation import get_data_set, get_yx_bag_simulator, get_xy_bag_simulator
from .training import train_mil
from .evaluation import test
from .models import SimpleMIL
from .pool import XYMILPool, MaxPool, AvgPool, YXMILPool
@click.command()
@click.option("-n", '--n_iter',  default=1000)
@click.option('-s', '--n_samples', default=1000)
@click.option('-b', '--bag_size', default=1000)
@click.option('-d', '--direction', default="yx")
@click.option('-w', '--witness_rate', default=0.01)
@click.option('--input_dim', default=1)
def main(n_iter, n_samples, bag_size, direction, witness_rate, input_dim):
    """Console script for milpool."""
    torch.manual_seed(42)
    np.random.seed(1)
    bag_sim = get_yx_bag_simulator(bag_size, witness_rate, n_dim=input_dim) if direction=="yx" else get_xy_bag_simulator(bag_size)
    X, y, z = get_data_set(bag_sim, n_samples)
    print(X.shape, y.shape, z.shape)
    tX, ty, tz = get_data_set(bag_sim, 10000)
    print(y.sum(), y.shape)
    for func in (YXMILPool(), MaxPool(), AvgPool()):
        model = SimpleMIL(input_dim, func)
        m = train_mil(X, y, model, n_iterations=n_iter)
        # m = train_flat(X, z, func)
        test(m.linear, tX.reshape(-1, input_dim), tz.reshape(-1, 1))
        test(m, tX, ty)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
