"""Console script for milpool."""
import sys
import click
import torch
import numpy as np
import logging
from .visualization import plot_distributions_2d, plot_distributions_1d, contour_model_2d
from .simulation import get_data_set, get_yx_bag_simulator, get_xy_bag_simulator
from .training import train_mil, train_flat
from .evaluation import test
from .models import SimpleMIL, ALinearMIL, MIL, AdjustedALinearMIL, QuadraticMIL
from .pool import XYMILPool, MaxPool, AvgPool, YXMILPool, SoftMaxPool
from .em import YXModel, GaussModel

logging.basicConfig(level=logging.INFO)
@click.group()
@click.option('-s', '--n_samples', default=1000)
@click.option('-b', '--bag_size', default=1000)
@click.option('-d', '--input_dim', default=1)
@click.pass_context
def simulate(ctx, n_samples, bag_size, input_dim):
    torch.manual_seed(42)
    np.random.seed(1)
    ctx.ensure_object(dict)
    ctx.obj["PARAMS"] = (n_samples, bag_size, input_dim)

@simulate.command()
@click.option('-e', '--epsilon', default=0.2)
@click.option('-T', '--threshold', default=2.2)
@click.option('-o', '--outfile', type=click.File("wb"))
@click.pass_context
def xy(ctx, epsilon, threshold, outfile):
    n_samples, bag_size, input_dim = ctx.obj["PARAMS"]
    bag_sim = get_xy_bag_simulator(bag_size, epsilon, threshold, input_dim)
    X, y, z = get_data_set(bag_sim, n_samples)
    np.savez(outfile, X=X, y=y, z=z)

@simulate.command()
@click.option('-w', '--witness_rate', default=0.1)
@click.option('-r', '--rho', default=0.5)
@click.option('-o', '--outfile', type=click.File("wb"))
@click.pass_context
def yx(ctx, witness_rate, rho, outfile):
    n_samples, bag_size, input_dim = ctx.obj["PARAMS"]
    bag_sim = get_yx_bag_simulator(bag_size, witness_rate, n_dim=input_dim, rho=rho)
    X, y, z = get_data_set(bag_sim, n_samples)
    np.savez(outfile, X=X, y=y, z=z)

@click.command()
@click.option('-i', '--infile', type=click.File("rb"))
@click.option('-oz', '--outfile_z', type=click.File("wb"))
@click.option('-oy', '--outfile_y', type=click.File("wb"))
def plot(infile, outfile_z, outfile_y):
    data = np.load(infile)
    plot_distributions_2d(torch.tensor(data["X"]), torch.tensor(data["z"]), outfile_z)
    y = np.ones_like(data["z"])*data["y"][..., None]
    plot_distributions_2d(torch.tensor(data["X"]), torch.tensor(y), outfile_y)

@click.command()
@click.option("-n", '--n_iter',  default=1000)
@click.option('-s', '--n_samples', default=1000)
@click.option('-b', '--bag_size', default=1000)
@click.option('-d', '--direction', default="yx")
@click.option('-w', '--witness_rate', default=0.01)
@click.option('--input_dim', default=1)
@click.option('--do_plot', default=False)
@click.option('-r', '--rho', default=0.5)
@click.option('-T', '--threshold', default=2.2)
@click.option('-i', '--insignificant', default=0)
def main(n_iter, n_samples, bag_size, direction, witness_rate, input_dim, do_plot, rho, threshold, insignificant):
    """Console script for milpool."""
    torch.manual_seed(42)
    np.random.seed(1)
    bag_sim = get_yx_bag_simulator(bag_size, witness_rate, n_dim=input_dim, rho=rho, insignificant=insignificant) if direction in ("yx", "em") else get_xy_bag_simulator(bag_size, rho, threshold, input_dim)
    X, y, z = get_data_set(bag_sim, n_samples)
    plotter = lambda *args: None
    if input_dim == 1:
        plotter = plot_distributions_1d
    elif input_dim == 2:
        plotter = plot_distributions_2d

    print(X.shape, y.shape, z.shape)
    if do_plot:
        plotter(X, z)
    tX, ty, tz = get_data_set(bag_sim, 1000)
    print(y.sum(), y.shape)

    if direction=="em":
        if do_plot:
            plotter(X, z)
        pos, neg = (GaussModel([0.5+i/10]*input_dim, [1.]*input_dim) for i in range(2))
        m = YXModel(pos, neg)
        m.train(X, y, n_iter)
        print(m)
        new_zs = test(m.instance_model, tX.reshape(-1, input_dim), tz.reshape(-1, 1))
        if do_plot:
            plotter(tX, new_zs)

        test(m, tX, ty)
        
    # m = train_flat(X, z, ALinearMIL(X.shape[-1]))
    # if do_plot:
        # plotter(tX, test(m, tX.reshape(-1, input_dim), tz.reshape(-1, 1)))
    #for func in (YXMILPool(), MaxPool(), SoftMaxPool(), YXMILPool(witness_rate)):
    for func in (YXMILPool(witness_rate), MaxPool(), SoftMaxPool()):
        print("----------------", func.__class__.__name__, "----------------")
        name = func.__class__.__name__
        # model = SimpleMIL(input_dim, func)
        model = QuadraticMIL(input_dim, func)
        #model = AdjustedALinearMIL(input_dim, func)
        # train_mil(X, y, MIL(input_dim, model.instance_model, MaxPool()),
        #           n_iterations=300)
        m = train_mil(X, y, model, n_iterations=n_iter)
        new_zs = test(m.instance_model, tX.reshape(-1, input_dim), tz.reshape(-1, 1))
        if do_plot:
            contour_model_2d(m.instance_model)
            plotter(tX, new_zs)
        test(m, tX, ty)
        
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
