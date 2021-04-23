import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

global_w = 0.01
global_q = 0.5

model = lambda size: np.random.normal(0, 0.5, size)
pos_model = lambda size: np.random.normal(3, 0.5, size)
noise_model = lambda size: np.random.normal(0, 0.5, size)


def get_models(n_dim=1, rho=0.5):
    m= lambda size: np.random.multivariate_normal(np.zeros(n_dim), np.eye(n_dim)*rho, size)
    # MultivariateNormal(torch.zeros(2), 0.5*torch.eye(2)).sample(torch.Size([size]))
    pm = lambda size: np.random.multivariate_normal(np.ones(n_dim), rho*np.eye(n_dim), size)
    # MultivariateNormal(2*torch.ones(2), 0.5*torch.eye(2)).sample(torch.Size([size]))
    return m, pm

def simulate_bag_xy(bag_size, model, noise_model, threshold):
    bag = model(bag_size)
    noise = noise_model(bag_size)
    labels = bag+noise >= threshold
    return bag, np.any(labels), labels


def simulate_bag_yx(bag_size, pos_model, neg_model, w, py):
    y = np.random.rand()<py
    labels = np.random.rand(bag_size)<w if y else np.zeros(bag_size, dtype="bool")
    bag = np.where(labels[:, None], pos_model(bag_size), neg_model(bag_size))
    return bag, y, labels

def get_data_set(simulator, n_bags):
    xs = []
    ys = []
    zs = []
    for _ in range(n_bags):
        x,y,z = simulator()
        xs.append(x)
        ys.append(y)
        zs.append(z)

    X = torch.tensor(xs, dtype=torch.float32)#[..., None]
    y = torch.tensor(ys, dtype=torch.float32)[..., None]
    z = torch.tensor(zs, dtype=torch.float32)[..., None]
    return X, y, z

def get_yx_bag_simulator(bag_size, witness_rate, q=0.5, n_dim=1):
    m, pm = get_models(n_dim, 0.5*np.sqrt(n_dim))#  (model, pos_model)
    #if n_dim == 2:
    #     m, pm = (model_2d, pos_model_2d)
    return lambda:  simulate_bag_yx(bag_size, pm, m, witness_rate, q)

def get_xy_bag_simulator(bag_size):
    return lambda: simulate_bag_xy(bag_size, model, noise_model, 2.2)

# xy_simulator = lambda b: simulate_bag_xy(b, model, noise_model, 2.2)
# yx_simulator = lambda b: simulate_bag_yx(b, pos_model, model, global_w, global_q)

get_xy_data = lambda n, b: get_data_set(xy_simulator, n, b)
get_yx_data = lambda n, b: get_data_set(yx_simulator, n, b)
