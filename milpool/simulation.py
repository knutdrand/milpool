import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

global_w = 0.01
global_q = 0.5

model = lambda size: np.random.normal(0, 0.5, size)
pos_model = lambda size: np.random.normal(3, 0.5, size)
noise_model = lambda size: np.random.normal(0, 0.5, size)

class YXModel:
    def __init__(self, q=0.5, w=0.1):
        self.q = q
        self.w = w


def get_circle_model(size, r_mu, rho=0.5):
    r = np.random.lognormal(r_mu, sigma=rho, size=size)
    theta = np.random.rand(size)*2*np.pi
    return np.array([r*np.cos(theta), r*np.sin(theta)]).T

def get_circle_models(n_dim=2, rho=0.5):
    assert n_dim==2
    m = lambda size: get_circle_model(size, 0, rho)
    pm = lambda size: get_circle_model(size, 1, rho)
    return m, pm

def get_overlap_models(n_dim=2, rho_factor=10):
    m = lambda size: np.random.multivariate_normal(np.zeros(n_dim), np.eye(n_dim)*rho_factor, size)
    pm = lambda size: np.random.multivariate_normal(np.zeros(n_dim), np.eye(n_dim), size)
    return m, pm


def get_models(n_dim=1, rho=0.5, n_insignificant_dim=0):
    m= lambda size: np.random.multivariate_normal(np.zeros(n_dim), np.eye(n_dim)*rho, size)
    # MultivariateNormal(torch.zeros(2), 0.5*torch.eye(2)).sample(torch.Size([size]))
    pm_mu = np.ones(n_dim)
    pm_mu[n_dim-n_insignificant_dim:] = 0
    pm = lambda size: np.random.multivariate_normal(pm_mu, rho*np.eye(n_dim), size)
    print(f"pm: N({pm_mu}, {rho})")
    print(f"nm: N(0, {rho})")
    # MultivariateNormal(2*torch.ones(2), 0.5*torch.eye(2)).sample(torch.Size([size]))
    return m, pm

def get_xy_models(n_dim=1, rho=0.5):
    model = lambda size: np.random.multivariate_normal(np.zeros(n_dim), np.eye(n_dim), size)
    noise = lambda size: np.random.normal(0, rho*np.sqrt(n_dim), size)
    return model, noise

def simulate_bag_xy(bag_size, model, noise_model, discriminator):
    bag = model(bag_size)
    noise = noise_model(bag_size)
    labels = discriminator(bag, noise) # bag.sum(axis=-1)+noise >= threshold*np.sqrt(bag.shape[-1])
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

def get_yx_bag_simulator(bag_size, witness_rate, q=0.5, n_dim=1, rho=0.5, insignificant=0):
    print(insignificant)
    # m, pm = get_models(n_dim, rho*np.sqrt(n_dim), n_insignificant_dim=insignificant)
    # m, pm = get_circle_models(n_dim, rho=rho)
    m, pm = get_overlap_models(n_dim, 1/rho)
    return lambda:  simulate_bag_yx(bag_size, pm, m, witness_rate, q)

def get_xy_bag_simulator(bag_size, rho=0.5, threshold=2.2, n_dim=1):
    model, noise = get_xy_models(n_dim, rho)
    circle_discriminator = lambda bag, noise: np.sqrt((bag**2).sum(axis=-1))+noise >= threshold
    linear_discriminator = lambda bag, noise: bag.sum(axis=-1)/bag.shape[-1]+noise >= threshold
    discriminator = linear_discriminator
    return lambda: simulate_bag_xy(bag_size, model, noise, linear_discriminator)
                                   # lambda bag, noise: bag.sum(axis=-1)+noise >= threshold*np.sqrt(bag.shape[-1]))


# xy_simulator = lambda b: simulate_bag_xy(b, model, noise_model, 2.2)
# yx_simulator = lambda b: simulate_bag_yx(b, pos_model, model, global_w, global_q)

get_xy_data = lambda n, b: get_data_set(xy_simulator, n, b)
get_yx_data = lambda n, b: get_data_set(yx_simulator, n, b)
