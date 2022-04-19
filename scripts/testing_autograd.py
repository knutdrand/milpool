import torch
import numpy as np
import logging
from milpool.distributions import full_triplet, rp_full_triplet, PureMixtureConditionalRP, PureMixtureConditional
from milpool.simpledist import rp_triplet, triplet
import matplotlib.pyplot as plt
logging.basicConfig(level="INFO")
np.set_printoptions(suppress=True)
seed = 13456


def get_var(information):
    # print(np.linalg.eig(information)[0])
    return np.linalg.inv(information)


def plot_errors(dist, color="red", i=0):
    name = dist.__class__.__name__
    n_samples = [200*i for i in range(1, 10)]
    errors = [dist.get_square_errors(n_samples=n, n_iterations=1000) for n in n_samples]
    I = dist.estimate_fisher_information()
    print(I)
    var = get_var(I)[i, i]
    plt.axline((0, 0), slope=1/var, color=color)
    print(name, np.array(errors).shape, np.array(errors)[:, i], var)
    plt.plot(n_samples, 1/np.array(errors)[:, i], color=color)
    plt.title(name)
    plt.ylabel("1/sigma**2")
    plt.xlabel("n_samples")


def main():
    I_pure = PureMixtureConditionalRP().estimate_fisher_information()
    print("Ipure")
    print(I_pure)
    print(get_var(I_pure))
    # for i in range(4):
    #     plot_errors(rp_full_triplet[1](), color="pink", i=i)
    #     plt.show()
    # return 
    x, xy, conditional = rp_full_triplet
    xy().estimate_parameters()
    torch.manual_seed(13456)
    I_X = x().estimate_fisher_information()
    torch.manual_seed(13456)
    I_XY = xy().estimate_fisher_information()
    #print("Params: ", xy().estimate_parameters(1000))
    #n_samples = [10*i for i in range(1, 30)]
    # errors = [xy().get_square_errors(n_samples=n, n_iterations=1000) for n in n_samples]
    #print(n_samples, errors)
    #plt.plot(n_samples, 1/np.array(errors)[:, 0])
    plot_errors(xy(), color="blue")
    #plt.show()
    plot_errors(PureMixtureConditionalRP())
    plt.show()
    torch.manual_seed(13456)
    I_X_g_Y = conditional().estimate_fisher_information()
    # print(I_XY)
    print("________")
    print("X")
    print(I_X)
    print("Y|X")
    print(I_X_g_Y)
    print("SUM")
    print(I_X+I_X_g_Y)
    print("XY")
    print(I_XY)
    print(get_var(I_X))
    print(get_var(I_X_g_Y))
    print(get_var(I_XY))


main()
