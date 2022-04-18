import torch
import numpy as np
from dataclasses import dataclass
import logging
np.set_printoptions(suppress=True)
from milpool.distributions import *
from milpool.simpledist import *
#logging.basicConfig(level="DEBUG")
logging.basicConfig(level="INFO")

seed = 13456


# print(NormalDistribution().estimate_fisher_information())
# x = torch.arange(-10, 10)
# y = torch.tensor(5)
# print(MixtureX().l(x)+MixtureConditional().l(x, y)/MixtureXY().l(x, y))
# print(MixtureXY().l(x, y))
def main():
    I_X = MixtureX().estimate_fisher_information()
    I_XY = MixtureXY().estimate_fisher_information()
    I_X_g_Y = MixtureConditional().estimate_fisher_information()
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

def get_var(I):
    print(np.linalg.eig(I)[0])
    return np.linalg.inv(I)
        

def simple_main():
    x, xy, conditional = rp_triplet
    I_X = x().estimate_fisher_information()
    I_XY = xy().estimate_fisher_information()
    I_X_g_Y = conditional().estimate_fisher_information()
    # print(I_XY)
    print("________")
    print("X")
    print(I_X, get_var(I_X))
    print("Y|X")
    print(I_X_g_Y, get_var(I_X_g_Y))
    print("SUM")
    print(I_X+I_X_g_Y)
    print("XY")
    print(I_XY, get_var(I_XY))
    
simple_main()
