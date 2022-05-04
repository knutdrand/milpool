from milpool.poissondistribution import PoissonDistribution, PoissonMixtureDistribution
from milpool.reparametrization import np_reparametrize
from milpool.distributions import MidPointReparam
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
#import torch


#dist = PoissonDistribution(4.)# [4., 1.])
#dist.plot_all_errors()
if False:
    dist = PoissonDistribution([4., 1., 3., 2.])
    dist.plot_all_errors(n_params=4)
    plt.show()
p1 = PoissonDistribution([1., 3.])
p2 = PoissonDistribution([10., 12.])
# p1 = PoissonDistribution([1.])
# p2 = PoissonDistribution([10.])

print(p1.params, p2.params)
mix = PoissonMixtureDistribution(
    [p1, p2], [0.1, 0.9])

#mix.plot()
#plt.show()

mix.plot_all_errors(n_params=(2+4))
plt.show()
# dist.estimate_parameters(100)
