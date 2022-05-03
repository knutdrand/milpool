from milpool.poissondistribution import PoissonDistribution
import matplotlib.pyplot as plt
#import torch


#dist = PoissonDistribution(4.)# [4., 1.])
#dist.plot_all_errors()
dist = PoissonDistribution([4., 1., 3., 2.])
#dist.plot()
#plt.show()
dist.plot_all_errors(n_params=2)
plt.show()
# dist.estimate_parameters(100)
