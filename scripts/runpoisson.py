from milpool.poissondistribution import PoissonDistribution
import matplotlib.pyplot as plt
#import torch


dist = PoissonDistribution(4.)
#dist.plot()
#plt.show()
dist.plot_all_errors()
plt.show()
# dist.estimate_parameters(100)
