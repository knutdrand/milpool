from milpool.distributions import MixtureX
import matplotlib.pyplot as plt
import torch
dist = MixtureX(mu_2=torch.tensor(2.), w=0.2)
dist.plot()
plt.show()
dist.estimate_parameters(100)
