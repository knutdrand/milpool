from milpool.MIL_distributions import dists, reparam_dists, RPPureConditional
import numpy as np
np.set_printoptions(suppress=True)
MILX, MILXY, MILConditional = reparam_dists
I = RPPureConditional().estimate_fisher_information(n=1000000)
print("Pure")
print(I)
print(np.linalg.inv(I))
print(np.linalg.eig(I)[0])
for dist in reparam_dists:
    I = dist().estimate_fisher_information(n=100000)
    print(dist.__name__)
    print(I)
    print(np.linalg.inv(I))


exit()
dist = MILXY()
I_XY = dist.estimate_fisher_information(n=100000)
print("XY")
print(I_XY)

dist = MILX()
I_X = dist.estimate_fisher_information(n=100000)
print("X")
print(I_X)

dist = MILConditional()
I_conditional = dist.estimate_fisher_information(n=100000)
print("Conditional")
print(I_conditional)
print(I_X+I_conditional)
