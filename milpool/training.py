import torch

from .models import SimpleMIL


def train_mil(X, y, pooling_func, n_iterations=1000, n_epochs=5):
    crit = torch.nn.BCEWithLogitsLoss(reduction='mean')
    model = SimpleMIL(X.shape[-1], pooling_func)
    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0)
    N = 1000
    for i in range(n_epochs*n_iterations):
        optim.zero_grad();
        rho = model(X)
        loss = crit(rho, y)
        loss.backward();
        optim.step()
        if (i+1) % n_iterations == 0:
            ps = [float(l) for l in model.linear.parameters()]
            print(-ps[1]/ps[0], [float(l) for l in model.parameters()], loss.detach().numpy())
    return model

def train_flat(X, z, pooling_func):
    model = SimpleMIL(X.shape[-1], pooling_func)
    X = X.reshape(-1, 1)
    plt.hist(X.numpy().flatten(), bins=100)
    plt.show()
    y = z.reshape(-1, 1)
    print(X.shape, y.shape)
    crit = torch.nn.BCEWithLogitsLoss(size_average=True)
    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    N = 1000
    for i in range(10*N):
        optim.zero_grad();
        rho = model.linear(X)
        loss = crit(rho, y)
        loss.backward();
        optim.step()
        if (i+1) % N == 0:
            ps = [float(l) for l in model.linear.parameters()]
            print(-ps[1]/ps[0], [float(l) for l in model.parameters()])
    return model

