import torch
from copy import deepcopy
from .visualization import plot_distributions_2d
from itertools import chain
def train_mil(X, y, model, n_iterations=1000, n_epochs=5):
    params = [
        {'params': model.instance_model.parameters(), "lr": 1e-1, "momentum": 0.0},
        {'params': model.pooling.parameters(), "lr": 1e-2, 'momentum': 0}]

    crit = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optim2 = torch.optim.SGD(params)
    optim_instance = torch.optim.Adam(model.instance_model.parameters(), lr=1e-2)#, weight_decay=1e-5)
    optim2 = optim_instance
    # optim_instance = torch.optim.Adam(chain(model.instance_model.parameters(), model.adjuster.parameters()), lr=1e-2)#, weight_decay=1e-5)
    #optim_pooling = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0)
    pooling_params = bool(list(model.pooling.parameters()))
    if pooling_params:    
        optim_pooling = torch.optim.SGD(model.pooling.parameters(), lr=1e-2, momentum=0)
    #optim3 = torch.optim.Adam(model.parameters(), lr=1e-2, momentum=0)
    N = 1000
    l = 0.0001
    best_loss = None
    best_model = None
    for i in range(n_epochs*n_iterations):
        #optim = optim1 if i == 0 else optim2
        # optim_instance.zero_grad()
        optim2.zero_grad()
        if pooling_params and False:    
            optim_pooling.zero_grad()
        rho = model(X)
        if pooling_params and False:
            loss = crit(rho, y) - model.pooling.logOw*l
        else:
            loss = crit(rho, y)
        fl = float(loss.detach())
        if best_loss is None or fl<best_loss:
            best_loss = fl
            best_model = deepcopy(model)
        loss.backward();
        # optim_instance.step()
        optim2.step()
        if pooling_params and False:    
            optim_pooling.step()
        if (i+1) % n_iterations == 0:
            # ps = [float(l) for l in model.linear.parameters()]
            ps = [p.detach().numpy() for p in model.parameters()]
            # print(ps[2:], loss.detach().numpy(), ps[1].shape)
            print(float(loss.detach().numpy()), [float(torch.sigmoid(p.detach()).numpy()) for p in model.pooling.parameters()])
            print(ps)
            # plot_distributions_2d(X, model.instance_model(X)>0)
            #Wx + b = 0
            #Wx=-b
            #x=W**(-1)(-b)
            # print([p.detach().numpy() for p in model.parameters()], loss.detach().numpy())
            # print(loss.detach().numpy())
    return best_model
    return model

def train_flat(X, z, model):
    X = X.reshape(-1, X.shape[-1])
    y = z.reshape(-1, 1)
    crit = torch.nn.BCEWithLogitsLoss(reduction="mean")
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    N = 1000
    for i in range(5*N):
        optim.zero_grad();
        rho = model.instance_model(X)
        loss = crit(rho, y)
        loss.backward();
        optim.step()
        if (i+1) % N == 0 and False:
            ps = [float(l) for l in model.instance_model.parameters()]
            print(-ps[1]/ps[0], [float(l) for l in model.parameters()])
    return model.instance_model

