import torch
from sklearn.metrics import confusion_matrix

def test(model, instances, labels):
    crit = torch.nn.BCELoss(reduction='mean')
    p = torch.sigmoid(model(instances))
    print(p.dtype, labels.dtype)
    print(crit(p, labels))
    print(confusion_matrix((p>0.5).numpy(),(labels>0.5).numpy()))

def plotter(X, ys, zs):
    import math
    q = global_q
    w = global_W
    px = lambda x: w*q*norm.pdf(x, 2, 1)+(1-w*q)*norm.pdf(x, 0, 1)
    pi = lambda x: w*q*norm.pdf(x, 2, 1)/px(x)

    logOi = lambda x: np.log(pi(x)/(1-pi(x)))
    Oy = lambda xs: q/(1-q)*math.prod((1-w)+(1-q*w)/q*Oi(xs))

    g = lambda I: np.log((1-w)+(1-q*w)/q*np.exp(logOi(I)))
    h = lambda S: np.log(q/(1-q))+S
    logOy = lambda xs: h(np.sum(g(xs)))
    py = np.array([logOy(x) for x in X.numpy()]).flatten()
    args = np.argsort(py)
    py = py[args]
    y_now = ys.numpy().flatten()[args]
    print(y_now.shape)
    meaned = np.convolve(y_now, np.ones(200)/200, mode='same')
    plt.plot(py, np.log(meaned/(1-meaned)),)
    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    plt.show()


    f = lambda x: px(x)*(q+pi(x)-q*pi(x)-q*w)/(w*q*(1-w*q))
    f1 = lambda x: px(x)*((1-w)*(1-pi(x))/(1-q*w)+w*pi(x)/(q*w))
    f0 = lambda x: px(x)*(1-pi(x))/(1-q*w)
    x = np.arange(-40, 40)/10
    y = f0(x)
    xs = [x for xs, y in zip(X.numpy(),ys.numpy()) for x in xs.flatten() if y==0]
    # xs = [x for xs, y in zip(X.numpy(),ys.numpy()) for x in xs.flatten()]
    plt.hist(xs, bins=200, density=True)
    plt.plot(x, y)
    plt.show()
