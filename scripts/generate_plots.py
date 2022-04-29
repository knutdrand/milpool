from milpool.distributions import NormalDistribution, MixtureX, MixtureXY, MixtureConditional, rp_full_triplet, PureMixtureConditionalRP, full_triplet, ab_full_triplet
from milpool.MIL_distributions import reparam_dists
import numpy as np
import torch
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)


def get_var(information):
    # print(np.linalg.eig(information)[0])
    print("#--------->", np.linalg.det(information))
    return np.linalg.inv(information)


def show_fig(name, filename=None):
    plt.title(name)

    if filename is not None:
        plt.savefig("images/"+filename)
    plt.show()


def plot_dists(params):
    mx = MixtureX(*params)
    p = mx.prob(x)
    plt.plot(x, p)
    show_fig(f"P(X) {params}", f"px{params}.png")
    plt.hist(mx.sample(n=1000))
    show_fig(f"X {params}", f"hist_x{params}.png")
    mxy = MixtureXY(*params)
    p_pos = mxy.prob(x, 0)
    p_neg = mxy.prob(x, 1)
    plt.plot(x, p_pos)
    plt.plot(x, p_neg)
    show_fig(f"P(x, y) {params}", f"pxy{params}.png")
    x_s, y = mxy.sample(n=1000)
    x_s = np.array(x_s).ravel()
    y = np.array(y).ravel()
    plt.hist(x_s[y==0], alpha=0.6)
    plt.hist(x_s[y==1], alpha=0.6)
    show_fig(f"X, Y {params}", f"hist_xy{params}.png")
    condxy = MixtureConditional(*params)
    p_cond = condxy.prob(x, torch.tensor(1))
    plt.plot(np.array(x), np.array(p_cond))
    show_fig(f"P(Y | X) {params}", f"pygx{params}.png")
    plt.scatter(x_s, y)
    show_fig("Y | X {params}", f"hist_ygx{params}.png")


def _plot_errors(dist, color="red", i=0):
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


def plot_all_errors(dist, color="red", n_params=None):
    if n_params is None:
        n_params = len(dist.params)
    name = dist.__class__.__name__
    I = dist.estimate_fisher_information()
    I = I[:n_params, :n_params]
    all_var = get_var(I)
    print(I)
    print(all_var)
    n_samples = [200*i for i in range(1, 10)]
    errors = [dist.get_square_errors(n_samples=n, n_iterations=200, do_plot=False) for n in n_samples]
    dist.get_square_errors(n_samples=n_samples[-1], n_iterations=200, do_plot=True)
    fig, axes = plt.subplots((n_params+1)//2, 2)
    if (n_params+1)//2 == 1:
        axes = [axes]
    for i, param in enumerate(dist.params[:n_params]):
        var = all_var[i, i]
        ax = axes[i//2][i % 2]
        ax.axline((0, 0), slope=1/var, color=color, label=name+" CRLB")
        ax.plot(n_samples, 1/np.array(errors)[:, i], color=color, label=name+" errors")
        ax.set_ylabel("1/sigma**2")
        ax.set_xlabel("n_samples")


def plot_errors(dist, color="red", i=0, inverse=True):
    name = dist.__class__.__name__
    I = dist.estimate_fisher_information()
    var = get_var(I)[i, i]
    print(I, get_var(I))
    n_samples = [200*i for i in range(1, 10)]
    errors = [dist.get_square_errors(n_samples=n, n_iterations=100) for n in n_samples]
    print(errors)
    if inverse:
        plt.axline((0, 0), slope=1/var if inverse else var, color=color, label=name+" CRLB")
    plt.plot(n_samples, 1/np.array(errors)[:, i] if inverse else np.array(errors)[:, i], color=color, label=name+" errors")
    plt.ylabel("1/sigma**2" if inverse else "sigma**2")
    plt.xlabel("n_samples")


def plot_neg():
    for params in all_params:
        mxy = MixtureXY(*params)
        p_pos = mxy.prob(x, 0)
        plt.plot(x, p_pos)
        show_fig(f"P neg {params}", f"neg{params}.png")
        x_s, y = mxy.sample(n=1000)
        plt.hist(np.array(x_s).ravel()[np.array(y).ravel()==0], alpha=0.6)
        show_fig(f"Hist neg {params}", f"histneg{params}.png")


def plot_erros_vs_cramer(i=1):
    params = all_params[0]
    for params in all_params:
        plot_errors(rp_full_triplet[i](*params), color="green")
        plot_errors(PureMixtureConditionalRP(*params), color="red")
        plt.legend()
        show_fig(f"Errors vs Cramer-Rao {params}", f"CR{params}.png")
    
        plot_errors(rp_full_triplet[i](*params), color="green", inverse=False)
        plot_errors(PureMixtureConditionalRP(*params), color="red", inverse=False)
        plt.legend()
        show_fig(f"Errors vs Cramer-Rao {params} (inv)", f"CRinv{params}.png")


def plot_info_vs_sigma():
    Is = []
    sigmas = torch.linspace(0.4, 1, 10)
    params = [(-1., 1., sigma, 0.3333) for sigma in sigmas]
    for rp_dist in rp_full_triplet:
        torch.manual_seed(123465)
        Is.append(np.array([rp_dist(*param).estimate_fisher_information(n=10000)[0, 0] for param in params]))
        plt.plot(sigmas, Is[-1], label=rp_dist.__name__)
    plt.xlabel("sigma")
    plt.ylabel("I")
    plt.legend()
    show_fig("Relative Information", "relative_information.png")
    plt.plot(sigmas, Is[2]/Is[1], label="conditional_ratio")
    show_fig("ratio")


def plot_witness_curve():
    one_ofs = np.arange(1, 10)
    X, XY, C = reparam_dists
    for n_pos in [10, 30, 70]:
        witness_rates = []
        I_xy = []
        I_x = []
        for one_of in one_ofs:
            w = 1/one_of
            witness_rates.append(w)
            group_size = one_of*n_pos
            xy = XY(-1., 1., 1.,  w)
            xy.group_size = group_size
            I_xy.append(xy.estimate_fisher_information(n=10000)[0, 0])
            c = C(-1., 1., 1.,  w)
            c.group_size = group_size
            I_x.append(c.estimate_fisher_information(n=10000)[0, 0])

        plt.plot(witness_rates, np.array(I_xy), label="FullDist")
        plt.plot(witness_rates, np.array(I_x), label="Conditional")
        plt.legend()
        plt.xlabel("Witness rate")
        plt.ylabel("I")
        show_fig(f"Information vs witness rate (n_pos={n_pos})", f"w{n_pos}.png")


for dist in ab_full_triplet:
    plot_all_errors(dist(), color="red", n_params=2)
    plt.show()
exit()
plot_errors(rp_full_triplet[0](), color="red", i=0, inverse=True)
plt.show()
# x = torch.linspace(-5, 5, 100)
#p = NormalDistribution().prob(x)

# plot_erros_vs_cramer()
# all_params = [(-1., 1., 0.5, 0.33333), (0., 1., 1.0, 0.33333)]
# plot_neg()
# plot_witness_curve()
