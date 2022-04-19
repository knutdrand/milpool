from distributions import MixtureXY
import torch


class MILXY(MixtureXY):
    q: float = torch.tensor(0.5)
    group_size: float = 10

    def sample(self, n=1):
        y = torch.bernoulli(self.q*torch.ones(n)[:, None])
        z = torch.bernoulli(y*torch.ones(self.group_size)*self.w)
        mu = self.mu_1*z+self.mu_2*(1-z)
        return torch.normal(mu, self.sigma), y

    def log_likelihood(self, x, y, mu_1, mu_2, sigma, w):
        l1 = torch.log(w)+np.log(1/np.sqrt(2*np.pi))-torch.log(sigma) -(x-mu_1)**2/(2*sigma**2)
        l2 = torch.log(1-w)+np.log(1/np.sqrt(2*np.pi))-torch.log(sigma) -(x-mu_2)**2/(2*sigma**2)
        return y*l1+(1-y)*l2

    def get_square_errors(self, n_samples=1000, n_iterations=1000):
        estimates = np.array([self.estimate_parameters(n_samples) for _ in range(n_iterations)])
        true_params = np.array(self.params)
        return ((estimates-true_params)**2).sum(axis=0)/n_iterations

    def estimate_parameters(self, n=1000):
        x, y = self.sample(n)
        x = np.array(x)
        y = np.array(y)

        group_2 = x[y == 0]
        group_1 = x[y == 1]
        mu_1 = np.mean(group_1)
        mu_2 = np.mean(group_2)
        sigma = np.sqrt((np.sum((group_1-mu_1)**2) + np.sum((group_2-mu_2)**2))/x.size)
        w = group_1.size/y.size
        return (mu_1, mu_2, sigma, w)

