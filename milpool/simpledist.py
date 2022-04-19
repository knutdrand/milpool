from .distributions import *

class SimpleMixtureXY(MixtureXY):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = (self.mu_1, self.mu_2)

    def log_likelihood(self, x, y, mu_1, mu_2):
        return super().log_likelihood(x, y, mu_1, mu_2, self.sigma, self.w)

class SimpleMixtureX(MixtureX):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = (self.mu_1, self.mu_2)

    def log_likelihood(self, x, mu_1, mu_2):
        return super().log_likelihood(x, mu_1, mu_2, self.sigma, self.w)

class SimpleMixtureConditional(MixtureConditional):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = (self.mu_1, self.mu_2)

    def log_likelihood(self, x, y, mu_1, mu_2):
        return super().log_likelihood(x, y, mu_1, mu_2, self.sigma, self.w)

class SimpleMixtureXYRP(MixtureXY):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mu_1, mu_2, sigma, w = (self.mu_1, self.mu_2, self.sigma, self.w)
        self.alpha = (mu_2**2-mu_1**2)/(2*sigma**2)+torch.log(w/(1-w))
        self.beta = (mu_1-mu_2)/sigma**2
        self.params = (self.alpha, self.beta)
        self.logodds_w = torch.log(self.w/(1-self.w))
        
    def get_mu_1(self, alpha, beta):
        return -(alpha-self.logodds_w)/beta+beta*self.sigma**2/2

    def get_mu_2(self, alpha, beta):
        return -(alpha-self.logodds_w)/beta-beta*self.sigma**2/2

    def log_likelihood(self, x, y, alpha, beta):
        return super().log_likelihood(x, y, self.get_mu_1(alpha, beta), self.get_mu_2(alpha, beta), self.sigma, self.w)

class SimpleMixtureXRP(MixtureX):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mu_1, mu_2, sigma, w = (self.mu_1, self.mu_2, self.sigma, self.w)
        self.alpha = (mu_2**2-mu_1**2)/(2*sigma**2)+torch.log(w/(1-w))
        self.beta = (mu_1-mu_2)/sigma**2
        self.params = (self.alpha, self.beta)
        self.logodds_w = torch.log(self.w/(1-self.w))


    def get_mu_1(self, alpha, beta):
        return -(alpha-self.logodds_w)/beta+beta*self.sigma**2/2

    def get_mu_2(self, alpha, beta):
        return -(alpha-self.logodds_w)/beta-beta*self.sigma**2/2

    def log_likelihood(self, x, alpha, beta):
        return super().log_likelihood(x, self.get_mu_1(alpha, beta), self.get_mu_2(alpha, beta), self.sigma, self.w)

class SimpleMixtureConditionalRP(MixtureConditional):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mu_1, mu_2, sigma, w = (self.mu_1, self.mu_2, self.sigma, self.w)
        self.alpha = (mu_2**2-mu_1**2)/(2*sigma**2)+torch.log(w/(1-w))
        self.beta = (mu_1-mu_2)/sigma**2
        self.params = (self.alpha, self.beta)

    def log_likelihood(self, x, y, alpha, beta):
        eta = alpha+beta*x
        return y*torch.log(torch.sigmoid(eta))+(1-y)*torch.log(torch.sigmoid(-eta))

triplet = (SimpleMixtureX, SimpleMixtureXY, SimpleMixtureConditional)
rp_triplet = (SimpleMixtureXRP, SimpleMixtureXYRP, SimpleMixtureConditionalRP)
"""reparam
a = (mu_1**2-mu_2**2)/(2*sigma**2) + inv_sigma(w)
b = (mu_2-mu_1)/sigma**2

a = -b(mu_2+mu_1)/2+inv_sigma(w)
(a-is(w))/(-b)*2 = mu_2+mu_1

mu_2-mu_1 = b*sigma**2

(a-is(w))/(-b)*2+b*sigma**2= 2mu_2
"""
