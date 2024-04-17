import torch


class ActionNoise(object):

    def reset(self):  # exists even if useless for non-temporally correlated noise
        pass


class NormalActionNoise(ActionNoise):

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor):
        """Additive action space Gaussian noise"""
        assert isinstance(mu, torch.Tensor) and isinstance(sigma, torch.Tensor)
        self.mu = mu
        self.sigma = sigma
        self.device = self.mu.device  # grap the device we are on (assumed sigma and mu on same)

    def generate(self):
        return torch.normal(self.mu, self.sigma).to(self.device)

    def __repr__(self):
        return f"NormalAcNoise(mu={self.mu}, sigma={self.sigma})"
