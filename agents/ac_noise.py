import math

import torch


class ActionNoise(object):

    def reset(self):  # exists even if useless for non-temporally correlated noise
        pass


class NormalActionNoise(ActionNoise):

    def __init__(self, mu, sigma):
        # additive action space Gaussian noise
        self.mu = mu
        self.sigma = sigma

    def generate(self):
        return torch.normal(self.mu, self.sigma)

    def __repr__(self):
        return f"NormalAcNoise(mu={self.mu}, sigma={self.sigma})"


class OUActionNoise(ActionNoise):

    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        """Temporally correlated noise generated via an Orstein-Uhlenbeck process,
        well-suited for physics-based models involving inertia, such as locomotion.
        Implementation is based on the following post:
        http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
        """
        self.mu = mu
        self.sigma = sigma

        self.device = self.mu.device  # grap the device we are on

        self.theta = theta
        self.dt = dt
        self.x0 = x0
        if self.x0 is not None:
            assert isinstance(self.x0, torch.Tensor)
            assert self.x0.size() == self.mu.size()  # could have picked sigma too

        # start the process
        self.reset()

    def generate(self):
        # generate noise via the process
        noise = self.prev_noise
        noise += self.theta * (self.mu - self.prev_noise) * self.dt
        sn_noise = torch.normal(mean=0., std=1., size=self.mu.shape).to(self.device)
        noise += self.sigma * math.sqrt(self.dt) * sn_noise
        self.prev_noise = noise  # update previous value to current
        return noise

    def reset(self):
        self.prev_noise = (
            self.x0 if self.x0 is not None else
            torch.zeros_like(self.mu).to(self.device)  # could have picked sigma too
        )

    def __repr__(self):
        return f"OUAcNoise(mu={self.mu}, sigma={self.sigma})"
