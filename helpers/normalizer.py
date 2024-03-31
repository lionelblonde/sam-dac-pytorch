import torch


class RunningMoments(object):

    def __init__(self, shape: dict, device: torch.device):
        """Maintain running statistics across workers leveraging Chan's method"""
        self.count: float = 1e-4  # haxx to avoid any division by zero
        # initialize mean and var with float64 precision (objectively more accurate)
        kwargs = {"shape": shape, "dtype": torch.float64, "device": device}
        self.mean, self.std = torch.zeros(**kwargs), torch.ones(**kwargs)
        self.device = device

    def update(self, x):
        """Update running statistics using the new batch's statistics"""
        assert isinstance(x, torch.Tensor) and x.device == self.device, "must: same device"
        self.update_moments(x.double().mean(dim=0), x.double().std(dim=0), x.size(0))

    def update_moments(self,
                       batch_mean: torch.Tensor,
                       batch_std: torch.Tensor,
                       batch_count: int):
        """ Implementation of Chan's method to compute and maintain mean and variance estimates
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        # at this stage, all tensors are on the same device (batch and self stats)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        # compute new mean
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = torch.square(self.std) * self.count
        m_b = torch.square(batch_std) * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        # compute new var
        new_var = m_2 / tot_count
        # compute new count
        new_count = tot_count
        # update moments
        self.mean = new_mean
        min_var = torch.Tensor(1e-2).double()
        self.std = torch.maximum(new_var, min_var).sqrt()
        assert self.mean.device == self.device and self.std.device == self.device, "device issue"
        self.count = new_count

    def standardize(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor) and x.device == self.device, "must: same device"
        return (x - self.mean) / self.std

    def destandardize(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor) and x.device == self.device, "must: same device"
        return (x * self.std) + self.mean

    def divide_by_std(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor) and x.device == self.device, "must: same device"
        return x / self.std

    def load_state_dict(self, state_dict: dict):
        self.__dict__.update(state_dict)

    def state_dict(self):
        _state_dict = self.__dict__.copy()
        _state_dict.pop("comm")
        return _state_dict
