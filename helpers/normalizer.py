from mpi4py import MPI
import torch

from helpers.distributed_util import mpi_moments

COMM = MPI.COMM_WORLD


class RunningMoments(object):

    def __init__(self, shape, comm=COMM, *, use_mpi=False):
        """Maintain running statistics across workers leveraging Chan's method"""
        self.use_mpi = use_mpi
        self.count = 1e-4  # haxx to avoid any division by zero
        self.comm = comm
        # initialize mean and var with float64 precision (objectively more accurate)
        kwargs = {"shape": shape, "dtype": torch.float64}
        self.mean, self.std = torch.zeros(**kwargs), torch.ones(**kwargs)

    def update(self, x):
        """Update running statistics using the new batch's statistics"""

        

        if isinstance(x, torch.Tensor):
            # Clone, change x type to double (float64) and detach
            x = x.clone().detach().double().cpu().numpy()
        else:
            x = x.astype(np.float64)
        # Compute the statistics of the batch
        if self.use_mpi:
            batch_mean, batch_std, batch_count = mpi_moments(x, axis=0, comm=self.comm)
        else:
            batch_mean = np.mean(x, axis=0)
            batch_std = np.std(x, axis=0)
            batch_count = x.shape[0]
        # Update moments
        self.update_moms(batch_mean, batch_std, batch_count)

    def update_moms(self, batch_mean, batch_std, batch_count):
        """ Implementation of Chan's method to compute and maintain mean and variance estimates
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        # Compute new mean
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = np.square(self.std) * self.count
        m_b = np.square(batch_std) * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        # Compute new var
        new_var = m_2 / tot_count
        # Compute new count
        new_count = tot_count
        # Update moments
        self.mean = new_mean
        self.std = np.sqrt(np.maximum(new_var, 1e-2))
        self.count = new_count

    def standardize(self, x):
        assert isinstance(x, torch.Tensor)
        mean = torch.Tensor(self.mean).to(x)
        std = torch.Tensor(self.std).to(x)
        return (x - mean) / std

    def destandardize(self, x):
        assert isinstance(x, torch.Tensor)
        mean = torch.Tensor(self.mean).to(x)
        std = torch.Tensor(self.std).to(x)
        return (x * std) + mean

    def divide_by_std(self, x):
        assert isinstance(x, torch.Tensor)
        std = torch.Tensor(self.std).to(x)
        return x / std

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def state_dict(self):
        _state_dict = self.__dict__.copy()
        _state_dict.pop("comm")
        return _state_dict
