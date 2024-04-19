from beartype import beartype
import torch


class AdaptiveParamNoise(object):

    @beartype
    def __init__(self, device: torch.device, initial_std: float = 0.1, delta: float = 0.1):
        """Adaptive parameter noise, as introduced in the paper
        "Parameter Space Noise for Exploration"
        Matthias Plappert, https://arxiv.org/abs/1706.01905

        Args:
            initial_std (float): Initial parameter noise standard deviation
            delta (float): Threshold used in the adaptive noise scaling heuristic
        """
        # create tensor from float data: we need use torch.tensor, not torch.Tensor
        self.initial_std = torch.tensor(initial_std).to(device)
        self.delta = torch.tensor(delta).to(device)
        self.cur_std = torch.tensor(initial_std).to(device)  # initialize the current std

    @beartype
    def adapt_std(self, dist: torch.Tensor):
        """Adapt the parameter noise standard deviation based on distance `dist`
            `dist`: distance between the actions predicted respectively by the actor and
                    the adaptive-parameter-noise-perturbed actor (action space distance).
        Iteratively multiplying/dividing the standard deviation by this distance can be
        interpreted as adapting the scale of the parameter space noise over time and
        relating it the variance in action space that it induces.
        This heuristic is based on the Levenberg-Marquardt heuristic.

        A good choice of `delta` is the std of the desired action space additive normal noise,
        as it results in effective action space noise that has the same std as regular
        Gaussian action space noise (holds only because `dist` is an l2 distance in action space).
        (cf. section "Adaptive noise scaling" in paper)
        """
        assert isinstance(dist, torch.Tensor)
        if dist < self.delta:  # increase standard deviation
            self.cur_std *= 1.01
        else:  # decrease standard deviation
            self.cur_std /= 1.01

    @beartype
    def adapt_delta(self, new_delta: torch.Tensor):
        """Adapt the threshold delta when following an eps-greedy heuristic"""
        assert isinstance(new_delta, torch.Tensor)
        self.delta = new_delta

    @beartype
    def __repr__(self):
        return f"AdaptiveParamNoise(initial_std={self.initial_std}, delta={self.delta})"
