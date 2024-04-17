from collections import OrderedDict
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as ff
from torch.nn.utils.parametrizations import spectral_norm

from helpers import logger
from helpers.normalizer import RunningMoments


STANDARDIZED_OB_CLAMPS = [-5., 5.]


def log_module_info(model: nn.Module):

    def _fmt(n) -> str:
        if n // 10 ** 6 > 0:
            out = str(round(n / 10 ** 6, 2)) + " M"
        elif n // 10 ** 3:
            out = str(round(n / 10 ** 3, 2)) + " k"
        else:
            out = str(n)
        return out

    logger.info("logging model specs")
    logger.info(model)
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info(f"total trainable params: {_fmt(num_params)}.")


def init(constant_bias: float = 0.) -> Callable[[nn.Module], None]:
    """Perform orthogonal initialization"""

    def _init(m: nn.Module) -> None:

        if (isinstance(m, (nn.Conv2d, nn.Linear))):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, constant_bias)
        elif (isinstance(m, (nn.BatchNorm2d, nn.LayerNorm))):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    return _init


def snwrap(*, use_sn: bool = False) -> Callable[[nn.Module], nn.Module]:
    """Spectral normalization wrapper"""

    def _snwrap(m: nn.Module) -> nn.Module:
        assert isinstance(m, nn.Linear)
        if use_sn:
            return spectral_norm(m)
        return m

    return _snwrap


# models

class Discriminator(nn.Module):

    def __init__(self, ob_shape: tuple[int], ac_shape: tuple[int],
                 hps: dict, rms_obs: RunningMoments):
        super(Discriminator, self).__init__()
        self.hps = hps
        ob_dim = ob_shape[-1]
        ac_dim = ac_shape[-1]
        if self.hps.wrap_absorb:
            ob_dim += 1
            ac_dim += 1
        if self.hps["d_batch_norm"]:
            self.rms_obs = rms_obs

        apply_sn = snwrap(use_sn=self.hps["spectral_norm"])  # spectral normalization

        # define the input dimension
        in_dim = ob_dim
        if self.hps["state_only"]:
            in_dim += ob_dim
        else:
            in_dim += ac_dim

        # assemble the layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ("fc_block_1", nn.Sequential(OrderedDict([
                ("fc", apply_sn(nn.Linear(in_dim, 100))),
                ("nl", nn.Softsign()),
            ]))),
            ("fc_block_2", nn.Sequential(OrderedDict([
                ("fc", apply_sn(nn.Linear(100, 100))),
                ("nl", nn.Softsign()),
            ]))),
        ]))
        self.d_head = nn.Linear(100, 1)
        # perform initialization
        self.fc_stack.apply(init())
        self.d_head.apply(init())

    def forward(self, input_a, input_b):
        if self.hps["d_batch_norm"]:
            # apply normalization
            if self.hps.wrap_absorb:
                # normalize state
                input_a_ = input_a.clone()[:, 0:-1]
                input_a_ = self.rms_obs.standardize(input_a_).clamp(*STANDARDIZED_OB_CLAMPS)
                input_a = torch.cat([input_a_, input_a[:, -1].unsqueeze(-1)], dim=-1)
                if self.hps["state_only"]:
                    # normalize next state
                    input_b_ = input_b.clone()[:, 0:-1]
                    input_b_ = self.rms_obs.standardize(input_b_).clamp(*STANDARDIZED_OB_CLAMPS)
                    input_b = torch.cat([input_b_, input_b[:, -1].unsqueeze(-1)], dim=-1)
            else:
                # normalize state
                input_a = self.rms_obs.standardize(input_a).clamp(*STANDARDIZED_OB_CLAMPS)
                if self.hps["state_only"]:
                    # normalize next state
                    input_b = self.rms_obs.standardize(input_b).clamp(*STANDARDIZED_OB_CLAMPS)
        else:
            input_a = input_a.clamp(*STANDARDIZED_OB_CLAMPS)
            if self.hps["state_only"]:
                input_b = input_b.clamp(*STANDARDIZED_OB_CLAMPS)
        # concatenate
        x = torch.cat([input_a, input_b], dim=-1)
        x = self.fc_stack(x)
        return self.d_head(x)  # no sigmoid here


class Actor(nn.Module):

    def __init__(self, ob_shape: tuple[int], ac_shape: tuple[int],
                 hps: dict, rms_obs: RunningMoments, max_ac: float):
        super(Actor, self).__init__()
        ob_dim = ob_shape[-1]
        ac_dim = ac_shape[-1]
        self.layer_norm = hps["layer_norm"]
        self.rms_obs = rms_obs
        self.max_ac = max_ac

        # assemble the last layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ("fc_block", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(ob_dim, 300)),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(300)),
                ("nl", nn.Softsign()),
            ]))),
        ]))
        self.a_fc_stack = nn.Sequential(OrderedDict([
            ("fc_block", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(300, 200)),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(200)),
                ("nl", nn.Softsign()),
            ]))),
        ]))
        self.a_head = nn.Linear(200, ac_dim)

        # perform initialization
        self.fc_stack.apply(init())
        self.a_fc_stack.apply(init())
        self.a_head.apply(init())

    def forward(self, ob):
        ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x = self.fc_stack(ob)
        return float(self.max_ac) * torch.tanh(self.a_head(self.a_fc_stack(x)))

    @property
    def perturbable_params(self):
        return [n for n, _ in self.named_parameters() if "ln" not in n]

    @property
    def non_perturbable_params(self):
        return [n for n, _ in self.named_parameters() if "ln" in n]


class Critic(nn.Module):

    def __init__(self, ob_shape: tuple[int], ac_shape: tuple[int],
                 hps: dict, rms_obs: RunningMoments):
        super(Critic, self).__init__()
        ob_dim = ob_shape[-1]
        ac_dim = ac_shape[-1]
        self.use_c51 = hps["use_c51"]
        self.layer_norm = hps["layer_norm"]
        self.rms_obs = rms_obs

        if self.use_c51:
            num_heads = hps["c51_num_atoms"]
        elif hps["use_qr"]:
            num_heads = hps["num_tau"]
        else:
            num_heads = 1

        # assemble the last layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ("fc_block_1", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(ob_dim + ac_dim, 400)),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(400)),
                ("nl", nn.Softsign()),
            ]))),
            ("fc_block_2", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(400, 300)),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(300)),
                ("nl", nn.Softsign()),
            ]))),
        ]))
        self.head = nn.Linear(300, num_heads)

        # perform initialization
        self.fc_stack.apply(init())
        self.head.apply(init())

    def forward(self, ob, ac):
        ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x = torch.cat([ob, ac], dim=-1)
        x = self.fc_stack(x)
        x = self.head(x)
        if self.use_c51:
            # return a categorical distribution
            x = ff.log_softmax(x, dim=1).exp()
        return x
