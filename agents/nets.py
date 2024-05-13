from collections import OrderedDict
from typing import Callable

from beartype import beartype
from einops import rearrange, pack
import torch
from torch import nn
from torch.nn import functional as ff
from torch.nn.utils.parametrizations import spectral_norm

from helpers import logger
from helpers.normalizer import RunningMoments


STANDARDIZED_OB_CLAMPS = [-5., 5.]


@beartype
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


@beartype
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


@beartype
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

    @beartype
    def __init__(self,
                 ob_shape: tuple[int, ...],
                 ac_shape: tuple[int, ...],
                 hid_size: int,
                 rms_obs: RunningMoments,
                 *,
                 wrap_absorb: bool,
                 d_batch_norm: bool,
                 spectral_norm: bool,
                 state_only: bool):
        super(Discriminator, self).__init__()
        self.wrap_absorb = wrap_absorb
        self.d_batch_norm = d_batch_norm
        self.spectral_norm = spectral_norm
        self.state_only = state_only
        if self.d_batch_norm:
            self.rms_obs = rms_obs
        else:
            logger.info("rms_obs not used by discriminator")

        # wrap absorbing ajustments
        ob_dim = ob_shape[-1]
        ac_dim = ac_shape[-1]
        if self.wrap_absorb:
            ob_dim += 1
            ac_dim += 1

        apply_sn = snwrap(use_sn=self.spectral_norm)  # spectral normalization

        # define the input dimension
        in_dim = ob_dim
        if self.state_only:
            in_dim += ob_dim
        else:
            in_dim += ac_dim

        # assemble the layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ("fc_block_1", nn.Sequential(OrderedDict([
                ("fc", apply_sn(nn.Linear(in_dim, hid_size))),
                ("nl", nn.LeakyReLU(negative_slope=0.1)),
            ]))),
            ("fc_block_2", nn.Sequential(OrderedDict([
                ("fc", apply_sn(nn.Linear(hid_size, hid_size))),
                ("nl", nn.LeakyReLU(negative_slope=0.1)),
            ]))),
        ]))
        self.d_head = nn.Linear(hid_size, 1)
        # perform initialization
        self.fc_stack.apply(init())
        self.d_head.apply(init())

    @beartype
    def forward(self, input_a, input_b):
        if self.d_batch_norm:
            # apply normalization
            if self.wrap_absorb:
                # normalize state
                input_a_ = input_a.clone()[:, 0:-1]
                input_a_ = self.rms_obs.standardize(input_a_).clamp(*STANDARDIZED_OB_CLAMPS)
                input_a, _ = pack([input_a_, rearrange(input_a[:, -1], "b -> b 1")], "b *")
                if self.state_only:
                    # normalize next state
                    input_b_ = input_b.clone()[:, 0:-1]
                    input_b_ = self.rms_obs.standardize(input_b_).clamp(*STANDARDIZED_OB_CLAMPS)
                    input_b, _ = pack([input_b_, rearrange(input_b[:, -1], "b -> b 1")], "b *")
            else:
                # normalize state
                input_a = self.rms_obs.standardize(input_a).clamp(*STANDARDIZED_OB_CLAMPS)
                if self.state_only:
                    # normalize next state
                    input_b = self.rms_obs.standardize(input_b).clamp(*STANDARDIZED_OB_CLAMPS)
        else:
            input_a = input_a.clamp(*STANDARDIZED_OB_CLAMPS)
            if self.state_only:
                input_b = input_b.clamp(*STANDARDIZED_OB_CLAMPS)
        x, _ = pack([input_a, input_b], "b *")  # concatenate along last dim
        x = self.fc_stack(x)
        return self.d_head(x)  # no sigmoid here


class Actor(nn.Module):

    @beartype
    def __init__(self,
                 ob_shape: tuple[int, ...],
                 ac_shape: tuple[int, ...],
                 rms_obs: RunningMoments,
                 max_ac: float,
                 *,
                 layer_norm: bool):
        super(Actor, self).__init__()
        ob_dim = ob_shape[-1]
        ac_dim = ac_shape[-1]
        self.rms_obs = rms_obs
        self.max_ac = max_ac
        self.layer_norm = layer_norm

        # assemble the last layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ("fc_block", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(ob_dim, 300)),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(300)),
                ("nl", nn.ReLU()),
            ]))),
        ]))
        self.a_fc_stack = nn.Sequential(OrderedDict([
            ("fc_block", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(300, 200)),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(200)),
                ("nl", nn.ReLU()),
            ]))),
        ]))
        self.a_head = nn.Linear(200, ac_dim)

        # perform initialization
        self.fc_stack.apply(init())
        self.a_fc_stack.apply(init())
        self.a_head.apply(init())

    @beartype
    def forward(self, ob):
        ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x = self.fc_stack(ob)
        return float(self.max_ac) * torch.tanh(self.a_head(self.a_fc_stack(x)))

    @beartype
    @property
    def perturbable_params(self):
        return [n for n, _ in self.named_parameters() if "ln" not in n]

    @beartype
    @property
    def non_perturbable_params(self):
        return [n for n, _ in self.named_parameters() if "ln" in n]


class Critic(nn.Module):

    @beartype
    def __init__(self,
                 ob_shape: tuple[int, ...],
                 ac_shape: tuple[int, ...],
                 rms_obs: RunningMoments,
                 *,
                 layer_norm: bool,
                 use_c51: bool,
                 c51_num_atoms: int,
                 use_qr: bool,
                 num_tau: int):
        super(Critic, self).__init__()
        ob_dim = ob_shape[-1]
        ac_dim = ac_shape[-1]
        self.rms_obs = rms_obs
        self.layer_norm = layer_norm
        self.use_c51 = use_c51
        self.c51_num_atoms = c51_num_atoms
        self.use_qr = use_qr
        self.num_tau = num_tau

        # number of head
        if self.use_c51:
            num_heads = self.c51_num_atoms
        elif self.use_qr:
            num_heads = self.num_tau
        else:
            num_heads = 1

        # assemble the last layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ("fc_block_1", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(ob_dim + ac_dim, 400)),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(400)),
                ("nl", nn.ReLU()),
            ]))),
            ("fc_block_2", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(400, 300)),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(300)),
                ("nl", nn.ReLU()),
            ]))),
        ]))
        self.head = nn.Linear(300, num_heads)

        # perform initialization
        self.fc_stack.apply(init())
        self.head.apply(init())

    @beartype
    def forward(self, ob, ac):
        ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x, _ = pack([ob, ac], "b *")
        x = self.fc_stack(x)
        x = self.head(x)
        if self.use_c51:
            # return a categorical distribution
            x = ff.log_softmax(x, dim=1).exp()
        return x
