from collections import defaultdict
import os
import os.path as osp

import numpy as np
import torch
from torch.nn.utils import clip_grad as cg
import torch.nn.functional as ff
from torch.utils.data import DataLoader
from torch import autograd
from torch.autograd import Variable

from helpers import logger
from helpers.console_util import log_module_info
from helpers.dataset import Dataset
from helpers.math_util import huber_quant_reg_loss, LRScheduler
from helpers.distributed_util import average_gradients, sync_with_root, RunMoms
from agents.nets import Actor, Critic, Discriminator
from agents.param_noise import AdaptiveParamNoise
from agents.ac_noise import NormalActionNoise, OUActionNoise


debug_lvl = os.environ.get('DEBUG_LVL', 0)
try:
    debug_lvl = np.clip(int(debug_lvl), a_min=0, a_max=3)
except ValueError:
    debug_lvl = 0
DEBUG = bool(debug_lvl >= 2)


class SPPAgent(object):

    def __init__(self, shapes, max_ac, device, hps, expert_dataset, replay_buffer):

        self.ob_shape, self.ac_shape = shapes['ob_shape'], shapes['ac_shape']
        self.max_ac = max_ac

        self.device = device
        self.hps = hps

        assert self.hps.lookahead > 1 or not self.hps.n_step_returns
        assert self.hps.rollout_len <= self.hps.batch_size
        if self.hps.clip_norm <= 0:
            logger.info(f"clip_norm={self.hps.clip_norm} <= 0, hence disabled.")

        # grab demo dataset
        self.expert_dataset = expert_dataset

        # know whether eval or not
        self.eval_mode = self.expert_dataset is None

        # grab replay buffer
        self.replay_buffer = replay_buffer

        # Define critic to use
        assert sum([self.hps.use_c51, self.hps.use_qr]) <= 1
        if self.hps.use_c51:
            assert not self.hps.clipped_double
            c51_supp_range = (self.hps.c51_vmin,
                              self.hps.c51_vmax,
                              self.hps.c51_num_atoms)
            self.c51_supp = torch.linspace(*c51_supp_range).to(self.device)
            self.c51_delta = ((self.hps.c51_vmax - self.hps.c51_vmin) /
                              (self.hps.c51_num_atoms - 1))
        elif self.hps.use_qr:
            assert not self.hps.clipped_double
            qr_cum_density = np.array([((2 * i) + 1) / (2.0 * self.hps.num_tau)
                                       for i in range(self.hps.num_tau)])
            qr_cum_density = torch.Tensor(qr_cum_density).to(self.device)
            self.qr_cum_density = qr_cum_density.view(1, 1, -1, 1).expand(self.hps.batch_size,
                                                                          self.hps.num_tau,
                                                                          self.hps.num_tau,
                                                                          -1).to(self.device)
        else:
            # if using neither distributional rl variant, use clipped double
            assert self.hps.clipped_double

        # Parse the noise types
        self.param_noise, self.ac_noise = self.parse_noise_type(self.hps.noise_type)

        # Create observation normalizer that maintains running statistics
        self.rms_obs = RunMoms(shape=self.ob_shape, use_mpi=True)

        assert self.hps.ret_norm or not self.hps.popart
        assert not (self.hps.use_c51 and self.hps.ret_norm)
        assert not (self.hps.use_qr and self.hps.ret_norm)
        if self.hps.ret_norm:
            # Create return normalizer that maintains running statistics
            self.rms_ret = RunMoms(shape=(1,), use_mpi=False)

        # Create online and target nets, and initilize the target nets
        self.actr = Actor(self.env, self.hps, self.rms_obs).to(self.device)
        sync_with_root(self.actr)
        self.targ_actr = Actor(self.env, self.hps, self.rms_obs).to(self.device)
        self.targ_actr.load_state_dict(self.actr.state_dict())
        self.crit = Critic(self.env, self.hps, self.rms_obs).to(self.device)
        sync_with_root(self.crit)
        self.targ_crit = Critic(self.env, self.hps, self.rms_obs).to(self.device)
        self.targ_crit.load_state_dict(self.crit.state_dict())
        if self.hps.clipped_double:
            # Create second ('twin') critic and target critic
            # TD3, https://arxiv.org/abs/1802.09477
            self.twin = Critic(self.env, self.hps, self.rms_obs).to(self.device)
            sync_with_root(self.twin)
            self.targ_twin = Critic(self.env, self.hps, self.rms_obs).to(self.device)
            self.targ_twin.load_state_dict(self.twin.state_dict())

        if self.param_noise is not None:
            # Create parameter-noise-perturbed ('pnp') actor
            self.pnp_actr = Actor(self.env, self.hps, self.rms_obs).to(self.device)
            self.pnp_actr.load_state_dict(self.actr.state_dict())
            # Create adaptive-parameter-noise-perturbed ('apnp') actor
            self.apnp_actr = Actor(self.env, self.hps, self.rms_obs).to(self.device)
            self.apnp_actr.load_state_dict(self.actr.state_dict())

        # Set up the optimizers
        self.actr_opt = torch.optim.Adam(self.actr.parameters(),
                                         lr=self.hps.actor_lr)
        self.crit_opt = torch.optim.Adam(self.crit.parameters(),
                                         lr=self.hps.critic_lr,
                                         weight_decay=self.hps.wd_scale)
        if self.hps.clipped_double:
            self.twin_opt = torch.optim.Adam(self.twin.parameters(),
                                             lr=self.hps.critic_lr,
                                             weight_decay=self.hps.wd_scale)

        # Set up lr scheduler
        self.actr_sched = LRScheduler(
            optimizer=self.actr_opt,
            initial_lr=self.hps.actor_lr,
            lr_schedule=self.hps.lr_schedule,
            total_num_steps=self.hps.num_timesteps,
        )

        if not self.eval_mode:
            # Set up demonstrations dataset
            self.e_batch_size = min(len(self.expert_dataset), self.hps.batch_size)
            self.e_dataloader = DataLoader(
                self.expert_dataset,
                self.e_batch_size,
                shuffle=True,
                drop_last=True,
            )
            assert len(self.e_dataloader) > 0
            # Create discriminator
            self.disc = Discriminator(self.env, self.hps, self.rms_obs).to(self.device)
            sync_with_root(self.disc)
            # Create optimizer
            self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=self.hps.d_lr)

        log_module_info(logger, 'actr', self.actr)
        log_module_info(logger, 'crit', self.crit)
        if self.hps.clipped_double:
            log_module_info(logger, 'twin', self.crit)
        if not self.eval_mode:
            log_module_info(logger, 'disc', self.disc)

    def norm_rets(self, x):
        """Standardize if return normalization is used, do nothing otherwise"""
        if self.hps.ret_norm:
            return self.rms_ret.standardize(x)
        else:
            return x

    def denorm_rets(self, x):
        """Standardize if return denormalization is used, do nothing otherwise"""
        if self.hps.ret_norm:
            return self.rms_ret.destandardize(x)
        else:
            return x

    def parse_noise_type(self, noise_type):
        """Parse the `noise_type` hyperparameter"""
        param_noise, ac_noise = None, None
        logger.info("parsing noise type")
        # parse the comma-seprated (with possible whitespaces) list of noise params
        for cur_noise_type in noise_type.split(','):
            cur_noise_type = cur_noise_type.strip()  # remove all whitespaces (start and end)
            # if the specified noise type is literally 'none'
            if cur_noise_type == 'none':
                pass
            # if 'adaptive-param' is in the specified string for noise type
            elif 'adaptive-param' in cur_noise_type:
                # set parameter noise
                _, std = cur_noise_type.split('_')
                param_noise = AdaptiveParamNoise(initial_std=float(std), delta=float(std))
                logger.info(f"{param_noise} configured")
            elif 'normal' in cur_noise_type:
                _, std = cur_noise_type.split('_')
                # spherical (isotropic) gaussian action noise
                mu = torch.zeros(self.ac_shape[0]).to(self.device)
                sigma = float(std) * torch.ones(self.ac_shape[0]).to(self.device)
                ac_noise = NormalActionNoise(mu=mu, sigma=sigma)
                logger.info(f"{ac_noise} configured")
            elif 'ou' in cur_noise_type:
                _, std = cur_noise_type.split('_')
                # Ornstein-Uhlenbeck action noise
                mu = torch.zeros(self.ac_shape[0]).to(self.device)
                sigma = float(std) * torch.ones(self.ac_shape[0]).to(self.device)
                ac_noise = OUActionNoise(mu=mu, sigma=sigma)
                logger.info(f"{ac_noise} configured")
            else:
                raise RuntimeError(f"unknown noise type: '{cur_noise_type}'")
        return param_noise, ac_noise

    def store_transition(self, transition):
        """Store the transition in memory and update running moments"""
        # Store transition in the replay buffer
        self.replay_buffer.append(transition)
        # Update the observation normalizer
        _state = transition['obs0']
        if self.hps.wrap_absorb:
            if np.all(np.equal(_state, np.append(np.zeros_like(_state[0:-1]), 1.))):
                # if this is a "flag" state (wrap absorbing): not used to update rms_obs
                return
            _state = _state[0:-1]
        self.rms_obs.update(_state)

    def sample_batch(self):
        """Sample a batch of transitions from the replay buffer"""

        # create patcher if needed
        def _patcher(x, y, z):
            return self.get_syn_rew(x, y, z).numpy(force=True)

        # get a batch of transitions from replay buffer
        if self.hps.n_step_returns:
            batch = self.replay_buffer.lookahead_sample(
                self.hps.batch_size,
                self.hps.lookahead,
                self.hps.gamma,
                patcher=_patcher if self.hps.historical_patching else None,
            )
        else:
            batch = self.replay_buffer.sample(
                self.hps.batch_size,
                patcher=_patcher if self.hps.historical_patching else None,
            )
        return batch

    def predict(self, ob, apply_noise):
        # predict an action, with or without perturbation

        # create tensor from the state (`require_grad=False` by default)
        ob = torch.Tensor(ob[None]).to(self.device)
        if apply_noise and self.param_noise is not None:
            # predict following a parameter-noise-perturbed actor
            ac = self.pnp_actr.act(ob)
        else:
            # predict following the non-perturbed actor
            ac = self.actr.act(ob)
        # Place on cpu and collapse into one dimension
        ac = ac.cpu().detach().numpy().flatten()
        if apply_noise and self.ac_noise is not None:
            # apply additive action noise once the action has been predicted,
            # in combination with parameter noise, or not.
            noise = self.ac_noise.generate()
            ac += noise
        ac = ac.clip(-self.max_ac, self.max_ac)
        return ac

    def remove_absorbing(self, x):
        non_absorbing_rows = []
        for j, row in enumerate([x[i, :] for i in range(x.shape[0])]):
            if torch.all(torch.eq(row, torch.cat([torch.zeros_like(row[0:-1]),
                                                  torch.Tensor([1.]).to(self.device)], dim=-1))):
                logger.info(f"removing absorbing row (#{j})")
                pass
            else:
                non_absorbing_rows.append(j)
        return x[non_absorbing_rows, :], non_absorbing_rows

    def update_actor_critic(self, batch, update_actor, iters_so_far):

        metrics = defaultdict(list)

        # transfer to device
        if self.hps.wrap_absorb:
            state = torch.Tensor(batch['obs0_orig']).to(self.device)
            action = torch.Tensor(batch['acs_orig']).to(self.device)
            next_state = torch.Tensor(batch['obs1_orig']).to(self.device)
        else:
            state = torch.Tensor(batch['obs0']).to(self.device)
            action = torch.Tensor(batch['acs']).to(self.device)
            next_state = torch.Tensor(batch['obs1']).to(self.device)
        reward = torch.Tensor(batch['rews']).to(self.device)
        done = torch.Tensor(batch['dones1'].astype('float32')).to(self.device)
        if self.hps.n_step_returns:
            td_len = torch.Tensor(batch['td_len']).to(self.device)
        else:
            td_len = torch.ones_like(done).to(self.device)

        # compute target action
        if self.hps.targ_actor_smoothing:
            n_ = action.clone().detach().data.normal_(0., self.hps.td3_std).to(self.device)
            n_ = n_.clamp(-self.hps.td3_c, self.hps.td3_c)
            next_action = (self.targ_actr.act(next_state) + n_).clamp(-self.max_ac, self.max_ac)
        else:
            next_action = self.targ_actr.act(next_state)

        # compute critic and actor losses

        if self.hps.use_c51:

            # compute qz estimate
            z = self.crit.qz(state, action).unsqueeze(-1)
            z.data.clamp_(0.01, 0.99)

            # compute target qz estimate
            z_prime = self.targ_crit.qz(next_state, next_action)
            z_prime.data.clamp_(0.01, 0.99)

            gamma_mask = ((self.hps.gamma ** td_len) * (1 - done))
            tz = reward + (gamma_mask * self.c51_supp.view(1, self.hps.c51_num_atoms))
            tz = tz.clamp(self.hps.c51_vmin, self.hps.c51_vmax)
            b = (tz - self.hps.c51_vmin) / self.c51_delta
            l = b.floor().long()  # noqa
            u = b.ceil().long()
            targ_z = z_prime.clone().zero_()
            z_prime_l = z_prime * (u + (l == u).float() - b)
            z_prime_u = z_prime * (b - l.float())
            for i in range(targ_z.size(0)):
                targ_z[i].index_add_(0, l[i], z_prime_l[i])
                targ_z[i].index_add_(0, u[i], z_prime_u[i])

            # reshape target to be of shape [batch_size, self.hps.c51_num_atoms, 1]
            targ_z = targ_z.view(-1, self.hps.c51_num_atoms, 1)

            # critic loss
            ce_losses = -(targ_z.detach() * torch.log(z)).sum(dim=1)
            crit_loss = ce_losses.mean()

            # actor loss
            actr_loss = -self.crit.qz(state, self.actr.act(state))  # [batch_size, num_atoms]
            actr_loss = actr_loss.matmul(self.c51_supp).unsqueeze(-1)  # [batch_size, 1]
            actr_loss = actr_loss.mean()

            # update critic
            self.crit_opt.zero_grad()
            crit_loss.backward()
            average_gradients(self.crit, self.device)
            self.crit_opt.step()

        elif self.hps.use_qr:

            # compute qz estimate
            z = self.crit.qz(state, action).unsqueeze(-1)

            # compute target qz estimate
            z_prime = self.targ_crit.qz(next_state, next_action)

            # reshape rewards to be of shape [batch_size x num_tau, 1]
            reward = reward.repeat(self.hps.num_tau, 1)
            # reshape product of gamma and mask to be of shape [batch_size x num_tau, 1]
            gamma_mask = ((self.hps.gamma ** td_len) * (1 - done)).repeat(self.hps.num_tau, 1)
            z_prime = z_prime.view(-1, 1)
            targ_z = reward + (gamma_mask * z_prime)
            # reshape target to be of shape [batch_size, num_tau, 1]
            targ_z = targ_z.view(-1, self.hps.num_tau, 1)

            # critic loss
            # compute the TD error loss
            # note: online version has shape [batch_size, num_tau, 1],
            # while the target version has shape [batch_size, num_tau, 1].
            td_errors = targ_z[:, :, None, :].detach() - z[:, None, :, :]  # broadcasting
            # the resulting shape is [batch_size, num_tau, num_tau, 1]

            # assemble the Huber quantile regression loss
            huber_td_errors = huber_quant_reg_loss(td_errors, self.qr_cum_density)
            # the resulting shape is [batch_size, num_tau_prime, num_tau, 1]

            # sum over current quantile value (tau, N in paper) dimension, and
            # average over target quantile value (tau prime, N' in paper) dimension.
            crit_loss = huber_td_errors.sum(dim=2)
            # resulting shape is [batch_size, num_tau_prime, 1]
            crit_loss = crit_loss.mean(dim=1)
            # resulting shape is [batch_size, 1]
            # average across the minibatch
            crit_loss = crit_loss.mean()

            # actor loss
            actr_loss = -self.crit.qz(state, self.actr.act(state))

            # update critic
            self.crit_opt.zero_grad()
            crit_loss.backward()
            average_gradients(self.crit, self.device)
            self.crit_opt.step()

        else:

            # compute qz estimates
            q = self.denorm_rets(self.crit.qz(state, action))
            twin_q = self.denorm_rets(self.twin.qz(state, action))

            # compute target qz estimate and same for twin
            q_prime = self.targ_crit.qz(next_state, next_action)
            twin_q_prime = self.targ_twin.qz(next_state, next_action)
            q_prime = (0.75 * torch.min(q_prime, twin_q_prime) +
                       0.25 * torch.max(q_prime, twin_q_prime))  # soft minimum from BCQ
            targ_q = (reward +
                      (self.hps.gamma ** td_len) * (1. - done) *
                      self.denorm_rets(q_prime).detach())
            targ_q = self.norm_rets(targ_q)

            if self.hps.ret_norm:
                if self.hps.popart:
                    # apply Pop-Art, https://arxiv.org/pdf/1602.07714.pdf
                    # save the pre-update running stats
                    old_mean = torch.Tensor(self.rms_ret.mean).to(self.device)
                    old_std = torch.Tensor(self.rms_ret.std).to(self.device)
                    # update the running stats
                    self.rms_ret.update(targ_q)
                    # get the post-update running statistics
                    new_mean = torch.Tensor(self.rms_ret.mean).to(self.device)
                    new_std = torch.Tensor(self.rms_ret.std).to(self.device)
                    # preserve the output from before the change of normalization old->new
                    # for both online and target critic(s)
                    outs = [self.crit.out_params, self.targ_crit.out_params]
                    # also use the twin
                    outs.extend([self.twin.out_params, self.targ_twin.out_params])
                    for out in outs:
                        w, b = out
                        w.data.copy_(w.data * old_std / new_std)
                        b.data.copy_(((b.data * old_std) + old_mean - new_mean) / new_std)
                else:
                    # update the running stats
                    self.rms_ret.update(targ_q)

            # critic and twin losses
            crit_loss = ff.smooth_l1_loss(q, targ_q)  # Huber loss
            twin_loss = ff.smooth_l1_loss(twin_q, targ_q)

            # actor loss
            actr_loss = -self.crit.qz(state, self.actr.act(state))

            # update critic
            self.crit_opt.zero_grad()
            crit_loss.backward()
            average_gradients(self.crit, self.device)
            self.crit_opt.step()
            # update twin
            self.twin_opt.zero_grad()
            twin_loss.backward()
            average_gradients(self.twin, self.device)
            self.twin_opt.step()

            metrics['twin_loss'].append(twin_loss)

        # log common metrics
        metrics['crit_loss'].append(crit_loss)
        metrics['actr_loss'].append(actr_loss)

        # update actor
        self.actr_opt.zero_grad()
        actr_loss.backward()
        average_gradients(self.actr, self.device)
        if self.hps.clip_norm > 0:
            cg.clip_grad_norm_(self.actr.parameters(), self.hps.clip_norm)

        _lr = self.hps.actor_lr  # initialize for first iteration

        if update_actor:

            self.actr_opt.step()

            _lr = self.actr_sched.step(steps_so_far=iters_so_far * self.hps.rollout_len)
            logger.info(f"lr is {_lr} after {iters_so_far} timesteps")

        # update target nets
        self.update_target_net(iters_so_far)

        metrics = {k: torch.stack(v).mean().cpu().numpy() for k, v in metrics.items()}
        lrnows = {'actr': _lr}

        return metrics, lrnows

    def update_discriminator(self, batch):

        metrics = defaultdict(list)

        # create DataLoader object to iterate over transitions in rollouts
        d_keys = ['obs0']
        if self.hps.state_only:
            if self.hps.n_step_returns:
                d_keys.append('obs1_td1')
            else:
                d_keys.append('obs1')
        else:
            d_keys.append('acs')

        d_dataset = Dataset({k: batch[k] for k in d_keys})
        d_dataloader = DataLoader(
            d_dataset,
            self.e_batch_size,
            shuffle=True,
            drop_last=True,
        )

        for e_batch in self.e_dataloader:

            # get a minibatch of policy data
            d_batch = next(iter(d_dataloader))

            # transfer to device
            p_input_a = d_batch['obs0'].to(self.device)
            e_input_a = e_batch['obs0'].to(self.device)
            if self.hps.state_only:
                if self.hps.n_step_returns:
                    p_input_b = d_batch['obs1_td1'].to(self.device)
                else:
                    p_input_b = d_batch['obs1'].to(self.device)
                e_input_b = e_batch['obs1'].to(self.device)
            else:
                p_input_b = d_batch['acs'].to(self.device)
                e_input_b = e_batch['acs'].to(self.device)

            # compute scores
            p_scores = self.disc(p_input_a, p_input_b)
            e_scores = self.disc(e_input_a, e_input_b)

            # entropy loss
            scores = torch.cat([p_scores, e_scores], dim=0)
            entropy = ff.binary_cross_entropy_with_logits(
                input=scores,
                target=torch.sigmoid(scores),
            )
            entropy_loss = -self.hps.ent_reg_scale * entropy

            # create labels
            fake_labels = 0. * torch.ones_like(p_scores).to(self.device)
            real_labels = 1. * torch.ones_like(e_scores).to(self.device)

            # apply label smoothing to real labels
            real_labels.uniform_(0.7, 1.2)

            # binary classification
            p_loss = ff.binary_cross_entropy_with_logits(
                input=p_scores,
                target=fake_labels,
            )
            e_loss = ff.binary_cross_entropy_with_logits(
                input=e_scores,
                target=real_labels,
            )
            p_e_loss = p_loss + e_loss

            # sum losses
            disc_loss = p_e_loss + entropy_loss

            metrics['entropy_loss'].append(entropy_loss)
            metrics['p_e_loss'].append(p_e_loss)
            metrics['disc_loss'].append(disc_loss)

            if self.hps.grad_pen:
                # add gradient penalty to loss
                grad_pen = self.grad_pen(p_input_a, p_input_b, e_input_a, e_input_b)
                metrics['grad_pen'].append(grad_pen)
                grad_pen *= self.hps.grad_pen_scale
                disc_loss += grad_pen

            # update parameters
            self.disc_opt.zero_grad()
            disc_loss.backward()
            average_gradients(self.disc, self.device)
            self.disc_opt.step()

        metrics = {k: torch.stack(v).mean().cpu().numpy() for k, v in metrics.items()}
        return metrics

    def grad_pen(self, p_input_a, p_input_b, e_input_a, e_input_b):

        # assemble interpolated inputs
        eps_a = torch.rand(p_input_a.size(0), 1).to(self.device)
        eps_b = torch.rand(p_input_b.size(0), 1).to(self.device)
        input_a_i = eps_a * p_input_a + ((1. - eps_a) * e_input_a)
        input_b_i = eps_b * p_input_b + ((1. - eps_b) * e_input_b)

        input_a_i.requires_grad = True
        input_b_i.requires_grad = True

        score = self.disc(input_a_i, input_b_i)

        # get the gradient of this operation w.r.t. its inputs
        grads = autograd.grad(
            outputs=score,
            inputs=[input_a_i, input_b_i],
            only_inputs=True,
            grad_outputs=[torch.ones_like(score)],
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )
        assert len(list(grads)) == 2, "length must be exactly 2"

        # return the gradient penalty (try to induce 1-Lipschitzness)
        grads = torch.cat(list(grads), dim=-1)
        grads_norm = grads.norm(2, dim=-1)

        if self.hps.one_sided_pen:
            # penalize the gradient for having a norm GREATER than k
            grad_pen = torch.max(
                torch.zeros_like(grads_norm),
                grads_norm - self.hps.grad_pen_targ
            ).pow(2)
        else:
            # penalize the gradient for having a norm LOWER OR GREATER than k
            grad_pen = (grads_norm - self.hps.grad_pen_targ).pow(2)

        # average over batch
        grad_pen = grad_pen.mean()

        return grad_pen

    def get_syn_rew(self, state, action, next_state):

        # define the discriminator inputs
        input_a = state
        if self.hps.state_only:
            input_b = next_state
        else:
            input_b = action
        assert sum([isinstance(x, torch.Tensor) for x in [input_a, input_b]]) in [0, 2]
        if not isinstance(input_a, torch.Tensor):  # then the other is not neither
            input_a = torch.Tensor(input_a)
            input_b = torch.Tensor(input_b)

        # transfer to device in use
        input_a = input_a.to(self.device)
        input_b = input_b.to(self.device)

        # compure score
        score = self.disc(input_a, input_b).detach().view(-1, 1)
        # counterpart of GAN's minimax (also called "saturating") loss
        # numerics: 0 for non-expert-like states, goes to +inf for expert-like states
        # compatible with envs with traj cutoffs for bad (non-expert-like) behavior
        # e.g. walking simulations that get cut off when the robot falls over
        minimax_reward = -torch.log(1. - torch.sigmoid(score) + 1e-8)
        if self.hps.minimax_only:
            reward = minimax_reward
        else:
            # counterpart of GAN's non-saturating loss
            # recommended in the original GAN paper and later in (Fedus et al. 2017)
            # numerics: 0 for expert-like states, goes to -inf for non-expert-like states
            # compatible with envs with traj cutoffs for good (expert-like) behavior
            # e.g. mountain car, which gets cut off when the car reaches the destination
            non_satur_reward = ff.logsigmoid(score)
            # return the sum the two previous reward functions (as in AIRL, Fu et al. 2018)
            # numerics: might be better might be way worse
            reward = non_satur_reward + minimax_reward

        return reward


    def update_target_net(self, iters_so_far):

        if sum([self.hps.use_c51, self.hps.use_qr]) == 0:
            # If non-distributional, targets slowly track their non-target counterparts
            for param, targ_param in zip(self.actr.parameters(), self.targ_actr.parameters()):
                targ_param.data.copy_(self.hps.polyak * param.data +
                                      (1. - self.hps.polyak) * targ_param.data)
            for param, targ_param in zip(self.crit.parameters(), self.targ_crit.parameters()):
                targ_param.data.copy_(self.hps.polyak * param.data +
                                      (1. - self.hps.polyak) * targ_param.data)
            if self.hps.clipped_double:
                for param, targ_param in zip(self.twin.parameters(), self.targ_twin.parameters()):
                    targ_param.data.copy_(self.hps.polyak * param.data +
                                          (1. - self.hps.polyak) * targ_param.data)
        else:
            # If distributional, periodically set target weights with online's
            if iters_so_far % self.hps.targ_up_freq == 0:
                self.targ_actr.load_state_dict(self.actr.state_dict())
                self.targ_crit.load_state_dict(self.crit.state_dict())
                if self.hps.clipped_double:
                    self.targ_twin.load_state_dict(self.twin.state_dict())

    def adapt_param_noise(self):
        # adapt the parameter noise standard deviation

        assert self.param_noise is not None, "you should not be here"

        # Perturb separate copy of the policy to adjust the scale for the next 'real' perturbation
        batch = self.replay_buffer.sample(self.hps.batch_size, patcher=None)
        state = torch.Tensor(batch['obs0']).to(self.device)
        # Update the perturbable params
        for p in self.actr.perturbable_params:
            param = (self.actr.state_dict()[p]).clone()
            param_ = param.clone()
            noise = param_.data.normal_(0, self.param_noise.cur_std)
            self.apnp_actr.state_dict()[p].data.copy_((param + noise).data)
        # Update the non-perturbable params
        for p in self.actr.non_perturbable_params:
            param = self.actr.state_dict()[p].clone()
            self.apnp_actr.state_dict()[p].data.copy_(param.data)

        # Compute distance between actor and adaptive-parameter-noise-perturbed actor predictions
        if self.hps.wrap_absorb:
            state = self.remove_absorbing(state)[0][:, 0:-1]
        self.pn_dist = torch.sqrt(ff.mse_loss(self.actr.act(state), self.apnp_actr.act(state)))
        self.pn_dist = self.pn_dist.cpu().data.numpy()

        # Adapt the parameter noise
        self.param_noise.adapt_std(self.pn_dist)

    def reset_noise(self):
        """Reset noise processes at episode termination"""

        # Reset action noise
        if self.ac_noise is not None:
            self.ac_noise.reset()

        # Reset parameter-noise-perturbed actor vars by redefining the pnp actor
        # w.r.t. the actor (by applying additive gaussian noise with current std)
        if self.param_noise is not None:
            # Update the perturbable params
            for p in self.actr.perturbable_params:
                param = (self.actr.state_dict()[p]).clone()
                param_ = param.clone()
                noise = param_.data.normal_(0, self.param_noise.cur_std)
                self.pnp_actr.state_dict()[p].data.copy_((param + noise).data)
            # Update the non-perturbable params
            for p in self.actr.non_perturbable_params:
                param = self.actr.state_dict()[p].clone()
                self.pnp_actr.state_dict()[p].data.copy_(param.data)

    def save(self, path, iters_so_far):
        torch.save(self.rms_obs.state_dict(), osp.join(path, f"rms_obs_{iters_so_far}.pth"))
        torch.save(self.actr.state_dict(), osp.join(path, f"actr_{iters_so_far}.pth"))
        torch.save(self.crit.state_dict(), osp.join(path, f"crit_{iters_so_far}.pth"))
        if self.hps.clipped_double:
            torch.save(self.twin.state_dict(), osp.join(path, f"twin_{iters_so_far}.pth"))
        if not self.eval_mode:
            torch.save(self.disc.state_dict(), osp.join(path, f"disc_{iters_so_far}.pth"))

    def load(self, path, iters_so_far):
        self.rms_obs.load_state_dict(torch.load(osp.join(path, f"rms_obs_{iters_so_far}.pth")))
        self.actr.load_state_dict(torch.load(osp.join(path, f"actr_{iters_so_far}.pth")))
        self.crit.load_state_dict(torch.load(osp.join(path, f"crit_{iters_so_far}.pth")))
        if self.hps.clipped_double:
            self.twin.load_state_dict(torch.load(osp.join(path, f"twin_{iters_so_far}.pth")))
        if not self.eval_mode:
            self.disc.load_state_dict(torch.load(osp.join(path, f"disc_{iters_so_far}.pth")))
