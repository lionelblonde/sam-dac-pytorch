import time
from copy import deepcopy
from pathlib import Path
from functools import partial
from typing import Union, List, Any, Callable

from beartype import beartype
from einops import rearrange
from omegaconf import OmegaConf, DictConfig
import wandb
from wandb.errors import CommError
import numpy as np

import gymnasium as gym
from gymnasium.core import Env
from gymnasium.experimental.vector.async_vector_env import AsyncVectorEnv
from gymnasium.experimental.vector.sync_vector_env import SyncVectorEnv

from helpers import logger
from helpers.misc_util import timed, log_iter_info
from helpers.opencv_util import record_video
from agents.spp_agent import SPPAgent


DEBUG = False


@beartype
def segment(env: Union[Env, AsyncVectorEnv, SyncVectorEnv],
            agent: SPPAgent,
            seed: int,
            segment_len: int,
            *,
            wrap_absorb: bool):

    assert isinstance(env.action_space, gym.spaces.Box)  # to ensure `high` and `low` exist
    ac_low, ac_high = env.action_space.low, env.action_space.high

    t = 0

    ob, _ = env.reset(seed=seed)  # seed is a keyword argument, not positional

    while True:

        # predict action
        ac = agent.predict(ob, apply_noise=True)
        # nan-proof and clip
        ac = np.nan_to_num(ac)
        ac = np.clip(ac, ac_low, ac_high)

        if t > 0 and t % segment_len == 0:
            yield

        # interact with env
        new_ob, _, terminated, truncated, _ = env.step(ac)  # reward and info ignored
        if not isinstance(env, (AsyncVectorEnv, SyncVectorEnv)):
            assert isinstance(env, Env)
            done = np.array(terminated or truncated)
            if truncated:
                logger.warn("termination caused by something like time limit or out of bounds?")
        else:
            done = np.logical_or(terminated, truncated)  # might not be used but diagnostics
        # read about what truncation means at the link below:
        # https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/#truncation

        tr_or_vtr = [
            ob, ac, new_ob, rearrange(done, "b -> b 1"), rearrange(terminated, "b -> b 1")]
        if isinstance(env, (AsyncVectorEnv, SyncVectorEnv)):
            pp_func = partial(postproc_vtr, env.num_envs)
        else:
            assert isinstance(env, Env)
            pp_func = postproc_tr
        pp_func(tr_or_vtr, agent, wrap_absorb=wrap_absorb)

        # set current state with the next
        ob = deepcopy(new_ob)

        if not isinstance(env, (AsyncVectorEnv, SyncVectorEnv)):
            assert isinstance(env, Env)
            if done:
                ob, _ = env.reset(seed=seed)

        t += 1


@beartype
def postproc_tr(tr: List[Any],
                agent: SPPAgent,
                *,
                wrap_absorb: bool):

    ob, ac, new_ob, done, terminated = tr

    if wrap_absorb:

        _ob = np.append(ob, 0)
        _ac = np.append(ac, 0)

        # previously this was the cond: `done and env._elapsed_steps != env._max_episode_steps`
        if terminated:
            # wrap with an absorbing state
            _new_ob = np.append(np.zeros(agent.ob_shape[-1]), 1)
            _rew = agent.get_syn_rew(
                rearrange(_ob, "d -> 1 d"),
                rearrange(_ac, "d -> 1 d"),
                rearrange(_new_ob, "d -> 1 d")).numpy(force=True)
            transition = {
                "obs0": _ob,
                "acs": _ac,
                "obs1": _new_ob,
                "rews": _rew,
                "dones1": done,
                "obs0_orig": ob,
                "acs_orig": ac,
                "obs1_orig": new_ob,
            }
            agent.store_transition(transition)
            # add absorbing transition
            _ob_a = np.append(np.zeros(agent.ob_shape[-1]), 1)
            _ac_a = np.append(np.zeros(agent.ac_shape[-1]), 1)
            _new_ob_a = np.append(np.zeros(agent.ob_shape[-1]), 1)
            _rew_a = agent.get_syn_rew(
                rearrange(_ob_a, "d -> 1 d"),
                rearrange(_ac_a, "d -> 1 d"),
                rearrange(_new_ob_a, "d -> 1 d")).numpy(force=True)
            transition_a = {
                "obs0": _ob_a,
                "acs": _ac_a,
                "obs1": _new_ob_a,
                "rews": _rew_a,
                "dones1": done,
                "obs0_orig": ob,  # from previous transition, with reward eval on absorbing
                "acs_orig": ac,  # from previous transition, with reward eval on absorbing
                "obs1_orig": new_ob,  # from previous transition, with reward eval on absorbing
            }
            agent.store_transition(transition_a)
        else:
            _new_ob = np.append(new_ob, 0)
            _rew = agent.get_syn_rew(
                rearrange(_ob, "d -> 1 d"),
                rearrange(_ac, "d -> 1 d"),
                rearrange(_new_ob, "d -> 1 d")).numpy(force=True)
            transition = {
                "obs0": _ob,
                "acs": _ac,
                "obs1": _new_ob,
                "rews": _rew,
                "dones1": done,
                "obs0_orig": ob,
                "acs_orig": ac,
                "obs1_orig": new_ob,
            }
            agent.store_transition(transition)
    else:
        rew = agent.get_syn_rew(
                rearrange(ob, "d -> 1 d"),
                rearrange(ac, "d -> 1 d"),
                rearrange(new_ob, "d -> 1 d")).numpy(force=True)
        transition = {
            "obs0": ob,
            "acs": ac,
            "obs1": new_ob,
            "rews": rew,
            "dones1": done,
        }
        agent.store_transition(transition)


@beartype
def postproc_vtr(num_envs: int,
                 vtr: List[Any],
                 agent: SPPAgent,
                 *,
                 wrap_absorb: bool):
    for i in range(num_envs):
        tr = [e[i] for e in vtr]
        postproc_tr(tr, agent, wrap_absorb=wrap_absorb)


@beartype
def episode(env: Env,
            agent: SPPAgent,
            seed: int):
    # generator that spits out a trajectory collected during a single episode
    # `append` operation is also significantly faster on lists than numpy arrays,
    # they will be converted to numpy arrays once complete right before the yield

    assert isinstance(env.action_space, gym.spaces.Box)  # to ensure `high` and `low` exist
    ac_low, ac_high = env.action_space.low, env.action_space.high

    rng = np.random.default_rng(seed)  # aligned on seed, so always reproducible
    logger.warn("remember: in episode generator, we generate a seed randomly")
    logger.warn("i.e. not using 'ob, _ = env.reset(seed=seed)' with same seed")
    # note that despite sampling a new seed, it is using a seeded rng: reproducible
    ob, _ = env.reset(seed=seed + rng.integers(100000, size=1).item())

    cur_ep_len = 0
    cur_ep_env_ret = 0
    obs = []
    acs = []
    env_rews = []

    while True:

        # predict action
        ac = agent.predict(ob, apply_noise=False)
        # nan-proof and clip
        ac = np.nan_to_num(ac)
        ac = np.clip(ac, ac_low, ac_high)

        obs.append(ob)
        acs.append(ac)
        new_ob, env_rew, terminated, truncated, _ = env.step(ac)
        done = terminated or truncated

        env_rews.append(env_rew)
        cur_ep_len += 1
        assert isinstance(env_rew, float)  # quiets the type-checker
        cur_ep_env_ret += env_rew
        ob = deepcopy(new_ob)

        if done:
            obs = np.array(obs)
            acs = np.array(acs)
            env_rews = np.array(env_rews)
            out = {
                "obs": obs,
                "acs": acs,
                "env_rews": env_rews,
                "ep_len": cur_ep_len,
                "ep_env_ret": cur_ep_env_ret,
            }
            yield out

            cur_ep_len = 0
            cur_ep_env_ret = 0
            obs = []
            acs = []
            env_rews = []
            logger.warn("remember: in episode generator, we generate a seed randomly")
            logger.warn("i.e. not using 'ob, _ = env.reset(seed=seed)' with same seed")
            ob, _ = env.reset(seed=seed + rng.integers(100000, size=1).item())


@beartype
def evaluate(cfg: DictConfig,
             env: Env,
             agent_wrapper: Callable[[], SPPAgent],
             name: str):

    assert isinstance(cfg, DictConfig)

    vid_dir = Path(cfg.video_dir) / name
    if cfg.record:
        vid_dir.mkdir(parents=True, exist_ok=True)

    # create an agent
    agent = agent_wrapper()

    # create episode generator
    ep_gen = episode(env, agent, cfg.seed)

    # load the model
    model_path = cfg.model_path
    agent.load_from_path(model_path)
    logger.info(f"model loaded from path:\n {model_path}")

    # collect trajectories

    num_trajs = cfg.num_trajs
    len_buff, env_ret_buff = [], []

    for i in range(num_trajs):

        logger.info(f"evaluating [{i + 1}/{num_trajs}]")
        traj = next(ep_gen)
        ep_len, ep_env_ret = traj["ep_len"], traj["ep_env_ret"]

        # aggregate to the history data structures
        len_buff.append(ep_len)
        env_ret_buff.append(ep_env_ret)

        if cfg.record:
            # record a video of the episode
            frame_collection = env.render()  # ref: https://younis.dev/blog/render-api/
            record_video(vid_dir, str(i), np.array(frame_collection))

    eval_metrics = {"ep_len": len_buff, "ep_env_ret": env_ret_buff}

    # log stats in csv
    logger.record_tabular("timestep", agent.timesteps_so_far)
    for k, v in eval_metrics.items():
        logger.record_tabular(f"{k}-mean", np.mean(v))
    logger.info("dumping stats in .csv file")
    logger.dump_tabular()


@beartype
def learn(cfg: DictConfig,
          env: Union[Env, AsyncVectorEnv, SyncVectorEnv],
          eval_env: Env,
          agent_wrapper: Callable[[], SPPAgent],
          name: str):

    assert isinstance(cfg, DictConfig)

    # create an agent
    agent = agent_wrapper()

    # start clock
    tstart = time.time()

    # set up model save directory
    ckpt_dir = Path(cfg.checkpoint_dir) / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    vid_dir = Path(cfg.video_dir) / name
    if cfg.record:
        vid_dir.mkdir(parents=True, exist_ok=True)

    # save the model as a dry run, to avoid bad surprises at the end
    agent.save_to_path(ckpt_dir, xtra="dryrun")
    logger.info(f"dry run. saving model @:\n{ckpt_dir}")

    # group by everything except the seed, which is last, hence index -1
    # it groups by uuid + gitSHA + env_id + num_demos
    group = ".".join(name.split(".")[:-1])
    # set up wandb
    while True:
        try:
            config = OmegaConf.to_object(cfg)
            assert isinstance(config, dict)
            wandb.init(
                project=cfg.wandb_project,
                name=name,
                id=name,
                group=group,
                config=config,
                dir=cfg.root,
            )
            break
        except CommError:
            pause = 10
            logger.info(f"wandb co error. Retrying in {pause} secs.")
            time.sleep(pause)
    logger.info("wandb co established!")

    for glob in ["train", "explore", "eval"]:  # wandb categories
        # define a custom x-axis
        wandb.define_metric(f"{glob}/step")
        wandb.define_metric(f"{glob}/*", step_metric=f"{glob}/step")

    # create segment generator for training the agent
    roll_gen = segment(env, agent, cfg.seed, cfg.segment_len, wrap_absorb=cfg.wrap_absorb)
    # create episode generator for evaluating the agent
    eval_seed = cfg.seed + 123456  # arbitrary choice
    ep_gen = episode(eval_env, agent, eval_seed)

    i = 0

    while agent.timesteps_so_far <= cfg.num_timesteps:

        if i % 100 == 0 or DEBUG:
            log_iter_info(i, cfg.num_timesteps // cfg.segment_len, tstart)

        with timed("interacting"):
            next(roll_gen)  # no need to get the returned segment, stored in buffer
            agent.timesteps_so_far += cfg.segment_len

        with timed("training"):
            for _ in range(cfg.training_steps_per_iter):

                if agent.param_noise is not None:
                    if agent.actr_updates_so_far % cfg.pn_adapt_frequency == 0:
                        # adapt parameter noise
                        agent.adapt_param_noise()
                        agent.send_to_dash(
                            {"pn_dist": agent.pn_dist.numpy(force=True),
                             "pn_cur_std": agent.param_noise.cur_std.numpy(force=True)},
                            step_metric=agent.actr_updates_so_far,
                            glob="explore",
                        )  # `pn_dist`: action-space dist between perturbed and non-perturbed
                        # `pn_cur_std`: store the new std resulting from the adaption

                for _ in range(agent.hps["g_steps"]):
                    # sample a batch of transitions from the replay buffer
                    batch = agent.sample_batch()
                    # determine if updating the actr
                    update_actr = not bool(
                        agent.crit_updates_so_far % cfg.actor_update_delay)
                    # update the actor and critic
                    agent.update_actr_crit(
                        batch=batch,
                        update_actr=update_actr,
                    )  # counters for actr and crit updates are incremented internally!

                for _ in range(agent.hps["d_steps"]):
                    # sample a batch of transitions from the replay buffer
                    batch = agent.sample_batch()
                    # update the discriminator
                    agent.update_disc(batch)  # update counter incremented internally too

        i += 1

        if i % cfg.eval_every == 0:

            with timed("evaluating"):

                len_buff, env_ret_buff = [], []

                for j in range(cfg.eval_steps_per_iter):

                    # sample an episode with non-perturbed actor
                    ep = next(ep_gen)
                    # none of it is collected in the replay buffer

                    len_buff.append(ep["ep_len"])
                    env_ret_buff.append(ep["ep_env_ret"])

                    if cfg.record:
                        # record a video of the episode
                        # ref: https://younis.dev/blog/render-api/
                        frame_collection = eval_env.render()
                        record_video(vid_dir, f"iter{i}-ep{j}", np.array(frame_collection))

                eval_metrics: dict[str, np.ndarray] = {
                    "ep_len": np.array(len_buff), "ep_env_ret": np.array(env_ret_buff)}

                # log stats in csv
                logger.record_tabular("timestep", agent.timesteps_so_far)
                for k, v in eval_metrics.items():
                    logger.record_tabular(f"{k}-mean", v.mean())
                logger.info("dumping stats in .csv file")
                logger.dump_tabular()

                # log stats in dashboard
                agent.send_to_dash(
                    {f"{k}-mean": v.mean() for k, v in eval_metrics.items()},
                    step_metric=agent.timesteps_so_far,
                    glob="eval",
                )

    # save once we are done
    agent.save_to_path(ckpt_dir, xtra="done")
    logger.info(f"we are done. saving model @:\n{ckpt_dir}\nbye.")
