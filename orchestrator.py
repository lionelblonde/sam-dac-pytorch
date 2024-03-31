import time
from copy import deepcopy
from pathlib import Path

import wandb
from wandb.errors import CommError
import numpy as np

from helpers import logger
from helpers.console_util import timed_cm_wrapper, log_iter_info
from helpers.opencv_util import record_video


DEBUG = False


def rollout(env, agent, rollout_len):

    t = 0
    # reset agent noise process
    agent.reset_noise()
    # reset agent env
    ob = np.array(env.reset())

    while True:

        # predict action
        ac = agent.predict(ob, apply_noise=True)
        # nan-proof and clip
        ac = np.nan_to_num(ac)
        ac = np.clip(ac, env.action_space.low, env.action_space.high)

        if t > 0 and t % rollout_len == 0:
            yield

        # interact with env
        new_ob, _, done, _ = env.step(ac)

        if agent.hps.wrap_absorb:
            _ob = np.append(ob, 0)
            _ac = np.append(ac, 0)
            if done and env._elapsed_steps != env._max_episode_steps:
                # wrap with an absorbing state
                _new_ob = np.append(np.zeros(agent.ob_shape), 1)
                _rew = agent.get_syn_rew(_ob[None], _ac[None], _new_ob[None])
                _rew = _rew.cpu().numpy().flatten().item()
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
                _ob_a = np.append(np.zeros(agent.ob_shape), 1)
                _ac_a = np.append(np.zeros(agent.ac_shape), 1)
                _new_ob_a = np.append(np.zeros(agent.ob_shape), 1)
                _rew_a = agent.get_syn_rew(_ob_a[None], _ac_a[None], _new_ob_a[None])
                _rew_a = _rew_a.cpu().numpy().flatten().item()
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
                _rew = agent.get_syn_rew(_ob[None], _ac[None], _new_ob[None])
                _rew = _rew.cpu().numpy().flatten().item()
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
            rew = agent.get_syn_rew(ob[None], ac[None], new_ob[None])
            rew = rew.cpu().numpy().flatten().item()
            transition = {
                "obs0": ob,
                "acs": ac,
                "obs1": new_ob,
                "rews": rew,
                "dones1": done,
            }
            agent.store_transition(transition)

        # set current state with the next
        ob = np.array(deepcopy(new_ob))

        if done:
            # reset agent noise process
            agent.reset_noise()
            # reset the env
            ob = np.array(env.reset())

        t += 1


def episode(env, agent, render):
    # generator that spits out a trajectory collected during a single episode
    # `append` operation is also significantly faster on lists than numpy arrays,
    # they will be converted to numpy arrays once complete right before the yield
    render_kwargs = {"mode": "rgb_array"}
    ob = np.array(env.reset())
    ob_rgb = env.render(**render_kwargs)

    cur_ep_len = 0
    cur_ep_env_ret = 0
    obs = []
    obs_rgb = []
    acs = []
    env_rews = []

    while True:

        # predict action
        ac = agent.predict(ob, apply_noise=False)
        # nan-proof and clip
        ac = np.nan_to_num(ac)
        ac = np.clip(ac, env.action_space.low, env.action_space.high)

        obs.append(ob)
        obs_rgb.append(ob_rgb)
        acs.append(ac)
        new_ob, env_rew, done, _ = env.step(ac)

        if render:
            env.render()

        ob_rgb = env.render(**render_kwargs)

        env_rews.append(env_rew)
        cur_ep_len += 1
        cur_ep_env_ret += env_rew
        ob = np.array(deepcopy(new_ob))

        if done:
            obs = np.array(obs)
            obs_rgb = np.array(obs_rgb)
            acs = np.array(acs)
            env_rews = np.array(env_rews)
            out = {
                "obs": obs,
                "obs_rgb": obs_rgb,
                "acs": acs,
                "env_rews": env_rews,
                "ep_len": cur_ep_len,
                "ep_env_ret": cur_ep_env_ret,
            }
            yield out

            cur_ep_len = 0
            cur_ep_env_ret = 0
            obs = []
            obs_rgb = []
            acs = []
            env_rews = []
            ob = np.array(env.reset())
            ob_rgb = env.render(**render_kwargs)


def evaluate(args, env, agent_wrapper, experiment_name):

    vid_dir = Path(args.video_dir) / experiment_name
    if args.record:
        vid_dir.mkdir(vid_dir, exist_ok=True)

    # create an agent
    agent = agent_wrapper()

    # create episode generator
    ep_gen = episode(env, agent, args.render)

    # load the model
    agent.load_from_path(args.model_path, args.iter_num)
    logger.info(f"model loaded from path:\n {args.model_path}")

    # collect trajectories

    len_buff, env_ret_buff = [], []

    for i in range(args.num_trajs):

        logger.info(f"evaluating [{i + 1}/{args.num_trajs}]")
        traj = ep_gen.__next__()
        ep_len, ep_env_ret = traj["ep_len"], traj["ep_env_ret"]

        # aggregate to the history data structures
        len_buff.append(ep_len)
        env_ret_buff.append(ep_env_ret)

        if args.record:
            # record a video of the episode
            record_video(vid_dir, i, traj["obs_rgb"])

    eval_metrics = {"ep_len": len_buff, "ep_env_ret": env_ret_buff}

    # log stats in csv
    logger.record_tabular("timestep", agent.timesteps_so_far)
    for k, v in eval_metrics.items():
        logger.record_tabular(f"{k}-mean", np.mean(v))
    logger.info("dumping stats in .csv file")
    logger.dump_tabular()


def learn(args, rank, env, eval_env, agent_wrapper, experiment_name):

    # create an agent
    agent = agent_wrapper()

    # create context manager that records the time taken by encapsulated ops
    timed = timed_cm_wrapper(logger, use=DEBUG)

    # start clock
    tstart = time.time()

    # set up model save directory
    ckpt_dir = Path(args.checkpoint_dir) / experiment_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:

        # save the model as a dry run, to avoid bad surprises at the end
        agent.save_to_path(ckpt_dir, xtra="dryrun")
        logger.info(f"dry run. saving model @:\n{ckpt_dir}")

        # group by everything except the seed, which is last, hence index -1
        # it groups by uuid + gitSHA + env_id + num_demos
        group = ".".join(experiment_name.split(".")[:-1])

        # set up wandb
        while True:
            try:
                wandb.init(
                    project=args.wandb_project,
                    name=experiment_name,
                    id=experiment_name,
                    group=group,
                    config=args.__dict__,
                    dir=args.root,
                )
                break
            except CommError:
                pause = 10
                logger.info("wandb co error. Retrying in {} secs.".format(pause))
                time.sleep(pause)
        logger.info("wandb co established!")

    for glob in ["train", "explore", "eval"]:  # wandb categories
        # define a custom x-axis
        wandb.define_metric(f"{glob}/step")
        wandb.define_metric(f"{glob}/*", step_metric=f"{glob}/step")

    # create rollout generator for training the agent
    roll_gen = rollout(env, agent, args.rollout_len)
    # create episode generator for evaluating the agent
    ep_gen = episode(eval_env, agent, args.render)
    # the eval_env is None for all nonzero ranked worker,
    # but only this worker will use its ep_gen (legibility)

    i = 0

    while agent.timesteps_so_far <= args.num_timesteps:

        if i % 100 == 0 or DEBUG:
            log_iter_info(i, args.num_timesteps // args.rollout_len, tstart)

        with timed("interacting"):
            next(roll_gen)  # no need to get the returned rollout, stored in buffer
            agent.timesteps_so_far += args.rollout_len

        with timed("training"):
            for _ in range(args.training_steps_per_iter):

                if agent.param_noise is not None:
                    if agent.actr_updates_so_far % args.pn_adapt_frequency == 0:
                        # adapt parameter noise
                        agent.adapt_param_noise()
                        if rank == 0:
                            agent.send_to_dash(
                                {"pn_dist": agent.pn_dist, "pn_cur_std": agent.pn_cur_std},
                                step_metric=agent.actr_update_so_far,
                                glob="explore",
                            )  # `pn_dist`: action-space dist between perturbed and non-perturbed
                            # `pn_cur_std`: store the new std resulting from the adaption

                for _ in range(agent.hps.g_steps):
                    # sample a batch of transitions from the replay buffer
                    batch = agent.sample_batch()
                    # determine if updating the actr
                    update_actr = not bool(agent.crit_updates_so_far % args.actor_update_delay)
                    # update the actor and critic
                    agent.update_actr_crit(
                        batch=batch,
                        update_actr=update_actr,
                    )  # counters for actr and crit updates are incremented internally!

                for _ in range(agent.hps.d_steps):
                    # sample a batch of transitions from the replay buffer
                    batch = agent.sample_batch()
                    # update the discriminator
                    agent.update_disc(batch)  # update counter incremented internally too

            agent.iters_so_far += 1

        i += 1

        if rank == 0 and agent.actr_updates_so_far % args.eval_frequency == 0:

            with timed("evaluating"):

                len_buff, env_ret_buff = [], []

                for _ in range(args.eval_steps_per_iter):

                    # sample an episode with non-perturbed actor
                    ep = next(ep_gen)
                    # none of it is collected in the replay buffer

                    len_buff.append(ep["ep_len"])
                    env_ret_buff.append(ep["ep_env_ret"])

                eval_metrics = {"ep_len": len_buff, "ep_env_ret": env_ret_buff}

                # log stats in csv
                logger.record_tabular("timestep", agent.timesteps_so_far)
                for k, v in eval_metrics.items():
                    logger.record_tabular(f"{k}-mean", np.mean(v))
                logger.info("dumping stats in .csv file")
                logger.dump_tabular()

                # log stats in dashboard
                agent.send_to_dash(eval_metrics, step_metric=agent.timesteps_so_far, glob="eval")

    if rank == 0:
        # save once we are done
        agent.save_to_path(ckpt_dir, xtra="done")
        logger.info(f"we are done. saving model @:\n{ckpt_dir}\nbye.")
