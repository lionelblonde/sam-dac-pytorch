import os
from pathlib import Path
from argparse import Namespace

import numpy as np
import torch

import orchestrator
from helpers import logger
from helpers.argparsers import argparser
from helpers.experiment import ExperimentInitializer
from helpers.env_makers import make_env
from helpers.dataset import DemoDataset
from agents.memory import ReplayBuffer
from agents.spp_agent import SPPAgent


def train(args: Namespace):

    # mlsys
    torch.set_num_threads(1)  # TODO(lionel): keep an eye on this

    # set printing options
    np.set_printoptions(precision=3)

    # init experiment
    experiment = ExperimentInitializer(args)
    experiment.configure_logging()
    experiment_name = experiment.get_name()

    # device
    assert not args.fp16 or args.cuda, "fp16 => cuda"
    if args.cuda:
        # use cuda
        assert torch.cuda.is_available()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda:0")
    else:
        if args.mps:  # TODO(lionel): add this as hp
            assert torch.mps
            # use Apple"s Metal Performance Shaders (MPS)
            device = torch.device("mps:0")
        else:
            # default case: just use plain old cpu, no cuda or m-chip gpu
            device = torch.device("cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # kill any possibility of usage
    logger.info(f"device in use: {device}")

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np_rng = np.random.default_rng(args.seed)

    # env
    env, net_shapes, erb_shapes, max_ac, max_episode_steps = make_env(
        args.env_id, args.wrap_absorb, record=False, render=args.render)

    # create an agent wrapper

    expert_dataset = DemoDataset(
        np_rng=np_rng,
        expert_path=args.expert_path,
        num_demos=args.num_demos,
        max_ep_steps=max_episode_steps,
        wrap_absorb=args.wrap_absorb,
    )
    replay_buffer = ReplayBuffer(
        np_rng=np_rng,
        capacity=args.mem_size,
        erb_shapes=erb_shapes,
    )
    logger.info(f"{replay_buffer} configured")

    def agent_wrapper():
        return SPPAgent(
            net_shapes=net_shapes,
            max_ac=max_ac,
            device=device,
            hps=args,
            expert_dataset=expert_dataset,
            replay_buffer=replay_buffer,
        )

    # create an evaluation environment not to mess up with training rollouts
    eval_env, _, _, _, _ = make_env(
        args.env_id, args.wrap_absorb, record=args.record, render=args.render)

    # train
    orchestrator.learn(
        args=args,
        env=env,
        eval_env=eval_env,
        agent_wrapper=agent_wrapper,
        experiment_name=experiment_name,
    )

    # cleanup
    env.close()
    eval_env.close()


def evaluate(args: Namespace):

    torch.set_num_threads(1)

    # init experiment
    experiment = ExperimentInitializer(args)
    experiment.configure_logging()
    experiment_name = experiment.get_name()

    # device
    device = torch.device("cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # kill any possibility of usage

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # env
    env, net_shapes, _, max_ac, _ = make_env(
        args.env_id, args.wrap_absorb, record=args.record, render=args.render)

    # create an agent wrapper
    def agent_wrapper():
        return SPPAgent(
            net_shapes=net_shapes,
            max_ac=max_ac,
            device=device,
            hps=args,
            expert_dataset=None,
            replay_buffer=None,
        )

    # evaluate
    orchestrator.evaluate(
        args=args,
        env=env,
        agent_wrapper=agent_wrapper,
        experiment_name=experiment_name,
    )

    # cleanup
    env.close()


if __name__ == "__main__":

    args: Namespace = argparser().parse_args()

    args.root = Path(__file__).resolve().parent  # make the paths absolute
    for k in ("checkpoints", "logs", "videos"):
        new_k = f"{k[:-1]}_dir"
        vars(args)[new_k] = Path(args.root) / k

    if args.task == "train":
        train(args)
    elif args.task == "evaluate":
        evaluate(args)
    else:
        raise NotImplementedError
