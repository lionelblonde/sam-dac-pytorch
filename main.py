import os
from pathlib import Path

from mpi4py import MPI

import numpy as np
import torch

import orchestrator
from helpers import logger
from helpers.console_util import log_env_info
from helpers.argparsers import argparser
from helpers.experiment import ExperimentInitializer
from helpers.distributed_util import setup_mpi_gpus
from helpers.env_makers import make_env
from helpers.dataset import DemoDataset
from agents.memory import ReplayBuffer
from agents.spp_agent import SPPAgent


def train(args):

    # mlsys
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    args.algo = f"{args.algo}-{str(world_size).zfill(3)}"

    torch.set_num_threads(1)

    # set printing options
    np.set_printoptions(precision=3)

    # init experiment
    experiment = ExperimentInitializer(args, rank=rank, world_size=world_size)
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
        setup_mpi_gpus()
    else:
        if args.mps:  # TODO(lionel): add this as hp
            assert torch.has_mps
            # use Apple"s Metal Performance Shaders (MPS)
            device = torch.device("mps")
        else:
            # default case: just use plain old cpu, no cuda or m-chip gpu
            device = torch.device("cpu")

        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # kill any possibility of usage

    logger.info(f"device in use: {device}")

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    np_rng = np.random.default_rng(args.seed)

    worker_seed = args.seed + (1000000 * (rank + 1))
    eval_seed = args.seed + 1000000

    # env
    env, shapes, max_ac = make_env(args.env_id, worker_seed, args.wrap_absorb)
    log_env_info(logger, env)

    # create an agent wrapper

    expert_dataset = DemoDataset(
        np_rng=np_rng,
        expert_path=args.expert_path,
        num_demos=args.num_demos,
        max_ep_steps=env._max_episode_steps,  # careful here when porting to other envs
        wrap_absorb=args.wrap_absorb,
    )
    replay_buffer = ReplayBuffer(
        np_rng=np_rng,
        capacity=args.mem_size,
        shapes=shapes,
    )
    logger.info(f"{replay_buffer} configured")

    def agent_wrapper():
        return SPPAgent(
            shapes=shapes,
            max_ac=max_ac,
            device=device,
            hps=args,
            expert_dataset=expert_dataset,
            replay_buffer=replay_buffer,
        )

    # create an evaluation environment not to mess up with training rollouts
    eval_env = None
    if rank == 0:
        eval_env, _, _ = make_env(args.env_id, eval_seed, args.wrap_absorb)

    # train
    orchestrator.learn(
        args=args,
        rank=rank,
        env=env,
        eval_env=eval_env,
        agent_wrapper=agent_wrapper,
        experiment_name=experiment_name,
    )

    # cleanup

    env.close()

    if eval_env is not None:
        assert rank == 0
        eval_env.close()


def evaluate(args):

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
    env, shapes, max_ac = make_env(args.env_id, args.seed, args.wrap_absorb)
    log_env_info(logger, env)

    # create an agent wrapper
    def agent_wrapper():
        return SPPAgent(
            shapes=shapes,
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

    args = argparser().parse_args()

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
