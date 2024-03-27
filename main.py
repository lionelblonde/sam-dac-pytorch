import os

from mpi4py import MPI

import numpy as np
import torch
from torch.backends import cudnn as cudnn

import orchestrator
from helpers import logger
from helpers.console_util import log_env_info
from helpers.argparsers import argparser
from helpers.experiment import ExperimentInitializer
from helpers.distributed_util import setup_mpi_gpus
from helpers.env_makers import make_env
from helpers.dataset import DemoDataset
from helpers.memory import ReplayBuffer
from agents.spp_agent import SPPAgent


def train(args):

    # Get the current process rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    args.algo = args.algo + '_' + str(world_size).zfill(3)

    torch.set_num_threads(1)

    # Initialize and configure experiment
    experiment = ExperimentInitializer(args, rank=rank, world_size=world_size)
    experiment.configure_logging()
    name = experiment.get_name()

    # device
    if args.cuda:
        assert torch.cuda.is_available()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda:0")
        setup_mpi_gpus()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # kill any possibility of usage
        device = torch.device("cpu")
    args.device = device  # add the device to hps for convenience
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

    # Create an evaluation environment not to mess up with training rollouts
    eval_env = None
    if rank == 0:
        eval_env = make_env(args.env_id, eval_seed)

    # Train
    orchestrator.learn(
        args=args,
        rank=rank,
        env=env,
        eval_env=eval_env,
        agent_wrapper=agent_wrapper,
        experiment_name=name,
    )

    # Close environment
    env.close()

    # Close the eval env
    if eval_env is not None:
        assert rank == 0
        eval_env.close()


def evaluate(args):

    # Initialize and configure experiment
    experiment = ExperimentInitializer(args)
    experiment.configure_logging()
    # Create experiment name
    experiment_name = experiment.get_name()

    # Seedify
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Create environment
    env = make_env(args.env_id, args.seed)

    # Create an agent wrapper
    if args.algo == 'sam-dac':
        def agent_wrapper():
            return SAMAgent(
                env=env,
                device='cpu',
                hps=args,
                expert_dataset=None,
            )
    else:
        raise NotImplementedError("algorithm not covered")

    # Evaluate
    orchestrator.evaluate(
        args=args,
        env=env,
        agent_wrapper=agent_wrapper,
        experiment_name=experiment_name,
    )

    # Close environment
    env.close()


if __name__ == '__main__':
    _args = argparser().parse_args()

    # Make the paths absolute
    _args.root = os.path.dirname(os.path.abspath(__file__))
    for k in ['checkpoints', 'logs', 'videos']:
        new_k = "{}_dir".format(k[:-1])
        vars(_args)[new_k] = os.path.join(_args.root, k)

    if _args.task == 'train':
        train(_args)
    elif _args.task == 'eval':
        evaluate(_args)
    else:
        raise NotImplementedError
