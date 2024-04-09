import os
import subprocess
from pathlib import Path
from typing import Optional

import fire
import yaml
import random
import numpy as np
import torch

import orchestrator
from helpers import logger
from helpers.env_makers import make_env
from helpers.dataset import DemoDataset
from agents.memory import ReplayBuffer
from agents.spp_agent import SPPAgent


DISABLE_LOGGER = False


def make_uuid(num_syllables=2, num_parts=3):
    """Randomly create a semi-pronounceable uuid"""
    part1 = ["s", "t", "r", "ch", "b", "c", "w", "z", "h", "k", "p", "ph", "sh", "f", "fr"]
    part2 = ["a", "oo", "ee", "e", "u", "er"]
    seps = ["_"]  # [ "-", "_", "."]
    result = ""
    for i in range(num_parts):
        if i > 0:
            result += seps[random.randrange(len(seps))]
        indices1 = [random.randrange(len(part1)) for _ in range(num_syllables)]
        indices2 = [random.randrange(len(part2)) for _ in range(num_syllables)]
        for i1, i2 in zip(indices1, indices2):
            result += part1[i1] + part2[i2]
    return result


def get_name(uuid, env_id, seed):
    """Assemble long experiment name"""
    name = uuid
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        sha = out.strip().decode("ascii")
        name += f".gitSHA_{sha}"
    except OSError:
        pass
    name += f".{env_id}"
    name += f".seed{str(seed).zfill(2)}"
    return name


class MagicRunner(object):

    def __init__(self, cfg: str,  # give the relative path to cfg here
                 uuid: Optional[str] = None,
                 wandb_project: Optional[str] = None,
                 env_id: Optional[str] = None,
                 seed: Optional[int] = None,
                 num_demos: Optional[int] = None,
                 expert_path: Optional[str] = None,
                 load_checkpoint: Optional[str] = None):

        # retrieve config from filesystem
        proj_root = Path(__file__).resolve().parent
        with (proj_root / Path(cfg)).open() as f:
            self._cfg = yaml.safe_load(f)

        self._cfg["root"] = proj_root  # in config: used by wandb
        for k in ("checkpoints", "logs", "videos"):
            new_k = f"{k[:-1]}_dir"
            self._cfg[new_k] = Path(self._cfg["root"]) / k

        # set only if nonexistant key in cfg
        self._cfg["uuid"] = self._cfg.get("uuid", uuid)  # key exists now, but can be None
        self._cfg["wandb_project"] = self._cfg.get("wandb_project", wandb_project)
        self._cfg["seed"] = self._cfg.get("seed", seed)
        self._cfg["env_id"] = self._cfg.get("env_id", env_id)
        self._cfg["num_demos"] = self._cfg.get("num_demos", num_demos)
        self._cfg["expert_path"] = self._cfg.get("expert_path", expert_path)
        self._cfg["load_checkpoint"] = self._cfg.get("load_checkpoint", load_checkpoint)

        if self._cfg["uuid"] is None:
            self._cfg["uuid"] = make_uuid()
        self.name = get_name(self._cfg["uuid"], self._cfg["env_id"], self._cfg["seed"])

    def train(self):

        # mlsys
        torch.set_num_threads(1)  # TODO(lionel): keep an eye on this

        # set printing options
        np.set_printoptions(precision=3)

        # name
        name = f"{self.name}.train_demos{str(self._cfg['num_demos']).zfill(3)}"
        # logger
        if DISABLE_LOGGER:
            logger.set_level(logger.DISABLED)  # turn the logging off
        else:
            log_path = Path(self._cfg["log_dir"]) / name
            log_path.mkdir(exist_ok=True)
            logger.configure(directory=log_path, format_strs=["stdout", "log", "json", "csv"])
            # config dump
            (log_path / "config.yml").write_text(yaml.dump(self._cfg, default_flow_style=False))

        # device
        assert not self._cfg["fp16"] or self._cfg["cuda"], "fp16 => cuda"
        if self._cfg["cuda"]:
            # use cuda
            assert torch.cuda.is_available()
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            device = torch.device("cuda:0")
        else:
            if self._cfg["mps"]:  # TODO(lionel): add this as hp
                assert torch.mps
                # use Apple"s Metal Performance Shaders (MPS)
                device = torch.device("mps:0")
            else:
                # default case: just use plain old cpu, no cuda or m-chip gpu
                device = torch.device("cpu")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""  # kill any possibility of usage
        logger.info(f"device in use: {device}")

        # seed
        torch.manual_seed(self._cfg["seed"])
        torch.cuda.manual_seed_all(self._cfg["seed"])
        np_rng = np.random.default_rng(self._cfg["seed"])

        # env
        env, net_shapes, erb_shapes, max_ac, max_episode_steps = make_env(
            self._cfg["env_id"],
            self._cfg["wrap_absorb"],
            record=False,
            render=self._cfg["render"],
        )

        # create an agent wrapper

        expert_dataset = DemoDataset(
            np_rng=np_rng,
            expert_path=self._cfg["expert_path"],
            num_demos=self._cfg["num_demos"],
            max_ep_steps=max_episode_steps,
            wrap_absorb=self._cfg["wrap_absorb"],
        )
        replay_buffer = ReplayBuffer(
            np_rng=np_rng,
            capacity=self._cfg["mem_size"],
            erb_shapes=erb_shapes,
        )
        logger.info(f"{replay_buffer} configured")

        def agent_wrapper():
            return SPPAgent(
                net_shapes=net_shapes,
                max_ac=max_ac,
                device=device,
                hps=self._cfg,
                expert_dataset=expert_dataset,
                replay_buffer=replay_buffer,
            )

        # create an evaluation environment not to mess up with training rollouts
        eval_env, _, _, _, _ = make_env(
            self._cfg["env_id"],
            self._cfg["wrap_absorb"],
            record=self._cfg["record"],
            render=self._cfg["render"],
        )

        # train
        orchestrator.learn(
            cfg=self._cfg,
            env=env,
            eval_env=eval_env,
            agent_wrapper=agent_wrapper,
            name=name,
        )

        # cleanup
        env.close()
        eval_env.close()

    def evaluate(self):

        # mlsys
        torch.set_num_threads(1)  # TODO(lionel): keep an eye on this

        # set printing options
        np.set_printoptions(precision=3)

        # name
        name = f"{self.name}.eval_trajs{str(self._cfg['num_trajs']).zfill(2)}"
        # logger
        if DISABLE_LOGGER:
            logger.set_level(logger.DISABLED)  # turn the logging off
        else:
            logger.configure(directory=None, format_strs=["stdout"])

        # device
        device = torch.device("cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # kill any possibility of usage

        # seed
        torch.manual_seed(self._cfg["seed"])
        torch.cuda.manual_seed_all(self._cfg["seed"])

        # env
        env, net_shapes, _, max_ac, _ = make_env(
            self._cfg["env_id"],
            self._cfg["wrap_absorb"],
            record=self._cfg["record"],
            render=self._cfg["render"],
        )

        # create an agent wrapper
        def agent_wrapper():
            return SPPAgent(
                net_shapes=net_shapes,
                max_ac=max_ac,
                device=device,
                hps=self._cfg,
                expert_dataset=None,
                replay_buffer=None,
            )

        # evaluate
        orchestrator.evaluate(
            cfg=self._cfg,
            env=env,
            agent_wrapper=agent_wrapper,
            name=name,
        )

        # cleanup
        env.close()


if __name__ == "__main__":
    fire.Fire(MagicRunner)
