import argparse
import itertools
from copy import deepcopy
import os
import sys
import numpy as np
import subprocess
import yaml
from pathlib import Path
from typing import Any

from helpers import logger
from helpers.misc_util import zipsame, boolean_flag
from helpers.experiment import uuid as create_uuid


ENV_BUNDLES = {
    "mujoco": {
        "debug": ["Hopper-v3"],
        "idp": ["InvertedDoublePendulum-v2"],
        "walker": ["Walker2d-v3"],
        "eevee": ["InvertedPendulum-v2",
                  "InvertedDoublePendulum-v2"],
        "jolteon": ["Hopper-v3",
                    "Walker2d-v3",
                    "HalfCheetah-v3"],
        "flareon": ["InvertedDoublePendulum-v2",
                    "Ant-v3"],
        "glaceon": ["Hopper-v3",
                    "Walker2d-v3",
                    "HalfCheetah-v3",
                    "Ant-v3"],
        "humanoid": ["Humanoid-v3"],
        "ant": ["Ant-v3"],
        "suite": ["InvertedDoublePendulum-v2",
                  "Hopper-v3",
                  "Walker2d-v3",
                  "HalfCheetah-v3",
                  "Ant-v3"],
    },
    "dmc": {
        "debug": ["Hopper-Hop-Feat-v0"],
        "flareon": ["Hopper-Hop-Feat-v0",
                    "Walker-Run-Feat-v0"],
        "glaceon": ["Hopper-Hop-Feat-v0",
                    "Cheetah-Run-Feat-v0",
                    "Walker-Run-Feat-v0"],
        "stacker": ["Stacker-Stack_2-Feat-v0",
                    "Stacker-Stack_4-Feat-v0"],
        "humanoid": ["Humanoid-Walk-Feat-v0",
                     "Humanoid-Run-Feat-v0"],
        "cmu": ["Humanoid_CMU-Stand-Feat-v0",
                "Humanoid_CMU-Run-Feat-v0"],
        "quad": ["Quadruped-Walk-Feat-v0",
                 "Quadruped-Run-Feat-v0",
                 "Quadruped-Escape-Feat-v0",
                 "Quadruped-Fetch-Feat-v0"],
        "dog": ["Dog-Run-Feat-v0",
                "Dog-Fetch-Feat-v0"],
    },
}

MEMORY = 16
NUM_NODES = 1
NUM_WORKERS = 1
NUM_SWEEP_TRIALS = 10


class Spawner(object):

    def __init__(self, args: argparse.Namespace):
        self.args = args

        # retrieve config from filesystem
        with Path(self.args.config).open() as f:
            self.config = yaml.safe_load(f)
        # make proper list of number of demos to tackle
        self.num_demos = [int(i) for i in self.args.num_demos]  # `num_demos` is a list!
        # assemble wandb project name
        proj = self.config["wandb_project"].upper()
        depl = self.args.deployment.upper()
        self.wandb_project = f"{proj}-{depl}"
        # define spawn type
        self.type = "sweep" if self.args.sweep else "fixed"
        # define the needed memory in GB
        self.memory = MEMORY

        # write out the boolean arguments (using the "boolean_flag" function)
        self.bool_args = [
            "cuda",
            "f16",
            "mps",
            "record",
            "layer_norm",
            "n_step_returns",
            "ret_norm",
            "popart",
            "clipped_double",
            "targ_actor_smoothing",
            "use_c51",
            "use_qr",
            "state_only",
            "minimax_only",
            "spectral_norm",
            "grad_pen",
            "one_sided_pen",
            "wrap_absorb",
            "d_batch_norm",
            "historical_patching",
        ]

        if self.args.deployment == "slurm":
            # translate intuitive caliber into duration and cluster partition
            calibers = {
                "short": "0-06:00:00",
                "long": "0-12:00:00",
                "verylong": "1-00:00:00",
                "veryverylong": "2-00:00:00",
                "veryveryverylong": "4-00:00:00",
            }
            self.duration = calibers[self.args.caliber]  # KeyError trigger if invalid caliber
            if "verylong" in self.args.caliber:
                if self.config["cuda"]:
                    self.partition = "private-cui-gpu"
                else:
                    self.partition = "public-cpu,private-cui-cpu,public-longrun-cpu"
            elif self.config["cuda"]:
                self.partition = "shared-gpu,private-cui-gpu"
            else:
                self.partition = "shared-cpu,public-cpu,private-cui-cpu"

        # define the set of considered environments from the considered suite
        self.envs = ENV_BUNDLES[self.config["benchmark"]][self.args.env_bundle]

        # create the list of demonstrations associated with the environments
        demo_dir = os.environ["DEMO_DIR"]
        self.demos = {k: Path(demo_dir) / k for k in self.envs}

        if "load_checkpoint" in self.config:
            # add the path to the pretrained model
            self.load_checkpoint = Path(os.environ["MODEL_DIR"]) / self.config["load_checkpoint"]

    @staticmethod
    def copy_and_add_seed(hpmap: dict[str, Any], seed: int) -> dict[str, Any]:
        hpmap_ = deepcopy(hpmap)

        # add the seed and edit the job uuid to only differ by the seed
        hpmap_.update({"seed": seed})

        # enrich the uuid with extra information
        gitsha = ""
        try:
            out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            sha = out.strip().decode("ascii")
            gitsha = f"gitSHA_{sha}"
        except OSError:
            pass

        uuid = f"{hpmap["uuid"]}.{gitsha}.{hpmap["env_id"]}_{NUM_WORKERS}"
        uuid += f".demos{str(hpmap["num_demos"]).zfill(3)}"
        uuid += f".seed{str(seed).zfill(2)}"  # add seed

        # update uuid in map
        hpmap_.update({"uuid": uuid})

        return hpmap_

    def copy_and_add_env(self, hpmap: dict[str, Any], env: str) -> dict[str, Any]:
        hpmap_ = deepcopy(hpmap)

        # add the env and demos
        hpmap_.update({"env_id": env, "expert_path": self.demos[env]})

        # overwrite per-env hps
        if env == "Hopper-v3":
            old_gamma = hpmap_["gamma"]
            new_gamma = 0.995
            logger.info(f"overwrite discount for {env}: {old_gamma} -> {new_gamma}")
            hpmap_.update({"gamma": 0.995})

        return hpmap_

    @staticmethod
    def copy_and_add_num_demos(hpmap: dict[str, Any], num_demos: int) -> dict[str, Any]:
        hpmap_ = deepcopy(hpmap)
        # add the num of demos
        hpmap_.update({"num_demos": num_demos})
        return hpmap_

    def get_hps(self):
        """Return a list of maps of hyperparameters"""

        # create a uuid to identify the current job
        uuid = create_uuid()

        # assemble the hyperparameter map
        hpmap = {
            "wandb_project": self.wandb_project,
            "uuid": uuid,
            "cuda": self.config["cuda"],
            "fp16": self.config["fp16"],
            "record": self.config.get("record", False),
            "task": self.config["task"],

            # training
            "num_timesteps": int(float(self.config.get("num_timesteps", 2e7))),
            "training_steps_per_iter": self.config.get("training_steps_per_iter", 2),
            "eval_steps_per_iter": self.config.get("eval_steps_per_iter", 10),
            "eval_every": self.config.get("eval_every", 10),
            "save_every": self.config.get("save_every", 10),

            # model
            "layer_norm": self.config["layer_norm"],

            # optimization
            "actor_lr": float(self.config.get("actor_lr", 3e-4)),
            "critic_lr": float(self.config.get("critic_lr", 3e-4)),
            "clip_norm": self.config["clip_norm"],
            "wd_scale": float(self.config.get("wd_scale", 3e-4)),
            "acc_grad_steps": int(self.config.get("acc_grad_step", 8)),

            # algorithm
            "rollout_len": self.config.get("rollout_len", 2),
            "batch_size": self.config.get("batch_size", 128),
            "gamma": self.config.get("gamma", 0.99),
            "mem_size": int(self.config.get("mem_size", 100000)),
            "noise_type": self.config["noise_type"],
            "pn_adapt_frequency": self.config.get("pn_adapt_frequency", 50),
            "polyak": self.config.get("polyak", 0.005),
            "targ_up_freq": self.config.get("targ_up_freq", 100),
            "n_step_returns": self.config.get("n_step_returns", False),
            "lookahead": self.config.get("lookahead", 10),
            "ret_norm": self.config.get("ret_norm", False),
            "popart": self.config.get("popart", False),

            # TD3
            "clipped_double": self.config.get("clipped_double", False),
            "targ_actor_smoothing": self.config.get("targ_actor_smoothing", False),
            "td3_std": self.config.get("td3_std", 0.2),
            "td3_c": self.config.get("td3_c", 0.5),
            "actor_update_delay": self.config.get("actor_update_delay", 2),

            # distributional RL
            "use_c51": self.config.get("use_c51", False),
            "use_qr": self.config.get("use_qr", False),
            "c51_num_atoms": self.config.get("c51_num_atoms", 51),
            "c51_vmin": self.config.get("c51_vmin", -10.),
            "c51_vmax": self.config.get("c51_vmax", 10.),
            "num_tau": self.config.get("num_tau", 200),

            # AIL
            "g_steps": self.config.get("g_steps", 3),
            "d_steps": self.config.get("d_steps", 1),
            "d_lr": float(self.config.get("d_lr", 1e-5)),
            "state_only": self.config.get("state_only", True),
            "minimax_only": self.config.get("minimax_only", True),
            "ent_reg_scale": self.config.get("ent_reg_scale", 0.001),
            "spectral_norm": self.config.get("spectral_norm", True),
            "grad_pen": self.config.get("grad_pen", True),
            "grad_pen_targ": self.config.get("grad_pen_targ", 1.),
            "grad_pen_scale": self.config.get("grad_pen_scale", 10.),
            "one_sided_pen": self.config.get("one_sided_pen", True),
            "historical_patching": self.config.get("historical_patching", True),
            "wrap_absorb": self.config.get("wrap_absorb", False),
            "d_batch_norm": self.config.get("d_batch_norm", False),
        }
        if "load_checkpoint" in self.config:
            hpmap.update({"load_checkpoint": self.load_checkpoint})

        if self.args.sweep:
            # random search: replace some entries with random values
            rng = np.random.default_rng(seed=None)
            hpmap.update({
                "batch_size": int(rng.choice([64, 128, 256])),
                "lr": float(rng.choice([1e-4, 3e-4])),
            })

        # carry out various duplications

        # duplicate for each environment
        hpmaps = [self.copy_and_add_env(hpmap, env) for env in self.envs]

        # duplicate for each number of demos
        hpmaps = [self.copy_and_add_num_demos(hpmap_, num_demos)
                  for hpmap_ in hpmaps
                  for num_demos in self.num_demos]

        # duplicate for each seed
        hpmaps = [self.copy_and_add_seed(hpmap_, seed)
                  for hpmap_ in hpmaps
                  for seed in range(self.args.num_seeds)]

        # verify that the correct number of configs have been created
        assert len(hpmaps) == self.args.num_seeds * len(self.envs) * len(self.num_demos)

        return hpmaps

    def unroll_options(self, hpmap: dict[str, Any]) -> str:
        """Transform the dictionary of hyperparameters into a string of bash options"""
        indent = 4 * " "  # choice: indents are defined as 4 spaces
        arguments = ""
        for k, v in hpmap.items():
            argument = (f"{k}" if v else f"no-{k}") if k in self.bool_args else f"{k}={v}"
            arguments += f"{indent}--{argument} \\\n"
        return arguments

    def create_job_str(self, name: str, command: str) -> str:
        """Build the batch script that launches a job"""

        # prepend python command with python binary path
        cmd = Path(os.environ["CONDA_PREFIX"]) / "bin" / command

        if self.args.deployment == "slurm":
            Path("./out").mkdir(exist_ok=True)
            # set sbatch config
            bash_script_str = ("#!/usr/bin/env bash\n\n")
            bash_script_str += (f"#SBATCH --job-name={name}\n"
                                f"#SBATCH --partition={self.partition}\n"
                                f"#SBATCH --nodes={NUM_NODES}\n"
                                f"#SBATCH --ntasks={NUM_WORKERS}\n"
                                "#SBATCH --cpus-per-task=4\n"
                                f"#SBATCH --time={self.duration}\n"
                                f"#SBATCH --mem={self.memory}000\n"
                                "#SBATCH --output=./out/run_%j.out\n")
            if self.args.deployment == "slurm":
                # Sometimes versions are needed (some clusters)
                if self.config["cuda"]:
                    constraint = ""
                    bash_script_str += ("#SBATCH --gpus=1\n")  # gpus=titan:1 if needed
                    if constraint:  # if not empty
                        bash_script_str += (f'#SBATCH --constraint="{constraint}"\n')
                bash_script_str += ("\n")

            # load modules
            bash_script_str += ("module load GCC/9.3.0\n")
            if self.config["cuda"]:
                bash_script_str += ("module load CUDA/11.5.0\n")

            if self.config["benchmark"] == "dmc":
                bash_script_str += ("module load Mesa/19.2.1\n")

            bash_script_str += ("\n")

            # launch command
            if self.args.deployment == "slurm":
                bash_script_str += (f"srun {cmd}")

        elif self.args.deployment == "tmux":
            # set header
            bash_script_str = ("#!/usr/bin/env bash\n\n")
            bash_script_str += (f"# job name: {name}\n\n")
            # launch command
            bash_script_str += (f"{cmd}")  # left in this format for easy edits

        else:
            raise NotImplementedError("cluster selected is not covered.")

        return bash_script_str[:-2]  # remove the last `\` and `\n` tokens


def run(args: argparse.Namespace):
    """Spawn jobs"""

    if args.wandb_upgrade:
        # upgrade the wandb package
        logger.info("====upgrading wandb pip package")
        out = subprocess.check_output([
            sys.executable, "-m", "pip", "install", "wandb", "--upgrade",
        ])
        logger.info(out.decode("utf-8"))

    # create a spawner object
    spawner = Spawner(args)

    # create directory for spawned jobs
    root = Path(__file__).resolve().parent
    spawn_dir = Path(root) / "spawn"
    Path(spawn_dir).mkdir(exist_ok=True)
    tmux_dir = Path(root) / "tmux"  # create name to prevent unbound from type-checker
    if args.deployment == "tmux":
        Path(tmux_dir).mkdir(exist_ok=True)

    # get the hyperparameter set(s)
    if args.sweep:
        hpmaps_ = [spawner.get_hps() for _ in range(NUM_SWEEP_TRIALS)]
        # flatten into a 1-dim list
        hpmaps = [x for hpmap in hpmaps_ for x in hpmap]
    else:
        hpmaps = spawner.get_hps()

    # create associated task strings
    commands = [f"python main.py \\\n{spawner.unroll_options(hpmap)}" for hpmap in hpmaps]
    if not len(commands) == len(set(commands)):
        # terminate in case of duplicate experiment (extremely unlikely though)
        raise ValueError("bad luck, there are dupes -> Try again (:")
    # create the job maps
    names = [f"{spawner.type}.{hpmap["uuid"]}.{i}" for i, hpmap in enumerate(hpmaps)]

    # finally get all the required job strings
    jobs = itertools.starmap(spawner.create_job_str, zipsame(names, commands))

    # spawn the jobs
    for i, (name, job) in enumerate(zipsame(names, jobs)):
        logger.info(f"job#={i},name={name} -> ready to be deployed.")
        if args.debug:
            logger.info("config below.")
            logger.info(job + "\n")
        dirname = name.split(".")[1]
        full_dirname = Path(spawn_dir) / dirname
        Path(full_dirname).mkdir(exist_ok=True)
        job_name = Path(full_dirname) / f"{name}.sh"
        job_name.write_text(job)
        if args.deploy_now and args.deployment != "tmux":
            # spawn the job!
            stdout = subprocess.run(["sbatch", job_name], check=True).stdout
            if args.debug:
                logger.info(f"[STDOUT]\n{stdout}")
            logger.info(f"job#={i},name={name} -> deployed on slurm.")

    if args.deployment == "tmux":
        dir_ = hpmaps[0]["uuid"].split(".")[0]  # arbitrarilly picked index 0
        session_name = f"{spawner.type}-{str(args.num_seeds).zfill(2)}seeds-{dir_}"
        yaml_content = {"session_name": session_name,
                        "windows": [],
                        "environment": {"DEMO_DIR": os.environ["DEMO_DIR"]}}
        for i, name in enumerate(names):
            executable = f"{name}.sh"
            pane = {"shell_command": [f"source activate {args.conda_env}",
                                      f"chmod u+x spawn/{dir_}/{executable}",
                                      f"spawn/{dir_}/{executable}"]}
            window = {"window_name": f"job{str(i).zfill(2)}",
                      "focus": False,
                      "panes": [pane]}
            yaml_content["windows"].append(window)
            logger.info(
                f"job#={i},name={name} -> will run in tmux, session={session_name},window={i}.",
            )

        # dump the assembled tmux config into a yaml file
        job_config = Path(tmux_dir) / f"{session_name}.yaml"
        job_config.write_text(yaml.dump(yaml_content, default_flow_style=False))
        if args.deploy_now:
            # spawn all the jobs in the tmux session!
            stdout = subprocess.run(["tmuxp", "load", "-d", job_config], check=True).stdout
            if args.debug:
                logger.info(f"[STDOUT]\n{stdout}")
            logger.info(
                f"[{len(list(jobs))}] jobs are now running in tmux session =={session_name}==.",
            )
    else:
        # summarize the number of jobs spawned
        logger.info(f"[{len(list(jobs))}] jobs were spawned.")


if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser(description="Job Spawner")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--conda_env", type=str, default=None)
    parser.add_argument("--env_bundle", type=str, default=None)
    parser.add_argument("--deployment", type=str,
        choices=["tmux", "slurm", "slurm2"],
        default="tmux",
        help="deploy how?",
    )
    parser.add_argument("--num_seeds", type=int, default=None)
    parser.add_argument("--caliber", type=str,
        choices=[
            "short", "long", "verylong", "veryverylong", "veryveryverylong",
        ], default="short")
    boolean_flag(parser, "deploy_now", default=True, hint="deploy immediately?")
    boolean_flag(parser, "sweep", default=False, hint="hp search?")
    boolean_flag(parser, "wandb_upgrade", default=True, hint="upgrade wandb?")
    boolean_flag(parser, "wandb_dryrun", default=True, hint="toggle wandb offline mode")
    parser.add_argument("--num_demos", "--list", nargs="+", type=str, default=None)
    boolean_flag(parser, "debug", default=False, hint="toggle debug/verbose mode in spawner")
    args = parser.parse_args()

    if args.wandb_dryrun:
        # run wandb in offline mode (does not sync with wandb servers in real time,
        # use `wandb sync` later on the local directory in `wandb/`
        # to sync to the wandb cloud hosted app)
        os.environ["WANDB_MODE"] = "dryrun"

    # create (and optionally deploy) the jobs
    run(args)
