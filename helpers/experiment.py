import random
from pathlib import Path
import subprocess

import yaml

from helpers import logger


def uuid(num_syllables=2, num_parts=3):
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


class ConfigDumper:

    def __init__(self, args, path):
        # log the job config into a file
        self.args = args
        Path(path).mkdir(exist_ok=True)
        self.path = path

    def dump(self):
        hpmap = self.args.__dict__
        path = Path(self.path) / "hyperparameter.yml"
        path.write_text(yaml.dump(hpmap, default_flow_style=False))


class ExperimentInitializer:

    def __init__(self, args, rank=None, world_size=None):
        self.uuid_provided = (args.uuid is not None)
        self.uuid = args.uuid if self.uuid_provided else uuid()
        self.args = args
        self.rank = rank
        self.world_size = world_size

    def configure_logging(self):

        if self.rank is None:  # task: evaluate
            logger.info("configuring logger for evaluation")
            logger.configure(dir_=None, format_strs=["stdout"])

        elif self.rank == 0:  # task: train, i.e. master here
            log_path = Path(self.args.log_dir) / self.get_name()
            formats_strs = ["stdout", "log", "csv"]
            fmtstr = "configuring logger"
            if self.rank == 0:
                fmtstr += " [master]"
            logger.info(fmtstr)
            logger.configure(dir_=log_path, format_strs=formats_strs)
            fmtstr = "logger configured"
            if self.rank == 0:
                fmtstr += " [master]"
            logger.info(fmtstr)
            logger.info(f"  directory: {log_path}")
            logger.info(f"  output formats: {formats_strs}")

            # in the same log folder, log args in a YAML file
            config_dumper = ConfigDumper(args=self.args, path=log_path)
            config_dumper.dump()
            logger.info(f"experiment configured [{self.world_size} MPI workers]")

        else:  # train, worker
            logger.info(f"configuring logger [worker #{self.rank}]")
            logger.configure(dir_=None, format_strs=None)
            logger.set_level(logger.DISABLED)

    def get_name(self):

        if self.uuid_provided:
            # if the uuid has been provided, use it
            return self.uuid

        # assemble the uuid
        name = self.uuid
        try:
            out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            name += f".gitSHA_{out.strip().decode("ascii")}"
        except OSError:
            pass

        name += f".{self.args.env_id}"
        name += f".{self.args.algo}"
        name += f".seed{str(self.args.seed).zfill(2)}"

        if self.args.task == "train":
            name += f".train=demos{str(self.args.num_demos).zfill(3)}"
        elif self.args.task == "evaluate":
            name += f".eval=trajs{str(self.args.num_trajs).zfill(2)}"
        else:
            raise ValueError("only tasks: train and evaluate")

        return name
