from collections import defaultdict
from pathlib import Path

from typing import Union
import h5py
from beartype import beartype
import numpy as np
from numpy.random import Generator
from torch.utils.data import Dataset

from helpers import logger


@beartype
def save_dict_h5py(data: dict[str, np.ndarray], fname: Union[str, Path]):
    """Save dictionary containing numpy arrays to h5py file."""
    with h5py.File(fname, "w") as hf:
        for key in data:
            hf.create_dataset(key, data=data[key])


@beartype
def load_dict_h5py(fname: Union[str, Path]) -> dict[str, np.ndarray]:
    """Restore dictionary containing numpy arrays from h5py file."""
    data = {}
    with h5py.File(fname, "r") as hf:
        for key in hf:
            dset = hf[key]
            if isinstance(dset, h5py.Dataset):
                data[key] = dset[()]
            else:
                raise TypeError(f"dset for key {key} has wrong type")
    return data


class DictDataset(Dataset):

    @beartype
    def __init__(self, data: dict[str, np.ndarray]):
        self.data = data

    @beartype
    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        return {k: v[i, ...].astype(np.float32) for k, v in self.data.items()}

    @beartype
    def __len__(self) -> int:
        return len(next(iter(self.data.values())))


class DemoDataset(DictDataset):

    @beartype
    def __init__(self,
                 np_rng: Generator,
                 expert_path: str,
                 num_demos: int,
                 max_ep_steps: int,
                 *,
                 wrap_absorb: bool):
        logger.info("creating dataset")
        logger.info(f"spec::expert path: {expert_path}")
        logger.info(f"spec::num_demos: {num_demos}")
        self.num_demos = num_demos
        self.stats, self.data = defaultdict(list), defaultdict(list)
        logger.info("::::loading demos")

        # go over the demos, sorted in alphabetical order
        for i, f in enumerate(sorted(Path(expert_path).glob("*.h5"))):

            # if the desired number of demos has been aggregated, leave
            if i == num_demos:
                break

            # log the location of the loaded demo
            logger.info(f"[DEMO DATASET]::demo #{str(i).zfill(3)} loaded from: {f}")

            # load the demo from the file
            tmp = load_dict_h5py(f)

            # remove undesirable keys (at least in this application)
            assert tmp["dones1"][-1], "by construction"  # making sure every ep ends with done
            try:
                tmp.pop("pix_obs0")
                tmp.pop("pix_obs1")
            except KeyError:
                logger.warn("keys to filter out are not in")
                logger.info("so no need to remove them")
            else:
                logger.info("keys properly removed")

            # extract and display content dims
            dims = {k: tmp[k].shape[1:] for k in tmp}
            dims = " | ".join([f"{k}={v}" for k, v in dims.items()])
            logger.info(f"[DEMO DATASET] dims: {dims}")

            # get episode statistics
            try:
                ep_len = tmp.pop("ep_lens")  # return and delete key
                ep_ret = tmp.pop("ep_env_rets")  # return and delete key
            except KeyError as ke:
                logger.error("required keys are missing")
                raise KeyError from ke

            padd_ep_len = "ep_len".ljust(20, "-")
            padd_ep_ret = "ep_ret".ljust(20, "-")
            logger.info(f"[DEMO DATASET]::{padd_ep_len}{ep_len}")
            logger.info(f"[DEMO DATASET]::{padd_ep_ret}{ep_ret}")
            self.stats["ep_len"].append(ep_len)
            self.stats["ep_ret"].append(ep_ret)

            # determine if terminal because of timeout or real termination
            terminal = ep_len != max_ep_steps

            # subsample trajectory: trajectories are not contiguous sequences
            sub_rate = 20  # N=20 in the original GAIL paper
            start = np_rng.integers(low=0, high=sub_rate)
            indices = [start + (i * sub_rate) for i in range(ep_len // sub_rate)]
            ep_len = len(indices)  # overwrite ep_len

            padd_substr = "subsample".ljust(15, "-")
            substr = f"{ep_len}(sub_rate={sub_rate})"
            logger.info(f"[DEMO DATASET]::{padd_substr}{substr}")
            for k in tmp:
                tmp[k] = tmp[k][indices]

            # collect the demo content
            if wrap_absorb:
                if tmp["dones1"][-1] and terminal:
                    # if the last subsampled transition is done, then it must be
                    # the very last transition of the episode, and testing whether it is
                    # a true terminal state is given by "terminal" determined above
                    logger.info("[DEMO DATASET]::wrapping with absorbing transition")
                    # wrap with an absorbing state
                    obs0 = np.concatenate(
                        [tmp["obs0"],
                         np.zeros((ep_len, 1))],
                        axis=-1,
                    )
                    acs = np.concatenate(
                        [tmp["acs"],
                         np.zeros((ep_len, 1))],
                        axis=-1,
                    )
                    obs1 = np.concatenate(
                        [tmp["obs1"],
                         np.concatenate(
                            [np.zeros((ep_len - 1, 1)),
                             np.ones((1, 1))],
                            axis=0)],
                        axis=-1,
                    )
                    # add absorbing transition
                    obs0 = np.concatenate([
                        obs0,
                        np.expand_dims(np.append(np.zeros_like(tmp["obs0"][-1]), 1), axis=0),
                    ], axis=0)
                    acs = np.concatenate([
                        acs,
                        np.expand_dims(np.append(np.zeros_like(tmp["acs"][-1]), 1), axis=0),
                    ], axis=0)
                    obs1 = np.concatenate([
                        obs1,
                        np.expand_dims(np.append(np.zeros_like(tmp["obs1"][-1]), 1), axis=0),
                    ], axis=0)
                    self.data["obs0"].append(obs0)
                    self.data["acs"].append(acs)
                    self.data["obs1"].append(obs1)
                else:
                    self.data["obs0"].append(np.concatenate([
                        tmp["obs0"],
                        np.zeros((ep_len, 1)),
                    ], axis=-1))
                    self.data["acs"].append(np.concatenate([
                        tmp["acs"],
                        np.zeros((ep_len, 1)),
                    ], axis=-1))
                    self.data["obs1"].append(np.concatenate([
                        tmp["obs1"],
                        np.zeros((ep_len, 1)),
                    ], axis=-1))

                self.data["obs0_orig"].append(tmp["obs0"])

            else:
                self.data["obs0"].append(tmp["obs0"])
                self.data["acs"].append(tmp["acs"])
                self.data["obs1"].append(tmp["obs1"])

        # transform structures into arrays
        self.np_stats, self.np_data = {}, {}
        for k, v in self.stats.items():
            self.np_stats[k] = np.array(v)
        for k, v in self.data.items():
            self.np_data[k] = np.concatenate(v, axis=0)

        self.stats = self.np_stats
        self.data = self.np_data

        logger.info(f"[DEMO DATASET]::keys extracted: {list(self.data.keys())}")
        lens, rets = self.stats["ep_len"], self.stats["ep_ret"]
        logger.info(f"[DEMO DATASET]::got {len(self)} transitions, from {self.num_demos} eps")
        logger.info(f"[DEMO DATASET]::episodic length: {np.mean(lens)}({np.std(lens)})")
        logger.info(f"[DEMO DATASET]::episodic return: {np.mean(rets)}({np.std(rets)})")
