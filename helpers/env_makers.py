import os
from pathlib import Path

import gym

import numpy as np

from helpers import logger
from helpers.dmc_envs import make_dmc
import environments


def get_benchmark(env_id):
    # verify that the specified env is amongst the admissible ones
    benchmark = None
    for k, v in environments.BENCHMARKS.items():
        if env_id in v:
            benchmark = k
            continue
    assert benchmark is not None, "unsupported environment"
    return benchmark


def make_env(env_id, seed, wrap_absorb):
    # create an environment
    benchmark = get_benchmark(env_id)

    if benchmark == "dmc":
        # import here to avoid glew issues altogether if not using anyway
        return make_dmc(env_id)

    if benchmark == "mujoco":
        # remove the lockfile if it exists
        lockfile = (Path(os.environ["CONDA_PREFIX"]) / "lib" / "python3.7" / "site-packages" /
                    "mujoco_py" / "generated" / "mujocopy-buildlock.lock")
        try:
            Path(lockfile).unlink()
            logger.info("[WARN] removed mujoco lockfile")
        except OSError:
            pass

    # create env and seed it
    env = gym.make(env_id)
    env.seed(seed)

    # build shapes for nets and replay buffer
    shapes = {}

    # for the nets
    ob_shape = env.obvervation_space.shape
    ac_space = env.action_space  # used now and later to get max action
    if hasattr(ac_space, "n"):
        raise AttributeError(f"env has discrete dim: {ac_space.n}")
    ac_shape = ac_space.shape
    shapes.update({"ob_shape": ob_shape, "ac_shape": ac_shape})

    # for the replay buffer
    if wrap_absorb:
        assert len(list(ob_shape)) == 1, "wrap absorb only work for non-pix envs"
        ob_dim = ob_dim_orig = ob_shape[0]
        ac_dim = ac_dim_orig = ac_shape[0]  # for both: num dims
        ob_dim += 1
        ac_dim += 1
        shapes.update({
            "obs0": (ob_dim,),
            "obs1": (ob_dim,),
            "acs": (ac_dim,),
            "rews": (1,),
            "dones1": (1,),
            "obs0_orig": (ob_dim_orig,),
            "obs1_orig": (ob_dim_orig,),
            "acs_orig": (ac_dim_orig,),
        })
    else:
        shapes.update({
            "obs0": ob_shape,
            "obs1": ob_shape,
            "acs": ac_shape,
            "rews": (1,),
            "dones1": (1,),
        })

    # max value for action
    max_ac = max(
        np.abs(np.amax(ac_space.high.astype("float32"))),
        np.abs(np.amin(ac_space.low.astype("float32"))),
    )

    if benchmark == "mujoco":
        pass  # weird, but struct kept general if adding other envs
    else:
        raise ValueError("unsupported benchmark")

    return env, shapes, max_ac
