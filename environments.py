# openai mujoco

MUJOCO_ROBOTS = [
    "InvertedPendulum",
    "InvertedDoublePendulum",
    "Reacher",
    "Hopper",
    "HalfCheetah",
    "Walker2d",
    "Ant",
    "Humanoid",
]

MUJOCO_ENVS = [f"{name}-v2" for name in MUJOCO_ROBOTS]
MUJOCO_ENVS.extend([f"{name}-v3" for name in MUJOCO_ROBOTS])

# dmc

DMC_ROBOTS = [
    "Hopper-Hop",
    "Cheetah-Run",
    "Walker-Walk",
    "Walker-Run",

    "Stacker-Stack_2",
    "Stacker-Stack_4",

    "Humanoid-Walk",
    "Humanoid-Run",
    "Humanoid-Run_Pure_State",

    "Humanoid_CMU-Stand",
    "Humanoid_CMU-Run",

    "Quadruped-Walk",
    "Quadruped-Run",
    "Quadruped-Escape",
    "Quadruped-Fetch",

    "Dog-Run",
    "Dog-Fetch",
]

DMC_ENVS = [f"{name}-Feat-v0" for name in DMC_ROBOTS]
# DMC_ENVS.extend(["{}-Pix-v0".format(name) for name in DMC_ROBOTS])  # no pixels yet


BENCHMARKS = {
    "mujoco": MUJOCO_ENVS,
    "dmc": DMC_ENVS,
}
