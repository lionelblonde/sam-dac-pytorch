import argparse

from helpers.misc_util import boolean_flag


def argparser(description="DDPG Experiment"):
    """Create an argparse.ArgumentParser"""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--wandb_project", help="wandb project name", default="DEFAULT")
    parser.add_argument("--env_id", help="environment identifier", default=None)
    parser.add_argument("--seed", help="RNG seed", type=int, default=0)
    parser.add_argument("--uuid", type=str, default=None)
    boolean_flag(parser, "cuda", default=False)
    boolean_flag(parser, "fp16", default=False)
    boolean_flag(parser, "render", hint="render the interaction traces", default=False)
    boolean_flag(parser, "record", hint="record the interaction traces", default=False)
    parser.add_argument("--task", type=str, choices=["train", "evaluate"], default=None)

    # training
    parser.add_argument("--num_timesteps", help="total number of interactions",
                        type=int, default=int(1e7))
    parser.add_argument("--training_steps_per_iter", type=int, default=4)
    parser.add_argument("--eval_steps_per_iter", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=10)

    # model
    boolean_flag(parser, "layer_norm", default=False)

    # optimization
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-4)
    parser.add_argument("--clip_norm", type=float, default=1.)
    parser.add_argument("--wd_scale", help="weight decay scale", type=float, default=0.001)
    parser.add_argument("--acc_grad_steps", type=int, default=8)

    # algorithm
    parser.add_argument("--rollout_len", help="number of interactions per iteration",
                        type=int, default=2)
    parser.add_argument("--batch_size", help="minibatch size", type=int, default=128)
    parser.add_argument("--gamma", help="discount factor", type=float, default=0.99)
    parser.add_argument("--mem_size", type=int, default=int(1e5))
    parser.add_argument("--noise_type", type=str, default="adaptive-param_0.2, ou_0.1, normal_0.1")
    parser.add_argument("--pn_adapt_frequency", type=float, default=50.)
    parser.add_argument("--polyak", type=float, default=0.005, help="soft target nets update")
    parser.add_argument("--targ_up_freq", type=int, default=100, help="hard target nets update")
    boolean_flag(parser, "n_step_returns", default=True)
    parser.add_argument("--lookahead", help="num lookahead steps", type=int, default=10)
    boolean_flag(parser, "ret_norm", default=False)
    boolean_flag(parser, "popart", default=False)

    # TD3
    boolean_flag(parser, "clipped_double", default=True)
    boolean_flag(parser, "targ_actor_smoothing", default=True)
    parser.add_argument("--td3_std", type=float, default=0.2,
                        help="std of smoothing noise applied to the target action")
    parser.add_argument("--td3_c", type=float, default=0.5,
                        help="limit for absolute value of target action smoothing noise")
    parser.add_argument("--actor_update_delay", type=int, default=2,
                        help="number of critic updates to perform per actor update")

    # distributional RL
    boolean_flag(parser, "use_c51", default=False)
    boolean_flag(parser, "use_qr", default=False)
    parser.add_argument("--c51_num_atoms", type=int, default=51)
    parser.add_argument("--c51_vmin", type=float, default=0.)
    parser.add_argument("--c51_vmax", type=float, default=100.)
    parser.add_argument("--num_tau", type=int, default=200)

    # AIL
    parser.add_argument("--expert_path", help="demos location", type=str, default=None)
    parser.add_argument("--num_demos", help="number of expert demo trajs for imitation",
                        type=int, default=None)
    parser.add_argument("--g_steps", type=int, default=3)
    parser.add_argument("--d_steps", type=int, default=1)
    parser.add_argument("--d_lr", type=float, default=1e-5)
    boolean_flag(parser, "state_only", default=True)
    boolean_flag(parser, "minimax_only", default=True)
    parser.add_argument("--ent_reg_scale", type=float, default=0.001)
    boolean_flag(parser, "spectral_norm", hint="whether to use spectral norm", default=False)
    boolean_flag(parser, "grad_pen", hint="whether to use gradient penalty", default=True)
    parser.add_argument("--grad_pen_targ", type=float, default=1.)
    parser.add_argument("--grad_pen_scale", type=float, default=10.)
    boolean_flag(parser, "one_sided_pen", hint="whether to use the one-sided v", default=True)
    boolean_flag(parser, "historical_patching", default=True)
    boolean_flag(parser, "wrap_absorb", default=True)
    boolean_flag(parser, "d_batch_norm", default=False)

    # evaluation
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--checkpoint_suffix", type=str, default=None)
    parser.add_argument("--num_trajs", type=int, default=10)
    return parser
