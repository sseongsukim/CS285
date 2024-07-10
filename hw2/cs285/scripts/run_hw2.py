from cs285.agents.pg_agent import PGAgent
import os
import time
from time import gmtime

import gym
import numpy as np
import imageio
import torch

from pathlib import Path
import tqdm

import wandb

from cs285.infrastructure import pytorch_util as ptu

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.action_noise_wrapper import ActionNoiseWrapper


MAX_NVIDEO = 2
def run_training_loop(args):
    logger = Logger(args.logdir)

    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = gym.make(args.env_name, render_mode=None)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # add action noise, if needed
    if args.action_noise_std > 0:
        assert not discrete, f"Cannot use --action_noise_std for discrete environment {args.env_name}"
        env = ActionNoiseWrapper(env, args.seed, args.action_noise_std)

    max_ep_len = args.ep_len or env.spec.max_episode_steps

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if hasattr(env, "model"):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata["render_fps"]

    # Wandb
    use_wandb = True
    if use_wandb:
        now = time.time()
        year, mon, mday, minute = gmtime(now).tm_year, gmtime(now).tm_mon, gmtime(now).tm_mday, gmtime(now).tm_min
        sc = gmtime(now).tm_sec
        index = f"{args.env_name}_{int(mon)}-{int(mday)}-{int(minute)}_{int(sc)}_use_reward_to_go_{args.use_reward_to_go}_normalize_advantages_{args.normalize_advantages}_batch_size_{args.batch_size}_baseline_{args.use_baseline}_{args.gae_lambda}"
        log_dir = Path('wandb_log').expanduser() / index
        Path(log_dir).mkdir(parents= True, exist_ok=True)
        video_dir = str(log_dir / 'videos')
        Path(video_dir).mkdir(exist_ok=True)
        wandb.init(
            entity="goto-rl",
            project="hw2",
            resume= index,
            config= vars(args),
            dir= log_dir,
            mode= 'online'
        )
        wandb.save()

    # initialize agent
    agent = PGAgent(
        ob_dim,
        ac_dim,
        discrete,
        n_layers=args.n_layers,
        layer_size=args.layer_size,
        gamma=args.discount,
        learning_rate=args.learning_rate,
        use_baseline=args.use_baseline,
        use_reward_to_go=args.use_reward_to_go,
        normalize_advantages=args.normalize_advantages,
        baseline_learning_rate=args.baseline_learning_rate,
        baseline_gradient_steps=args.baseline_gradient_steps,
        gae_lambda=args.gae_lambda,
    )

    total_envsteps = 0
    start_time = time.time()

    for itr in range(args.n_iter + 1):

        print(f"\n********** Iteration {itr} ************")
        # TODO: sample `args.batch_size` transitions using utils.sample_trajectories
        # make sure to use `max_ep_len`
        trajs, envsteps_this_batch = utils.sample_trajectories(
            env= env,
            policy= agent.actor,
            min_timesteps_per_batch= args.batch_size,
            max_length= max_ep_len,
        )
        total_envsteps += envsteps_this_batch

        # trajs should be a list of dictionaries of NumPy arrays, where each dictionary corresponds to a trajectory.
        # this line converts this into a single dictionary of lists of NumPy arrays.
        trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}

        # TODO: train the agent using the sampled trajectories and the agent's update function
        train_info = agent.update(
            obs= trajs_dict["observation"],
            actions= trajs_dict["action"],
            rewards= trajs_dict["reward"],
            terminals= trajs_dict["terminal"],
        )

        if itr % args.scalar_log_freq == 0:
            # save eval metrics
            print("\nCollecting data for eval...")
            eval_trajs, eval_envsteps_this_batch = utils.sample_trajectories(
                env, agent.actor, args.eval_batch_size, max_ep_len
            )

            logs = utils.compute_metrics(trajs, eval_trajs)
            # compute additional metrics
            logs.update(train_info)
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs[
                    "Train_AverageReturn"
                ]

            # perform the logging
            for key, value in logs.items():
                print("{} : {}".format(key, value))
                logger.log_scalar(value, key, itr)
                if use_wandb:
                    wandb.log({f'{key}': value}, step= itr)
            print("Done logging...\n\n")

            logger.flush()

        if args.video_log_freq != -1 and itr % args.video_log_freq == 0:
            print("\nCollecting video rollouts...")
            eval_video_trajs = utils.sample_n_trajectories(
                env, agent.actor, MAX_NVIDEO, max_ep_len, render=True
            )
            if use_wandb:
                num_of_videos = len(eval_video_trajs)
                for i in range(num_of_videos):
                    video = eval_video_trajs[i]["image_obs"]
                    video_file_name = f'{video_dir}/iteration_{itr}_{i + 1}.mp4'
                    imageio.mimsave(video_file_name, video, fps=30)
                    wandb.log({f"video_{i}": wandb.Video(video_file_name, fps=4, format= "mp4")})

            logger.log_trajs_as_videos(
                eval_video_trajs,
                itr,
                fps=fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="eval_rollouts",
            )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=False, default= "CartPole-v0")
    parser.add_argument("--exp_name", type=str, required=False, default= "cartpole")
    parser.add_argument("--n_iter", "-n", type=int, default=100)

    parser.add_argument("--use_reward_to_go", "-rtg", action="store_true")
    parser.add_argument("--use_baseline", action="store_true")
    parser.add_argument("--baseline_learning_rate", "-blr", type=float, default=5e-3)
    parser.add_argument("--baseline_gradient_steps", "-bgs", type=int, default=5)
    parser.add_argument("--gae_lambda", type=float, default=None)
    parser.add_argument("--normalize_advantages", "-na", action="store_true")
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1000
    )  # steps collected per train iteration
    parser.add_argument(
        "--eval_batch_size", "-eb", type=int, default=400
    )  # steps collected per eval iteration

    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--layer_size", "-s", type=int, default=64)

    parser.add_argument(
        "--ep_len", type=int
    )  # students shouldn't change this away from env's default
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default= 10)
    parser.add_argument("--scalar_log_freq", type=int, default=1)
    parser.add_argument("--action_noise_std", type=float, default=0)

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "q2_pg_"  # keep for autograder

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data")

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = (
        logdir_prefix
        + args.exp_name
        + "_"
        + args.env_name
        + "_"
        + time.strftime("%d-%m-%Y_%H-%M-%S")
    )
    logdir = os.path.join(data_path, logdir)
    args.logdir = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    run_training_loop(args)


if __name__ == "__main__":
    main()
