import os
import time
import yaml

from cs285.agents.soft_actor_critic import SoftActorCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
import cs285.env_configs

import os
import time

import gym
from gym import wrappers
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
import tqdm

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger

from scripting_utils import make_logger, make_config

import argparse

from collections import OrderedDict
import wandb
import imageio
from pathlib import Path
from time import gmtime


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"]()
    eval_env = config["make_env"]()
    render_env = config["make_env"](render=True)

    ep_len = config["ep_len"] or env.spec.max_episode_steps
    batch_size = config["batch_size"] or batch_size

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert (
        not discrete
    ), "Our actor-critic implementation only supports continuous action spaces. (This isn't a fundamental limitation, just a current implementation decision.)"

    ob_shape = env.observation_space.shape
    ac_dim = env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata["render_fps"]

    # Wandb
    use_wandb = True
    config_name = args.config_file
    config_name = config_name.split("/")[-1]
    if use_wandb:
        now = time.time()
        year, mon, mday, minute = gmtime(now).tm_year, gmtime(now).tm_mon, gmtime(now).tm_mday, gmtime(now).tm_min
        index = f"SAC_{config_name}_{int(mon)}-{int(mday)}-{int(minute)}"
        log_dir = Path('wandb_log').expanduser() / index
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        video_dir = str(log_dir / 'videos')
        Path(video_dir).mkdir(exist_ok=True)
        wandb.init(
            entity="goto-rl",
            project="cs285-hw3",
            resume=index,
            config=vars(args),
            dir=log_dir,
            mode='online'
        )
        wandb.save()

    # initialize agent
    agent = SoftActorCritic(
        ob_shape,
        ac_dim,
        **config["agent_kwargs"],
    )

    replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])

    observation = env.reset()

    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        if step < config["random_steps"]:
            action = env.action_space.sample()
        else:
            # TODO(student): Select an action
            action = agent.get_action(observation= observation)

        # Step the environment and add the data to the replay buffer
        next_observation, reward, done, info = env.step(action)
        replay_buffer.insert(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done and not info.get("TimeLimit.truncated", False),
        )

        if done:
            logger.log_scalar(info["episode"]["r"], "train_return", step)
            logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
            observation = env.reset()
        else:
            observation = next_observation

        # Train the agent
        if step >= config["training_starts"]:
            # TODO(student): Sample a batch of config["batch_size"] transitions from the replay buffer
            batch = replay_buffer.sample(config["batch_size"])
            batch = ptu.from_numpy(batch)
            update_info = agent.update(
                observations= batch["observations"],
                actions= batch["actions"],
                rewards=batch["rewards"],
                next_observations=batch["next_observations"],
                dones= batch["dones"],
                step= step
            )

            # Logging
            update_info["actor_lr"] = agent.actor_lr_scheduler.get_last_lr()[0]
            update_info["critic_lr"] = agent.critic_lr_scheduler.get_last_lr()[0]

            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    logger.log_scalar(v, k, step)
                    logger.log_scalars
                logger.flush()
                if use_wandb:
                    for k, v in update_info.items():
                        wandb.log({k: v}, step=step)

        # Run evaluation
        if step % args.eval_interval == 0:
            trajectories = utils.sample_n_trajectories(
                eval_env,
                policy=agent,
                ntraj=args.num_eval_trajectories,
                max_length=ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

            # Wandb
            if use_wandb:
                wandb_dict = OrderedDict()
                wandb_dict["eval_return"] = np.mean(returns)
                wandb_dict["eval_ep_len"] = np.mean(ep_lens)
                wandb_dict["eval/eval_ep_len_std"] = np.std(ep_lens)
                wandb_dict["eval/eval_ep_len_max"] = np.max(ep_lens)
                wandb_dict["eval/eval_ep_len_min"] = np.min(ep_lens)
                wandb_dict["eval/return_std"] = np.std(returns)
                wandb_dict["eval/return_max"] = np.max(returns)
                wandb_dict["eval/return_min"] = np.min(returns)
                for k, v in wandb_dict.items():
                    wandb.log({f"{k}": v}, step=step)

            if args.num_render_trajectories > 0:
                video_trajectories = utils.sample_n_trajectories(
                    render_env,
                    agent,
                    args.num_render_trajectories,
                    ep_len,
                    render=True,
                )

                if use_wandb:
                    length = len(video_trajectories)
                    for i in range(length):
                        videos = video_trajectories[i]["image_obs"]
                        video_file_name = f'{video_dir}/iteration_{step}_{i + 1}.mp4'
                        imageio.mimsave(video_file_name, videos, fps=15)
                        wandb.log({f"video_{i}": wandb.Video(video_file_name, fps=4, format="mp4")})

                logger.log_paths_as_videos(
                    video_trajectories,
                    step,
                    fps=fps,
                    max_videos_to_save=args.num_render_trajectories,
                    video_title="eval_rollouts",
                )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=5000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)
    parser.add_argument("--log_interval", type=int, default=1000)

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "hw3_sac_"  # keep for autograder

    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
