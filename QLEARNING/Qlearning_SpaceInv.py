import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


import QLEARNING.hugging_hub as hug



#%%
env = gym.make("ALE/SpaceInvaders-v5")


print("_____OBSERVATION SPACE_____ \n")
print("Observation Space shape", env.env.observation_space.shape)
print("\n _____ACTION SPACE_____ \n")

rand_act = env.action_space.sample()
print("Action Space Sample", rand_act ,f"({env.get_action_meanings()[rand_act]})")  # Take a random action

action_space = env.action_space.n
print("There are ", action_space, " possible actions")
env.unwrapped.get_action_meanings()


#%%
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default= 'sched-eps-greedy',
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="SpaceInvaders-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=1000,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.01,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.10,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training")
    parser.add_argument("--td-target", type=str, default="expsarsa",
        help="td target calculation method (expsarsa or qlearning)")
    parser.add_argument("--policy", type=str, default="sched-eps-greedy",
        help="policy type (softmax, sched-eps-greedy etc.)")
    parser.add_argument("--softmax-tau", type=float, default=1.,
        help="tau for policy softmax")
    
    # Adding HuggingFace argument
    parser.add_argument("--repo-id", type=str, default="Farbum/Space_invader", help="id of the model repository from the Hugging Face Hub {username/repo_name}")

    args = parser.parse_args()
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                os.makedirs("./videos",exist_ok=True)
                env = gym.wrappers.RecordVideo(env, f"./videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), # resulting feature space 20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), # resulting feature space 9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), # resulting feature space 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), # 3136 = 7*7*64
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.expand(1,-1,-1,-1)
        return self.network(x / 255.0)




def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def softmax_temp(action_values, tau=1.0):
    target_qval = target_network(data.next_observations)  #[bs, action_space_n]
    target_max, _ = target_qval.max(dim=1)
    logit = (target_qval - target_max )/ args.softmax_tau
    action_probs = nn.Softmax(dim=1)(logit)  #[bs, action_space_n]
    return action_probs

if __name__ == "__main__":
    os.chdir("/home/had/Python works/RL - hugging face/QLEARNING/")
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/space_invaders_experiments/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding and deterministic learning
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    rand_generator = np.random.RandomState(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False, #H. modifs -> Changed to False 
    )
    start_time = time.time()
    
    #Record reward progression
    best_reward = 0
    best_length = 0
    
    # Training 
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if args.policy == "softmax":
            tau = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
            with torch.no_grad(): 
                q_values = q_network(torch.Tensor(obs).to(device)) #[bs, action_space_n]
                #q_max, _ = q_values.max(dim=1)
                logit = q_values/ tau  #(q_values - q_max )/ args.softmax_tau
                action_probs = nn.Softmax(dim=1)(logit).cpu().detach().numpy()  #[bs, action_space_n]
                actions = rand_generator.choice(env.action_space.n, p = action_probs.squeeze())
                actions = np.array([actions for _ in range(envs.num_envs)])           

        elif args.policy == "sched-eps-greedy":
            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
            if random.random() < epsilon:
                q_values = q_network(torch.Tensor(obs).to(device)) #only for logging
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                q_values = q_network(torch.Tensor(obs).to(device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # execute the game
        next_obs, rewards, dones, infos = envs.step(actions)


        # log training data
        for info in infos:
            if "episode" in info.keys():
                episodic_return = info['episode']['r']
                episodic_length = info['episode']['l']
                print(f"global_step={global_step}, episodic_return={episodic_return}")
                if episodic_return > best_reward:
                    best_reward = episodic_return
                    model_folder = f"./models/{args.env_id}/{run_name}" 
                    os.makedirs(model_folder, exist_ok = True)
                    torch.save(q_network.state_dict(), model_folder + f"/SpaceInv_st{global_step}_rw{int(episodic_return)}.pt")
                if episodic_length > best_length:
                    best_length = episodic_length
                    model_folder = f"./models/{args.env_id}/{run_name}" 
                    os.makedirs(model_folder, exist_ok = True)
                    torch.save(q_network.state_dict(), model_folder + f"/SpaceInv_st{global_step}_length{int(episodic_length)}.pt")
 
                    
                writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                writer.add_scalar("charts/episodic_length", episodic_length, global_step)
                if "eps" in args.policy:
                    writer.add_scalar("charts/epsilon_tau", epsilon, global_step)
                elif "softmax" in args.policy:
                    writer.add_scalar("charts/epsilon_tau", tau, global_step)
                qval_for_hist = q_values.cpu().detach().numpy()
                writer.add_histogram('hist/q_val', qval_for_hist, global_step)
                break

        # save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)


        obs = next_obs

        # ALGO LOGIC: training. 
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():  #inference mode, reduce memory consumption as gradient calculation is disabled
                    #No gradient is flowing from here onward    
                    if args.td_target == "qlearning":
                        target_max, _ = target_network(data.next_observations).max(dim=1)
                        td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                    
                    elif args.td_target == "expsarsa":
                        
                        if args.policy == "sched_eps_greedy":
                            target_qval = target_network(data.next_observations)  #[bs, action_space_n]
                            target_max, _ = target_qval.max(dim=1)
                            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
                            target_mean = target_max*(1-epsilon) + torch.mean(target_qval,dim=-1)*epsilon
                                                          
                        elif args.policy == "softmax":
                            tau = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
                            target_qval = target_network(data.next_observations)
                            logit = target_qval/ tau 
                            action_probs = nn.Softmax(dim=1)(logit)  #[bs, action_space_n]
                            target_mean  = torch.mean(torch.mul(target_qval, action_probs),axis=-1)
                        
                        td_target = data.rewards.flatten() + args.gamma * target_mean * (1 - data.dones.flatten())

        
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    #print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update the target network   -> it lags behind q_network which is updated every step
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())
                
    envs.close()
    writer.close()
    
    
    


#%%  Evaluate and send to HF

model = QNetwork(envs).to(device)
model.load_state_dict(torch.load('./models/SpaceInvaders-v4/SpaceInvaders-v4__sched-eps-greedy__1__1711935772/SpaceInv_st119604_rw915.pt'))


def make_eval_env(env_id, seed):
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


hug.package_to_hub(
    repo_id=args.repo_id,
    model= model,
    hyperparameters=args,
    eval_env=make_eval_env(args.env_id, args.seed), 
    logs=f"runs/{run_name}"
)