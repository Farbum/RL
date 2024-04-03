#%% Libraries

import numpy as np
import time
from collections import deque
from itertools import chain
import os

import argparse

import matplotlib.pyplot as plt
%matplotlib inline

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


# Gym
import gym
import gym_pygame
from gym.wrappers import Monitor

# Hugging Face Hub
from huggingface_hub import notebook_login 
import imageio



#%%
env = gym.make("Pixelcopter-PLE-v0")


print("_____OBSERVATION SPACE_____")
print("Observation Space shape", env.env.observation_space.shape)
print("\n _____ACTION SPACE_____")
print("number of possible actions", env.env.action_space.n)

rand_act = env.action_space.sample()
print("\nAction Space Sample", rand_act)  # Take a random action


#%%

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="simple_env_baseline",
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
    parser.add_argument("--log-every", type=int, default=10,
        help="tensorboard, log every X episode")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Pixelcopter-PLE-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="Number of total steps")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
        help="the learning rate of the optimizer")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--buffer-size", type=int, default=100000,
        help="the replay memory buffer size")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")

    args = parser.parse_args()
    # fmt: on
    return args

#%% Pixelcopter-PLE-v0


def make_env(env_id, seed, capture_video, run_name, envsync_id = 0):
    # Create the env
    env = gym.make(env_id)
    
    if capture_video:
        if envsync_id == 0:
            os.makedirs(f"./videos/{run_name}",exist_ok=True)
            env = gym.wrappers.RecordVideo(env = env, video_folder = f"./videos/{run_name}",
                                           episode_trigger = lambda x: x % 2000 == 0)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

#%% testing environment and seeds  -> seed does make the starting position deterministic
args = parse_args()
test_env = make_env("Pixelcopter-PLE-v0", 2, False, 'test')


observation = test_env.reset()

# Render the environment to obtain the frame
frame = test_env.render(mode='rgb_array')
# Perform an action (e.g., choose a random action)
action = 0 #test_env.action_space.sample()

# Step through the environment
observation, reward, done, info = test_env.step(action)

frame = test_env.render(mode='rgb_array')
plt.imshow(frame)


#%% NN model
class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size * 2)
        self.fc3 = nn.Linear(h_size * 2, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        if state.ndim == 1:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        else:
            state = torch.from_numpy(state).float().to(device)
        probs = self.forward(state).cpu()
        #print("probs shape:",probs.shape)
        m = Categorical(probs)
        action = m.sample()
        #print("action.shape:", action.shape)
        return action.numpy(), m.log_prob(action)

    
#%%

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}__{int(time.time())}"
    
    # Set up and initialization
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic #if True, cuDNN will only run deterministic algorithm, always producing the same results, but less performant
    rand_generator = np.random.RandomState(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # env setup
    num_envs = 1
    envs = gym.vector.AsyncVectorEnv([lambda: make_env(args.env_id, args.seed + x, args.capture_video, run_name, x) for x in range(num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    
    #tensorboard
    os.makedirs(f"./runs/{args.env_id}/", exist_ok = True)
    writer = SummaryWriter(f"./runs/{args.env_id}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    

    policy = Policy(s_size = 7, a_size = 2, h_size = 64).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)


    states = envs.reset()
    states = np.stack(states, axis=0)    
    
    
    #debug_saved_rewards = []
    #debug_saved_log_probs = []
    #debug_save_dones = []
    

    saved_rewards    = []
    saved_log_probs  = []
    scores_deque = deque(maxlen=10)  #used to report scores, averaged over maxlen
    scores = []
    reward_metric = deque(maxlen=500)
    
    i_episode = 0
    for global_step in range(1, args.total_timesteps+1):

        actions, log_probs = policy.act(states)
        states, rewards, dones, infos = envs.step(actions)
        
        
        saved_log_probs.append(log_probs[0])
        saved_rewards.append(rewards[0])
        
        reward_metric.append(rewards[0])
        
        
        #debug_saved_rewards.append(rewards[0])
        #debug_saved_log_probs.append(log_probs[0])
        #debug_save_dones.append(dones[0])
        
        if dones[0]:
            i_episode += 1
            
            scores_deque.append(sum(saved_rewards))
            scores.append(sum(saved_rewards))
            
            
            returns = deque(maxlen=len(saved_rewards)) 
            n_steps = len(saved_rewards) 


            for t in range(n_steps)[::-1]:
                disc_return_t = (returns[0] if len(returns)>0 else 0)
                returns.appendleft( args.gamma*disc_return_t + saved_rewards[t]   )    
                
            ## standardization of returns
            eps = np.finfo(np.float32).eps.item()
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)
            
            policy_loss = []
            for log_prob, disc_return in zip(saved_log_probs, returns):
                policy_loss.append(-log_prob * disc_return)
            policy_loss = torch.stack(policy_loss, axis=0).sum()  #concatenate and sum 

            
            #model update
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            #Resetting        
            saved_rewards    = []
            saved_log_probs  = []
            scores = []

        if global_step % 2000 == 0:
            av_reward_per_agent = sum(reward_metric)
            print('timestep {}\tepisode {}\t Av Rewards per agent: {:.2f}'.format(global_step, i_episode, av_reward_per_agent))
            writer.add_scalar("losses/av_reward_per_agent", av_reward_per_agent, global_step)
            writer.add_scalar("losses/policy_loss", policy_loss, global_step)

