#%%  REQUIREMENTS

# whole sectionprobably not necessary, probably made for colab or notebooks
#!sudo apt-get update
#!sudo apt-get install python-opengl ffmpeg xvfb
!pip3 install pyvirtualdisplay
!pip install pyglet==1.5.1


!pip install gym
!pip install git+https://github.com/ntasfi/PyGame-Learning-Environment.git  #PyGame Learning Environment
!pip install git+https://github.com/qlan3/gym-games.git  #This is a gym compatible version of various games for reinforcement learning.
!pip install huggingface_hub
!pip install imageio-ffmpeg
!pip install pyyaml==6.0

#%%  Run the virtual Screen -> same probably not necessary

from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

#%% Libraries

import numpy as np
import time
from collections import deque
import os

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

# Hugging Face Hub
from huggingface_hub import notebook_login # To log to our Hugging Face account to be able to upload models to the Hub.
import imageio



#%%
env = gym.make("CartPole-v1")


print("_____OBSERVATION SPACE_____ \n")
print("Observation Space shape", env.env.observation_space.shape)
print("\n _____ACTION SPACE_____ \n")

rand_act = env.action_space.sample()
print("Action Space Sample", rand_act)  # Take a random action

action_space = env.action_space.n
print("There are ", action_space, " possible actions")


#%% Cart Pole v1    -> https://www.gymlibrary.dev/environments/classic_control/cart_pole/


def make_env(capture_video, run_name, env_id = "CartPole-v1"):
    # Create the env
    env = gym.make(env_id)
    
    # Create the evaluation env
    eval_env = gym.make(env_id)
    
    if capture_video:
        os.makedirs("./videos",exist_ok=True)
        env = gym.wrappers.RecordVideo(env = env, video_folder = f"./videos/{run_name}",
                                       episode_trigger = lambda x: x % 100 == 0)
    
    return env, eval_env


#%% NN model
class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs) #created distribution to sample from
        action = m.sample()
        return action.item(), m.log_prob(action)  #log_prob calculates Ln of the policy for selected action 
                                                  #which is Ln of the probability of taking the selected action     


#%%
def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, writer, log_every = 10):
    
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    # Line 3 of pseudocode
    for i_episode in range(1, n_training_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break 
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        
        #Had comments: seems like the update is done after the episode ends. 
        #TODO: rewrite logic to fully take adventage of online learning.
        
        # Line 6 of pseudocode: calculate the return
        returns = deque(maxlen=max_t) 
        n_steps = len(rewards) 
        # Compute the discounted returns at each timestep,
        # G_t = r_(t+1) + gamma*G_(t+1)
        # G_(t-1) = r_t + gamma* G_t

        ## We compute this starting from the last timestep to the first, in order

        ## Hence, the queue "returns" will hold the returns in chronological order, from t=0 to t=n_steps
        ## thanks to the appendleft() function which allows to append to the position 0 in constant time O(1)
        ## a normal python list would instead require O(N) to do this.
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft( gamma*disc_return_t + rewards[t]   )    
            
        ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()
        ## eps is the smallest representable float, which is 
        # added to the standard deviation of the returns to avoid numerical instabilities        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        
        # Line 7:
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()  #concatenate and sum
        
        ## Hadrien comment: I do not see gamma^t * disc_return * log_prob as suggested by Sutton p328
        #TODO: try to add gamma
        
        # Line 8: PyTorch prefers gradient descent 
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if i_episode % log_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            writer.add_scalar("losses/returns", returns.sum(), i_episode)
            writer.add_scalar("losses/policy_loss", policy_loss, i_episode)
            writer.add_scalar("episode_steps/steps", n_steps, i_episode)

    return scores


#%%
cartpole_hyperparameters = {
    "h_size": 16,
    "n_training_episodes": 1000,
    "n_evaluation_episodes": 10,
    "max_t": 1000,
    "gamma": 1.0,
    "lr": 1e-2,
    "env_id": "CartPole-v1",
    "state_space": s_size,
    "action_space": a_size,
}

# Create policy and place it to the device
cartpole_policy = Policy(cartpole_hyperparameters["state_space"], cartpole_hyperparameters["action_space"], cartpole_hyperparameters["h_size"]).to(device)
cartpole_optimizer = optim.Adam(cartpole_policy.parameters(), lr=cartpole_hyperparameters["lr"])

#tensorboard
os.makedirs("./runs/cartpole/", exist_ok=True)
run_name = f"cartpole__{int(time.time())}"


env, eval_env = make_env(capture_video=True, run_name=run_name)

writer = SummaryWriter(f"./runs/cartpole/{run_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join(
        [f"|{key}|{value}|" for key, value in cartpole_hyperparameters.items()])),
)

scores = reinforce(cartpole_policy,
                   cartpole_optimizer,
                   cartpole_hyperparameters["n_training_episodes"], 
                   cartpole_hyperparameters["max_t"],
                   cartpole_hyperparameters["gamma"], 
                   writer,
                   10)

#%%
def evaluate_agent(env, max_steps, n_eval_episodes, policy):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param policy: The Reinforce agent
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0
        
        for step in range(max_steps):
            action, _ = policy.act(state)
            new_state, reward, done, info = env.step(action)
            total_rewards_ep += reward
            
            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    return mean_reward, std_reward


mean_reward, std_reward = evaluate_agent(eval_env, 400, 50, cartpole_policy)
print(f"mean_reward:{mean_reward}\nstd_reward:{std_reward}")





#%%    PIXEL COPTER

def make_env(capture_video, run_name, env_id = "Pixelcopter-PLE-v0",video_every = 2500):
    # Create the env
    env = gym.make(env_id)
    
    # Create the evaluation env
    eval_env = gym.make(env_id)
    
    if capture_video:
        os.makedirs("./videos",exist_ok=True)
        env = gym.wrappers.RecordVideo(env = env, video_folder = f"./videos/{run_name}",
                                       episode_trigger = lambda x: x % video_every == 0)
    
    return env, eval_env

# Get the state space and action space  #https://pygame-learning-environment.readthedocs.io/en/latest/user/games/pixelcopter.html
test_env, _ = make_env(capture_video=False, run_name = 'test')
s_size = test_env.observation_space.shape[0]
a_size = test_env.action_space.n

print("_____OBSERVATION SPACE_____ \n")
print("The State Space is: ", s_size)
print("Sample observation", test_env.observation_space.sample()) # Get a random observation

print("\n _____ACTION SPACE_____ \n")
print("The Action Space is: ", a_size)
print("Action Space Sample", test_env.action_space.sample()) # Take a random action


#%%
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
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
    
#%%
pixelcopter_hyperparameters = {
    "h_size": 64,
    "n_training_episodes": 50000,
    "n_evaluation_episodes": 10,
    "max_t": 10000,
    "gamma": 0.99,
    "lr": 1e-4,
    "env_id": "Pixelcopter-PLE-v0",
    "state_space": s_size,
    "action_space": a_size,
}

# Create policy and place it to the device
pixelcopter_policy = Policy(
    pixelcopter_hyperparameters["state_space"],
    pixelcopter_hyperparameters["action_space"],
    pixelcopter_hyperparameters["h_size"],
).to(device)
pixelcopter_optimizer = optim.Adam(pixelcopter_policy.parameters(), lr=pixelcopter_hyperparameters["lr"])

#tensorboard
os.makedirs("./runs/pixel_copter/", exist_ok = True)
run_name = f"pixel_copter__{int(time.time())}"


env, eval_env = make_env(capture_video = True, run_name = run_name, video_every = 2500)

writer = SummaryWriter(f"./runs/pixel_copter/{run_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in pixelcopter_hyperparameters.items()])),
)


scores = reinforce(
    pixelcopter_policy,
    pixelcopter_optimizer,
    pixelcopter_hyperparameters["n_training_episodes"],
    pixelcopter_hyperparameters["max_t"],
    pixelcopter_hyperparameters["gamma"],
    writer,
    100
)

