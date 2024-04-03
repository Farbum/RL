#%%  REQUIREMENTS

# whole sectionprobably not necessary, probably made for colab or notebooks
#!sudo apt-get update
#!sudo apt-get install python-opengl ffmpeg xvfb
# !pip3 install pyvirtualdisplay
# !pip install pyglet==1.5.1


# !pip install gym
# !pip install git+https://github.com/ntasfi/PyGame-Learning-Environment.git  #PyGame Learning Environment
# !pip install git+https://github.com/qlan3/gym-games.git  #This is a gym compatible version of various games for reinforcement learning.
# !pip install huggingface_hub
# !pip install imageio-ffmpeg
# !pip install pyyaml==6.0


#%% Libraries

import numpy as np
import time
from collections import deque
import os
import glob
import argparse

import matplotlib.pyplot as plt
#%matplotlib inline

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim as optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import StepLR
#from torch.utils.tensorboard import SummaryWriter

# Tensorboard
#import tensorflow
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

# Gym
import gym
import gym_pygame
import stable_baselines3

#from gym.wrappers import Monitor
#from gym.vector import AsyncVectorEnv

from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder


# Hugging Face Hub
from huggingface_hub import notebook_login # To log to our Hugging Face account to be able to upload models to the Hub.



#%%

os.chdir("/home/had/Python works/RL - hugging face/AC/")

print("Gym version:", gym.__version__)
print("Stable Baselines3 version:", stable_baselines3.__version__)


#%%

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="AC_C10step_tau10_gc2",
        help="the name of this experiment")
    parser.add_argument("--hp-seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--hp-torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")
    parser.add_argument("--log-every", type=int, default=500,
        help="tensorboard, log every X steps")

    # Algorithm specific arguments
    parser.add_argument("--debug-mode", type=bool, default=True,
        help="whether to activate debug mode and evaluate each state")
    parser.add_argument("--env-id", type=str, default="Pixelcopter-PLE-v0",
        help="the id of the environment")
    parser.add_argument("--hp-nb-frames", type=int, default=1,
        help="Number of frames to return from environment")
    parser.add_argument("--hp-total-timesteps", type=int, default=100000,
        help="Number of total steps")
    parser.add_argument("--hp-critic-nstep", type=int, default=10,
        help="n-step TD for critic")
    parser.add_argument("--hp-tau-min", type=int, default=3,
        help="timesteps in between learning")
    parser.add_argument("--hp-tau-max", type=int, default=10,
        help="timesteps in between learning")
    parser.add_argument("--hp-num-envs", type=int, default=12,
        help="Number of parallel environments")
    parser.add_argument("--hp-learning-rate-actor", type=float, default=5e-5,   #8e-5
        help="the learning rate of the optimizer")
    parser.add_argument("--hp-learning-rate-critic", type=float, default=1e-4,    #1e-4
        help="the learning rate of the critic optimizer")
    parser.add_argument("--hp-gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--hp-buffer-size", type=int, default=640,
        help="the replay memory buffer size")
    parser.add_argument("--hp-batch-size", type=int, default=64,
        help="the batch size of sample from the reply memory")

    args = parser.parse_args()
    # fmt: on
    return args

#%% Pixelcopter-PLE-v0


def make_env(env_id, seed):
    # Create the env
    env = gym.make(env_id)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


class Custom_rgb_env(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self, env_id, seed, frame_stack):
        self.env = gym.make(env_id)
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        
        self.frame_stack = frame_stack
        if self.frame_stack > 1:
            self.frames = deque([], maxlen=frame_stack)
            
    def __getattr__(self, name):
        # If an attribute isn't found in this class, return it from the env (Environment A)
        return getattr(self.env, name)
        
    def reset(self):
        self.env.reset()
        frame_rgb  = self.env.render(mode='rgb_array')
        frame_grey = np.dot(frame_rgb[...,:3], [0.299, 0.587, 0.114]) 
        self.last_frame = frame_rgb
        
        if self.frame_stack > 1:
            self.frames.clear()
            for _ in range(self.frame_stack):
                self.frames.append(frame_grey)
            return self._get_observation()
                
        elif self.frame_stack == 1:  
            return frame_grey

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        frame_rgb  = self.env.render(mode='rgb_array')
        frame_grey = np.dot(frame_rgb[...,:3], [0.299, 0.587, 0.114])  
        self.last_frame = frame_rgb
        
        if self.frame_stack > 1:
            self.frames.append(frame_grey)
            return self._get_observation(), reward, done, info
        
        elif self.frame_stack == 1: 
            return frame_grey, reward, done, info
    
    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.last_frame
        else:
            self.env.render(mode)
            
    def close(self):
        self.env.close()
    
    def seed(self, seed=None):
        self.env.seed(seed)
        
    def _get_observation(self):
        assert len(self.frames) == self.frame_stack
        #print("DEBUG obs:",[f.shape for f in self.frames])
        return np.stack(list(self.frames), axis=-1)
     
# test_env = Custom_rgb_env("Pixelcopter-PLE-v0", 1234, 3)    
# frames = [] 
# actions = []
# frame1 = test_env.reset()
# frames.append(frame1[:,:,-1])
# actions.append(999)
# for i in range(19):
#     action = 1 #test_env.action_space.sample()
#     frame, reward, done, info = test_env.step(action)
#     frames.append(frame[:,:,-1])
#     actions.append(action)

# fig, ax = plt.subplots(4,5, figsize = (15,13))
# for ix, f in enumerate(frames):
#     ax[ix // 5, ix % 5].imshow(f)
#     ax[ix // 5, ix % 5].set_title(f"frame {ix}, action {actions[ix]}")


#%% NN model


class AC_CNN(nn.Module):
    """
    role: actor or critic
    """
    def __init__(self, role):  #input = 48x48 for Pixelcopter

        super(AC_CNN, self).__init__()

        self.cnn1 = nn.Conv2d(1, 16, kernel_size = 6, stride = 2) #22x22x16
        self.cnn2 = nn.Conv2d(16, 32, kernel_size = 4, stride = 2) #10x10x32
        self.cnn3 = nn.Conv2d(32, 64, kernel_size = 3, stride = 2) #4x4x64
        self.fc1 = nn.Linear(1024, 128) 

        self.fc2_actor  = nn.Linear(128, 2) #env.env.action_space.n)
        self.fc2_critic = nn.Linear(128, 1) #value function
        
        self.role = role
        
    def forward(self, x, global_step = 1):
        tau = (args.hp_tau_max - args.hp_tau_min) * (args.hp_total_timesteps - global_step) / args.hp_total_timesteps + args.hp_tau_min
        #print("DEBUG x shape", x.shape)
        #print("DEBUG x type", type(x))
        #print("DEBUG x first item", x[0])
        
        # pass through CNN layers
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        x = torch.flatten(x, start_dim=1) 
        x = F.relu(self.fc1(x))

        # pass through final FC layer (only take final time step)
        if self.role == 'actor':
            x = self.fc2_actor(x)
            return F.softmax(x / tau, dim=1)
        else:
            return self.fc2_critic(x)

class AC_CNN_LSTM(nn.Module):
    """
    role: actor or critic
    """
    def __init__(self, role, seq_length=3):  #input = 48x48 for Pixelcopter

        super(AC_CNN_LSTM, self).__init__()
        
        self.seq_length = seq_length
        
        self.cnn1 = nn.Conv2d(1, 16, kernel_size = 6, stride = 2) #22x22x16
        self.cnn2 = nn.Conv2d(16, 32, kernel_size = 4, stride = 2) #10x10x32
        self.cnn3 = nn.Conv2d(32, 64, kernel_size = 3, stride = 2) #4x4x64
        self.fc1 = nn.Linear(1024, 128) 
        self.bn_fc1 = nn.BatchNorm1d(128)

        # Add LSTM layer
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        
        self.fc2_actor  = nn.Linear(128, 2) #env.env.action_space.n)
        self.fc2_critic = nn.Linear(128, 1) #value function
        
        self.role = role
        
    def forward(self, x, glob_step = 1):
        tau = (args.hp_tau_max - args.hp_tau_min) * (args.hp_total_timesteps - global_step) / args.hp_total_timesteps + args.hp_tau_min

        #print("DEBUG x shape", x.shape)
        #print("DEBUG x type", type(x))
        #print("DEBUG x first item", x[0])
        
        batch_size, seq_length, C, H, W = x.shape

        # reshape input data to (batch_size * seq_length, C, H, W)
        x = x.reshape(-1, C, H, W)
        
        # pass through CNN layers
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        x = torch.flatten(x, start_dim=1) 
        x = F.relu(self.fc1(x))
        
        # reshape back to (batch_size, seq_length, -1)
        x = x.view(batch_size, seq_length, -1)

        # pass through LSTM layer
        x, _ = self.lstm(x)

        # pass through final FC layer (only take final time step)
        x = x[:, -1, :]
        if self.role == 'actor':
            x = self.fc2_actor(x)
            return F.softmax(x / tau, dim=1)
        else:
            return self.fc2_critic(x)


class actor_NN:
    def __init__(self, lr, device):
        self.model = AC_CNN(role = 'actor').to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size = 50000, gamma = 0.32)

    def act(self, state, global_step):
        
        self.model.eval()
        assert state.ndim == (4 if args.hp_nb_frames > 1 else 3), f"state tensor expected (N,H,W,seq_len), got {state.shape}" 
        state = np.expand_dims(state, axis = 1)  #  (N, 1, H, W, seq_len) if hp_nb_frames > 1 else (N, 1, H, W)
        if args.hp_nb_frames > 1:
            state = torch.from_numpy(np.moveaxis(state, -1, 1)).float().to(device) #(N, seq_len, 1, H, W)
        else:
            state = torch.from_numpy(state).float().to(device) #(N, 1, H, W)
        with torch.no_grad():
            probs = self.model.forward(state, global_step).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.numpy(), probs

    
    def update(self, action, state, TD, I):
        self.model.train()
        assert state.ndim == (4 if args.hp_nb_frames > 1 else 3), f"state tensor expected (N,H,W,seq_len), got {state.shape}" 
        state = np.expand_dims(state, axis = 1)  #  (N, 1, H, W, seq_len)
        if args.hp_nb_frames > 1:
            state = torch.from_numpy(np.moveaxis(state, -1, 1)).float().to(device) #(N, seq_len, 1, H, W)
        else:
            state = torch.from_numpy(state).float().to(device) #(N, 1, H, W)
        probs = self.model.forward(state).cpu()
        m = Categorical(probs)
        log_probs = m.log_prob(torch.from_numpy(action))
        
        update_val = args.hp_gamma * I * TD *log_probs
        actor_loss = - update_val.mean()
        self.optimizer.zero_grad()
        actor_loss.backward()
        utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        self.scheduler.step()

class critic_NN:
    def __init__(self, lr, device):
        self.model = AC_CNN(role = 'critic').to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size = 50000, gamma = 0.32)
        self.loss_fn = nn.MSELoss()

    def value(self, state, mode = 'eval'):
        assert state.ndim == (4 if args.hp_nb_frames > 1 else 3), f"state tensor expected (N,H,W,seq_len), got {state.shape}"
        state = np.expand_dims(state, axis = 1)  #  (N, 1, H, W, seq_len)
        if args.hp_nb_frames > 1:
            state = torch.from_numpy(np.moveaxis(state, -1, 1)).float().to(device) #(N, seq_len, 1, H, W)
        else:
            state = torch.from_numpy(state).float().to(device) #(N, 1, H, W)
        if mode == 'eval':
            self.model.eval()
            with torch.no_grad():
                value =  self.model.forward(state).cpu()   #.item() ?
        elif mode == 'train':
            self.model.train()
            value =  self.model.forward(state).cpu()
        return torch.squeeze(value)
    
    def update(self, state, target):
        self.model.train()
        cur_state_value = self.value(state, mode = 'train')
        loss = self.loss_fn(cur_state_value, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        self.scheduler.step()
        
        return cur_state_value.detach()   #for TD error, needed for actor update, 
                                          # detach because we do not want to re-use the critic's graph



#%%   REPLAY BUFFER

class ReplayBuffer:
    '''
    state: state to update
    action: action taken in state
    reward: n-step gain
    done: Has episode terminated in the n-steps following state
    next_state: next_state after n-step
    I: I for actor update
    debug_val: critic's value estimation of state
    '''
    def __init__(self, capacity, rng):
        self.capacity = capacity
        self.buffer = deque([], maxlen=self.capacity)
        self.rng = rng
        
    def push(self, state, action, reward, done, next_state, I, debug_val, debug_envid):
        if type(state) == list:            
            for s,a,r,d, ns,i,dv,did in zip(state, action, reward, done, next_state, I, debug_val,debug_envid):
                self.buffer.append((s,a,r,d, ns,i, dv, did))             
            
        else:
            self.buffer.append((state, action, reward, done, next_state, I))

    def sample(self, batch_size):
        indices = self.rng.choice(self.__len__(), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        state, action, reward, done, next_state, I, dv, _ = map(np.stack, zip(*batch))

        return state, action, reward, done, next_state, I, dv
    
    def debug_last_obs(self, nb, env_id):
        #returns last nb observations for env_id
        batch = [self.buffer[-i] for i in reversed(range(1, nb*(args.hp_num_envs+1)))]
        state, action, reward, done, next_state, I, dv, did = map(np.stack, zip(*batch))
        ixes_env = np.where(did == env_id)
        return state[ixes_env], action[ixes_env], reward[ixes_env], done[ixes_env], next_state[ixes_env], I[ixes_env], dv[ixes_env]

    def __len__(self):
        return len(self.buffer)
    
#rng = np.random.RandomState(1234)
#a = ReplayBuffer(100, rng)
#for i in range(10):
#    t = str(i)
#    a.push(["s"+t]*12, ["a"+t]*12, ["r"+t]*12, ["ns"+t]*12,  ['i'+t]*12, [torch.Tensor([5])]*12,)
#state, action, reward, next_state, I, log_prob = a.sample(5)
#b = a.sample(9)


#%%  Tensorboard HP (table of HP) config
def TB_log_hparams(run_name):
    """ I[np.where(dones == True)] = 1   
    Logs hyperparameters and metrics of an specific experiment to the Hparams tab in Tensorboard.
    """
    
    def format_value(v):
        return v if type(v) in [bool, int, float, str] else str(v)
    arg_value_list = [(name, getattr(args, name)) for name in vars(args) if "hp_" in name]
    hparams = {k:format_value(v) for k,v in arg_value_list}
    hp.hparams(hparams, trial_id = run_name)
    
#%%  Launch training

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}__{int(time.time())}"
    
    # Set up and initialization
    eps = np.finfo(np.float32).eps.item()
    np.random.seed(args.hp_seed)
    torch.manual_seed(args.hp_seed)
    torch.backends.cudnn.deterministic = args.hp_torch_deterministic #if True, cuDNN will only run deterministic algorithm, always producing the same results, but less performant
    torch.autograd.set_detect_anomaly(True)
    rand_generator = np.random.RandomState(args.hp_seed)
    
    #RBuff = ReplayBuffer(args.hp_buffer_size, rand_generator)
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # env setup
    #env_fns = [lambda: make_env(args.env_id, args.hp_seed + x) for x in range(args.hp_num_envs)]
    env_fns = [lambda: Custom_rgb_env(args.env_id, args.hp_seed + x, args.hp_nb_frames) for x in range(args.hp_num_envs)]     
    
    envs = SubprocVecEnv(env_fns)
    
    # Create the VecVideoRecorder for environment 0
    video_dir = f"./videos/{run_name}/"  # Directory to store the videos
    record_video_trigger = lambda x: x in [0,50000,100000,200000,300000,400000,499000] # Record every 5000 steps
    video_length = 1000  # Adjust the video length as needed
    envs = VecVideoRecorder(envs, video_dir, 
                            record_video_trigger, video_length)

    

    #tensorboard
    os.makedirs(f"./runs/{args.env_id}/", exist_ok = True)
    
    #Creating parent HP writer only if it does not already exist
    files = glob.glob(f"./runs/{args.env_id}/*")
    if sum(["events.out.tfevents" in f for f in files]) == 0:
        print("Creating parent HP writer")
        parent_HP_writer = tf.summary.create_file_writer(f"./runs/{args.env_id}/")
        with parent_HP_writer.as_default():
            arg_value_list = [(name, getattr(args, name)) for name in vars(args) if "hp_" in name]
            hp.hparams_config([hp.HParam(h, display_name=h.split("hp_")[-1]) for h,v in arg_value_list],
                              [hp.Metric('losses/av_reward_per_agent_500ts',display_name = 'av_reward_agent')])
        
    summary_writer = tf.summary.create_file_writer(f"./runs/{args.env_id}/{run_name}")
    with summary_writer.as_default():
        TB_log_hparams(run_name)
    
    #PYTORCH TENSORBOARD WRITER
    #writer = SummaryWriter(f"./runs/{args.env_id}/{run_name}")

    actor = actor_NN(args.hp_learning_rate_actor, device)
    critic = critic_NN(args.hp_learning_rate_critic, device)
    
    states = envs.reset()
    states = np.stack(states, axis=0)    
    print("Initial states shape", states.shape)
    
    reward_metric = deque(maxlen=500)
    
    
    saved_rewards_debug    = []
    saved_dones_debug      = []
    saved_epsteps_debug    = []
    saved_I_debug  = []
    saved_Gains_debug  = []
    saved_infos_debug = []
    saved_T_debug = []
    
    
    total_episodes = 0

    nsteps_rewards    = deque(maxlen=args.hp_critic_nstep) # stores last n-step rewards
    nsteps_states     = deque(maxlen=args.hp_critic_nstep) # stores last n-step states to update
    nsteps_actions    = deque(maxlen=args.hp_critic_nstep) # stores last n-step actions
    nsteps_dones      = deque(maxlen=args.hp_critic_nstep) # stores last n-step dones
    nsteps_Is         = deque(maxlen=args.hp_critic_nstep)
    nsteps_Term       = deque(maxlen=args.hp_critic_nstep) # stores last n-step terminal (True if state was Terminal)
    nsteps_gamma  = np.array([[args.hp_gamma**x,] for x in range(args.hp_critic_nstep)])
    
    ep_steps   = np.zeros(args.hp_num_envs)
    I          = np.ones(args.hp_num_envs)
    
    
    for global_step in range(1, args.hp_total_timesteps+1):
        
        actions, probs = actor.act(states, global_step)
        next_states, rewards, dones, infos = envs.step(actions)
        

        reward_metric.append(rewards.copy()) #copy to measure against the original reward
        
        #Modify rewards
        rewards[np.logical_or(rewards == 0,rewards == 1)] = 0.1
        rewards[np.logical_or(rewards == -5,rewards == -4)] = -5
        
        #N-step TD operations
        state_terminal = nsteps_dones[-1] == True if global_step > 1 else np.array([False]*args.hp_num_envs)
        nsteps_Term.append(state_terminal) #True if state is Terminal
        nsteps_rewards.append(rewards)
        nsteps_states.append(states)
        nsteps_actions.append(actions)
        nsteps_dones.append(dones)
        nsteps_Is.append(I)



        ep_steps += 1
        
        
        #For debbug
        saved_rewards_debug.append(rewards)  
        saved_dones_debug.append(dones)  
        saved_epsteps_debug.append(ep_steps)
        saved_I_debug.append(I)
        saved_infos_debug.append(infos)
        saved_T_debug.append(nsteps_Term[-1])
        

        

        Term_last_n_steps = np.stack(nsteps_dones,axis=0).sum(axis = 0)
        first_T   = np.argmax(np.stack(nsteps_dones,axis=0) == True, axis=0)
        rewards_steps = np.stack(nsteps_rewards, axis=0)
        
        
        up_gains       = np.array([])
        up_dones       = np.array([])
        up_states      = np.zeros([1,48,48])
        up_next_states = np.zeros([1,48,48])
        up_I           = np.array([])
        up_actions     = np.array([])

        push = False        
        for env_id in range(args.hp_num_envs):
            if ep_steps[env_id] >= args.hp_critic_nstep and not(state_terminal[env_id]):
                env_reward         = rewards_steps[:, env_id]
                env_gain           = [np.sum(env_reward * nsteps_gamma.T)]
                env_dones       = [nsteps_dones[-1][env_id]]      
                env_next_states = [next_states[env_id]] #next_states after n-step
                env_states      = [nsteps_states[0][env_id]] #states to update
                env_actions     = [nsteps_actions[0][env_id]] #actions taken in states to update
                env_I          = [nsteps_Is[0][env_id]]
                
                #RBuff.push(env_states, env_actions, env_gain, env_dones,
                #            env_next_states,  env_I, [env_id], [env_id]) 
                up_gains       = np.concatenate([up_gains, env_gain])
                up_dones       = np.concatenate([up_dones, env_dones])
                up_states      = np.concatenate([up_states, env_states])
                up_next_states = np.concatenate([up_next_states, env_next_states])
                up_actions     = np.concatenate([up_actions, env_actions])
                up_I           = np.concatenate([up_I, env_I])                
                        
                
                
                #debug
                # if push == True:
                #     if env_id == 0:
                #         saved_Gains_debug[-1] = np.array([env_gain[0],saved_Gains_debug[-1]])
                #     else:
                #         saved_Gains_debug[-1] = np.array([saved_Gains_debug[-1], env_gain[0]])
                # else:
                #     saved_Gains_debug.append(env_gain[0])
                # push = True

                    

            if Term_last_n_steps[env_id] > 0 and ep_steps[env_id] < args.hp_critic_nstep and not(state_terminal[env_id]):   

                env_first_T  = first_T[env_id] + 1 #used as a slice hence +1
                env_rewards  =  rewards_steps[:, env_id][:env_first_T]
                env_gain     = [np.sum(env_rewards * np.squeeze(nsteps_gamma[:env_first_T]))]
                env_dones = [True]
                env_next_states = [next_states[env_id]] #next_states after n-step
                env_states      = [nsteps_states[0][env_id]] #states to update
                env_actions     = [nsteps_actions[0][env_id]] #actions taken in states to update
                env_I          = [nsteps_Is[0][env_id]]

                # RBuff.push(env_states, env_actions, env_gain, env_dones,
                #             env_next_states,  env_I, [env_id], [env_id])  
                up_gains       = np.concatenate([up_gains, env_gain])
                up_dones       = np.concatenate([up_dones, env_dones])
                up_states      = np.concatenate([up_states, env_states])
                up_next_states = np.concatenate([up_next_states, env_next_states])
                up_actions     = np.concatenate([up_actions, env_actions])                
                up_I           = np.concatenate([up_I, env_I])                 
                
                
                #debug
                # if push == True:
                #     if env_id == 0:
                #         saved_Gains_debug[-1] = np.array([env_gain[0],saved_Gains_debug[-1]])
                #     else:
                #         saved_Gains_debug[-1] = np.array([saved_Gains_debug[-1], env_gain[0]])
                # else:
                #     saved_Gains_debug.append(env_gain[0])
                # push = True

                
            # Reset episode step counter if state is Terminal
            if state_terminal[env_id]:
                ep_steps[env_id] = 0
                
                #debug
                # if push == True:
                #     if env_id == 0:
                #         saved_Gains_debug[-1] = np.array([-99,saved_Gains_debug[-1]])
                #     else:
                #         saved_Gains_debug[-1] = np.array([saved_Gains_debug[-1], -99])
                # else:
                #     saved_Gains_debug.append(-99)
                # push = True

                
        if len(up_gains) >= 1 and global_step >= args.hp_critic_nstep:
            
            #remove the initialized zeros
            up_states = up_states[1:,:,:]
            up_next_states = up_next_states[1:,:,:]
            
            assert up_next_states.shape[0] == up_states.shape[0] == len(up_dones)
            
            up_gains = torch.Tensor(up_gains)
            up_I = torch.Tensor(up_I)
            
            # flag_non_terminal -> so that value of next step if Terminal is not computed 
            # in target for critics update, since SubprocVecEnv automatically reset env
            b_flag_nonT = torch.Tensor((up_dones == False) * 1)   # =0 where episode is Terminal           
 
            # #update critic
            up_targets = torch.Tensor(up_gains) + args.hp_gamma**(args.hp_critic_nstep) * b_flag_nonT * critic.value(up_next_states, mode = 'eval')
            up_cur_state_val = critic.update(up_states, up_targets)            
            
            #Update actor
            TD  = up_targets - up_cur_state_val
            actor.update(up_actions, up_states, TD, up_I)

        # #Take update step        
        # if RBuff.__len__() >= args.hp_batch_size:
            
            
        #     b_states, b_actions, b_gains, b_dones, b_nstep_states, b_I, _  = RBuff.sample(args.hp_batch_size)
            
        #     b_gains = torch.Tensor(b_gains)
        #     b_I = torch.Tensor(b_I)
            
            #normalize rewards
            #b_gains = (b_gains - b_gains.mean()) / (b_gains.std() + eps)
            
            
            # flag_non_terminal -> so that value of next step if Terminal is not computed 
            # in target for critics update, since SubprocVecEnv automatically reset env
            # b_flag_nonT = torch.Tensor((b_dones == False) * 1)   # =0 where episode is Terminal
            
            # #update critic
            # b_targets = torch.Tensor(b_gains) + args.hp_gamma**(args.hp_critic_nstep) * b_flag_nonT * critic.value(b_nstep_states, mode = 'eval')
            # b_cur_state_val = critic.update(b_states, b_targets)            
            
            # #Update actor
            # TD  = b_targets - b_cur_state_val
            # actor.update(b_actions, b_states, TD, b_I)
        
        
        # else:
        #     b_cur_state_val = torch.from_numpy(-99 * np.ones((args.hp_batch_size)))

        
        #End of loop operations
        I = I*args.hp_gamma   
        I[np.where(dones == True)] = 1      
        states = next_states.copy()                
        ep_steps = ep_steps * (dones == False)

        
        if global_step %  args.log_every ==0: #2000 == 0:
            av_reward_per_agent = np.stack(reward_metric, axis=0).sum() / args.hp_num_envs
            print('timestep {}\tepisode {}\t Av Rewards per agent last 500 steps: {:.2f}'.format(global_step, total_episodes, av_reward_per_agent))
            with summary_writer.as_default():
                tf.summary.histogram("A_values/action_probs", probs, step=global_step)
                tf.summary.scalar("losses/av_reward_per_agent_500ts", av_reward_per_agent, global_step)
        #         tf.summary.histogram("C_values/critics_values", b_cur_state_val, step=global_step)
                tf.summary.scalar("LR/critics_lr", critic.scheduler.get_last_lr()[0], step=global_step)
                tf.summary.scalar("LR/actor_lr", actor.scheduler.get_last_lr()[0], step=global_step)
                

            
    envs.close()
    
    
#%%  DEBUG


# import pandas as pd
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.width', 400)

# temp = [np.array([-99,-99])]*3
# temp.extend(saved_Gains_debug)
# debug = pd.DataFrame.from_dict({"rewards":saved_rewards_debug,
#                                 "dones":saved_dones_debug,
#                                 "I":saved_I_debug,
#                                 "epsteps":saved_epsteps_debug,
#                                 "Gains": temp})

# debug


# import sys
# np.set_printoptions(threshold=sys.maxsize)

# TS = 20

# b_states, b_actions, b_gains, b_dones, b_nstep_states, _, b_v = RBuff.debug_last_obs(TS, env_id = 5)


# b_gains = torch.Tensor(b_gains)
# b_I = torch.Tensor(b_I)

# # flag_non_terminal -> so that value of next step if Terminal is not computed 
# # in target for critics update, since SubprocVecEnv automatically reset env
# b_flag_nonT = torch.Tensor((b_dones == False) * 1)   # =0 where episode is Terminal

# #update critic
# b_targets = torch.Tensor(b_gains) + args.hp_gamma**(args.hp_critic_nstep) * b_flag_nonT * critic.value(b_nstep_states, mode = 'eval')
# b_cur_state_val = critic.value(b_states)          

# #Update actor
# TD  = b_targets - b_cur_state_val

# fig, ax = plt.subplots(TS // 5,5, figsize = (TS,15))
# it = 0
# for i in range(TS):
#     s = b_states[i]
#     v = round(b_cur_state_val[i].item(),2)
#     #d = b_dones[i]
#     a = b_actions[i]
#     g = round(b_gains[i].item(),2)
#     td = round(TD[i].item(),2)
#     ax[it // 5, it % 5].imshow(s)
#     ax[it // 5, it % 5].set_title(f"f{i} | v{v} | a{a} | g{g} | TD{td}")
#     it += 1
    