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
import torch.optim as optim
from torch.distributions import Categorical
#from torch.utils.tensorboard import SummaryWriter

# Tensorboard
#import tensorflow
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

# Gym
import gym
import gym_pygame
import stable_baselines3
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder


# Hugging Face Hub
from huggingface_hub import notebook_login 
import REINFORCE.hugging_hub as hug

#%%
print("Gym version:", gym.__version__)
print("Stable Baselines3 version:", stable_baselines3.__version__)


#%%

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="for_HF_folup",
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
    parser.add_argument("--log-every", type=int, default=2000,
        help="tensorboard, log every X steps")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Pixelcopter-PLE-v0",
        help="the id of the environment")
    parser.add_argument("--hp-nb-frames", type=int, default=8,
        help="Number of frames to return from environment")
    parser.add_argument("--hp-total-timesteps", type=int, default=1005000,
        help="Number of total steps")
    parser.add_argument("--hp-learning-t", type=int, default=500,
        help="timesteps in between learning")
    parser.add_argument("--hp-num-envs", type=int, default=12,
        help="Number of parallel environments")
    parser.add_argument("--hp-learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--hp-gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--hp-buffer-size", type=int, default=10000,
        help="the replay memory buffer size")
    parser.add_argument("--hp-batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    
    # Adding HuggingFace argument
    parser.add_argument("--repo-id", type=str, default="Farbum/Pixelcopter", help="id of the model repository from the Hugging Face Hub {username/repo_name}")

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
     
test_env = Custom_rgb_env("Pixelcopter-PLE-v0", 1234, 3)    
frames = [] 
actions = []
frame1 = test_env.reset()
frames.append(frame1[:,:,-1])
actions.append(999)
for i in range(19):
    action = 1 #test_env.action_space.sample()
    frame, reward, done, info = test_env.step(action)
    frames.append(frame[:,:,-1])
    actions.append(action)

fig, ax = plt.subplots(4,5, figsize = (15,13))
for ix, f in enumerate(frames):
    ax[ix // 5, ix % 5].imshow(f)
    ax[ix // 5, ix % 5].set_title(f"frame {ix}, action {actions[ix]}")


#%% NN model
class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super().__init__()
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
        #print("probs:",probs)
        m = Categorical(probs)
        action = m.sample()
        #print("action.shape:", action.shape)
        return action.numpy(), m.log_prob(action)

class Policy_CNN(nn.Module):
    def __init__(self):  #input = 48x48 for Pixelcopter
        super().__init__()
        
        self.channel_num = args.hp_nb_frames
        
        self.cnn1 = nn.Conv2d(self.channel_num, 16, kernel_size = 6, stride = 2) #22x22x16
        self.cnn2 = nn.Conv2d(16, 32, kernel_size = 4, stride = 2) #10x10x32
        self.cnn3 = nn.Conv2d(32, 64, kernel_size = 3, stride = 2) #4x4x32
        self.fc1 = nn.Linear(1024, 128) 
        self.fc2 = nn.Linear(128, 2) #env.env.action_space.n)
        
    def forward(self, x):
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        x = torch.flatten(x, start_dim=1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        if state.ndim == 3: #1 frame (N, width, height)
            state = torch.from_numpy(np.expand_dims(states,axis=1)).float().to(device)
        
        elif state.ndim == 4: #3 frames (N, width, height, C)
            state = torch.from_numpy(np.moveaxis(states, -1, 1)).float().to(device)
        else:
            state = torch.from_numpy(state).float().to(device)
            
        probs = self.forward(state).cpu()
        #print("probs shape:",probs.shape)
        #print("probs:",probs)
        m = Categorical(probs)
        action = m.sample()
        #print("action.shape:", action.shape)
        return action.numpy(), m.log_prob(action)


class Policy_CNN_LSTM(nn.Module):
    def __init__(self, seq_length=3):  #input = 48x48 for Pixelcopter
        super().__init__()
        
        self.seq_length = seq_length
        
        self.cnn1 = nn.Conv2d(1, 16, kernel_size = 6, stride = 2) #22x22x16
        self.cnn2 = nn.Conv2d(16, 32, kernel_size = 4, stride = 2) #10x10x32
        self.cnn3 = nn.Conv2d(32, 64, kernel_size = 3, stride = 2) #4x4x64
        self.fc1 = nn.Linear(1024, 128) 

        # Add LSTM layer
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        
        self.fc2 = nn.Linear(128, 2) #env.env.action_space.n)
        
    def forward(self, x):
        batch_size, seq_length, C, H, W = x.size()
        
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
        x = self.fc2(x[:, -1, :])
        
        return F.softmax(x, dim=1)

    def act(self, state):
        if state.ndim == 3: #happens for single environment, during eval
            state = np.expand_dims(state, axis = 0)
        assert state.ndim == 4 # seq_len frames (N, H, W, seq_len)
        state = np.expand_dims(state, axis = 1)  #  (N, 1, H, W, seq_len)
        #state_cs = (state - 128) / 255
        state = torch.from_numpy(np.moveaxis(state, -1, 1)).float().to(device) #(N, seq_len, 1, H, W)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.numpy(), m.log_prob(action)




#%%  Tensorboard HP (table of HP) config
def TB_log_hparams(run_name):
    """
    Logs hyperparameters and metrics of an specific experiment to the Hparams tab in Tensorboard.
    """
    
    def format_value(v):
        return v if type(v) in [bool, int, float, str] else str(v)
    arg_value_list = [(name, getattr(args, name)) for name in vars(args) if "hp_" in name]
    hparams = {k:format_value(v) for k,v in arg_value_list}
    hp.hparams(hparams, trial_id = run_name)
    
#%%  Launch training

if __name__ == "__main__":
    os.chdir("/home/had/Python works/RL - hugging face/REINFORCE/")
    args = parse_args()
    run_name = f"{args.exp_name}__{int(time.time())}"
    
    # Set up and initialization
    np.random.seed(args.hp_seed)
    torch.manual_seed(args.hp_seed)
    torch.backends.cudnn.deterministic = args.hp_torch_deterministic #if True, cuDNN will only run deterministic algorithm, always producing the same results, but less performant
    rand_generator = np.random.RandomState(args.hp_seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # env setup
    #env_fns = [lambda: make_env(args.env_id, args.hp_seed + x) for x in range(args.hp_num_envs)]
    env_fns = [lambda: Custom_rgb_env(args.env_id, args.hp_seed + x, args.hp_nb_frames) for x in range(args.hp_num_envs)]     
    
    envs = SubprocVecEnv(env_fns)
    
    # Create the VecVideoRecorder for environment 0
    video_dir = f"./videos/{run_name}/"  # Directory to store the videos
    record_video_trigger = lambda x: x % 50000 == 0  # Record every 5000 steps
    video_length = 500  # Adjust the video length as needed
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
                              [hp.Metric('losses/av_reward_per_agent',display_name = 'av_reward_agent')])
        
    summary_writer = tf.summary.create_file_writer(f"./runs/{args.env_id}/{run_name}")
    with summary_writer.as_default():
        TB_log_hparams(run_name)
    
    #PYTORCH TENSORBOARD WRITER
    #writer = SummaryWriter(f"./runs/{args.env_id}/{run_name}")

    policy = Policy_CNN_LSTM(seq_length = args.hp_nb_frames).to(device)
    policy.load_state_dict(torch.load('./models/Pixelcopter-PLE-v0/for_HF__1711939123/PixelCopter68338_rw920.pt'))
    optimizer = optim.Adam(policy.parameters(), lr=args.hp_learning_rate)


    states = envs.reset()
    states = np.stack(states, axis=0)    
    

    saved_rewards    = [] 
    saved_dones      = [] 
    saved_log_probs  = []
    
    reward_metric = deque(maxlen=500)
    
    #saved_rewards_debug    = []
    #saved_dones_debug      = []
    #saved_log_probs_debug  = []
    
    steps_since_last_update = 0
    total_episodes = 0
    for global_step in range(1, args.hp_total_timesteps+1):

        steps_since_last_update += 1

        actions, log_probs = policy.act(states)
        states, rewards, dones, infos = envs.step(actions)

        saved_rewards.append(rewards)
        saved_dones.append(dones)
        saved_log_probs.append(log_probs)
        
        reward_metric.append(rewards)
        
        #Saving model every N steps, as reward is capped by the episode length
        #The agent will improve even after reward asymptotically reaches max
        if global_step %  50000 == 0: 
            stepk = int(global_step/1000)
            model_folder = f"./models/{args.env_id}/{run_name}" 
            os.makedirs(model_folder, exist_ok = True)
            torch.save(policy.state_dict(), model_folder + f"/PixelCopter_step{stepk}K.pt")

        
        #saved_rewards_debug.append(rewards)  
        #saved_dones_debug.append(dones)  
        #saved_log_probs_debug.append(log_probs)  
        
        
        #if global_step % args.hp_learning_t == 0 and sum(saved_dones) > 0: #TODO: remove the second condition
        if (np.sum(saved_dones) >= args.hp_num_envs + 1) or (steps_since_last_update >= args.hp_learning_t):
            total_episodes += np.sum(saved_dones)
            saved_rewards_np = np.stack(saved_rewards, axis=0)
            saved_dones_np = np.stack(saved_dones, axis=0)
            saved_log_probs_pt = torch.stack([*saved_log_probs], axis=0)
            
            
            #X-env lists needed to compute loss across all enviornments 
            Xenv_loss = []

            
            termination_pos, env_T_ix = np.where(saved_dones_np)
            
            #ENVIRONMENT LOOP
            for env_id in range(args.hp_num_envs):
                env_terminations = termination_pos[env_T_ix == env_id]
                #print("env_terminations:",env_terminations)
                '''
                env_terminations = 
                for env_id0: [2 6]
                for env_id1: [0 5 9]
                '''
                
                env_num_episodes =  len(env_terminations) 
                #print("env_num_episodes:", env_num_episodes)
                '''
                we consider unfinished episodes terminated at current step, this should be
                okay as long as different rewards were given during the first chunk of episode'
                env_num_episodes = 
                for env_id0: 2
                for env_id1: 3
                '''
                dummy = np.concatenate([[-1], env_terminations], axis=0)[:-1]                           
                env_ep_totsteps = env_terminations - dummy
                #print(env_ep_totsteps)
                '''
                env_ep_totsteps = 
                for env_id0: [3 4]
                for env_id1: [1 5 4]
                '''

                #EPISODE LOOP
                env_terminations = np.concatenate(([-1],env_terminations)) #adding -1 so that first timestep 0  is taken into account
                for num_ep in range(env_num_episodes)[::-1]:   
                    
                    #Only move forward if episode has more than 1 step
                    if env_ep_totsteps[num_ep] == 1:
                        continue
                    
                    s_ep = env_terminations[num_ep] + 1
                    e_ep = env_terminations[num_ep+1]

                    
                    env_ep_log_probs = [*saved_log_probs_pt[s_ep:(e_ep+1),env_id]]
                    
                    
                    env_ep_returns = deque(maxlen=int(env_ep_totsteps[num_ep]))
                    t_steps = deque(maxlen=int(env_ep_totsteps[num_ep]))
                    env_av_n_steps = env_ep_totsteps.mean()
                    for t in range(s_ep, e_ep+1)[::-1]:
                        disc_return_t = (env_ep_returns[0] if len(env_ep_returns)>0 else 0)
                        env_ep_returns.appendleft( args.hp_gamma*disc_return_t + saved_rewards_np[t,env_id])    
                        t_steps.appendleft(t)
                        

                    #Calculating return and loss for given ENV episode
                    eps = np.finfo(np.float32).eps.item()       
                    env_ep_returns = torch.tensor(env_ep_returns)
                    env_ep_returns = (env_ep_returns - env_ep_returns.mean()) / (env_ep_returns.std() + eps)
                    env_ep_policy_loss = []
                    for log_prob, disc_return in zip(env_ep_log_probs, env_ep_returns):
                        env_ep_policy_loss.append(-log_prob * disc_return)
                    env_ep_policy_loss = torch.stack(env_ep_policy_loss, axis=0).sum()  #concatenate and sum 
                
                    
                
                    Xenv_loss.append(env_ep_policy_loss)
                    
            # Average loss over Environments AND Env episodes
            Xenv_loss_final = torch.stack(Xenv_loss, axis=0).mean() 
            
            # Gradient descent 
            optimizer.zero_grad()
            Xenv_loss_final.backward()
            optimizer.step()
            
            #Resetting 
            saved_rewards = []
            saved_dones = []
            saved_log_probs = []
            Xenv_loss = []
            steps_since_last_update = 0
        
        if global_step %  args.log_every ==0: #2000 == 0:
            av_reward_per_agent = np.stack(reward_metric, axis=0).sum() / args.hp_num_envs
            print('timestep {}\tepisode {}\t Av Rewards per agent: {:.2f}'.format(global_step, total_episodes, av_reward_per_agent))
            with summary_writer.as_default():
                tf.summary.scalar("losses/av_reward_per_agent", av_reward_per_agent, global_step)
                tf.summary.scalar("losses/policy_loss", Xenv_loss_final.detach().numpy().astype(float), global_step)
            
            #PYTORCH WRITER
            #writer.add_scalar("losses/av_reward_per_agent", av_reward_per_agent, global_step)
            #writer.add_scalar("losses/policy_loss", Xenv_loss_final, global_step)
            
    envs.close()
    
    
    
#%%  Evaluate and send to HF

model = Policy_CNN_LSTM(seq_length = args.hp_nb_frames).to(device)
model.load_state_dict(torch.load('./models/Pixelcopter-PLE-v0/for_HF_folup__1711942382/PixelCopter_step998K.pt'))

hug.package_to_hub(
    repo_id=args.repo_id,
    model= model,
    hyperparameters=args,
    eval_env=Custom_rgb_env(args.env_id, 2, args.hp_nb_frames),
    logs=f"runs/{run_name}"
)