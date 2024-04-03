import os
import time
import numpy as np
import time
from collections import deque
import os
import glob
import argparse
from distutils.util import strtobool

import gymnasium as gym
import panda_gym

import stable_baselines3
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

import matplotlib.pyplot as plt

from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim as optim
import torch.distributions as distributions
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

import AC.hugging_hub as hug

#%%

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="for_hf",
        help="the name of this experiment")
    parser.add_argument("--hp-seed", type=int, default=2444,
        help="seed of the experiment")
    parser.add_argument("--capture-vid", type=bool, default=True,
        help="Wheter to capture video")
    parser.add_argument("--TB_log", type=bool, default=True,
        help="Wheter to log to Tensorboard")
    parser.add_argument("--log-every", type=int, default=100,
        help="tensorboard, log every X steps")
    parser.add_argument("--vid-every", type=int, default=5000,
        help="capture video every X steps")
    parser.add_argument("--vid-length", type=int, default=500,
        help="Length of video")
    parser.add_argument("--hp-torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")

    # Algorithm specific arguments
    parser.add_argument("--debug-mode", type=bool, default=True,
        help="whether to activate debug mode and evaluate each state")
    parser.add_argument("--env-id", type=str, default="PandaReachDense-v3",
        help="the id of the environment")
    parser.add_argument("--hp-total-timesteps", type=int, default=20500,
        help="Number of total steps")
    parser.add_argument("--hp-critic-nstep", type=int, default=1,
        help="n-step TD for critic")
    parser.add_argument("--hp-num-envs", type=int, default=12,
        help="Number of parallel environments")
    parser.add_argument("--hp-learning-rate-actor", type=float, default=1e-3,   #1e-3
        help="the learning rate of the optimizer")
    parser.add_argument("--hp-learning-rate-critic", type=float, default=5e-3,    #5e-3
        help="the learning rate of the critic optimizer")
    parser.add_argument("--hp-minlr-actor", type=float, default=2e-6, 
        help="minimum LR rate for actor")
    parser.add_argument("--hp-minlr-critic", type=float, default=1e-5,
        help="minimum LR rate for critic")
    parser.add_argument("--hp-gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--hp-reg-term", type=float, default=3,
        help="regularization term for action difference")
    parser.add_argument("--hp-batch-size", type=int, default=64,
        help="the batch size of sample from the reply memory")

    # Adding HuggingFace argument
    parser.add_argument("--repo-id", type=str, default="Farbum/PandaReachv3", help="id of the model repository from the Hugging Face Hub {username/repo_name}")

    args = parser.parse_args()
        
    # fmt: on
    return args


#%%  Environment discovery

# if __name__ == '__main__':
#     envs = make_vec_env("PandaReachDense-v3", n_envs=1, seed=1)
#     envs = VecNormalize(envs, norm_obs=False, norm_reward=False, clip_obs=10.)


#     '''
#         achieved_goal: (x,y,z) actual position of end-effector
#         desired_goal: (x,y,z) desired position.
#         observation: position (x,y,z) (aka achieved_goal) and velocity of the end-effector (vx, vy, vz).
#         After 50 actions without finding the ball, T=True and reset
#         If the end-effector goes out of bounds, T=True and reset
#         T = True when the action leads to a Terminal state. 
#         The Terminal state is never shown, T=True and the env resets 
#         Reward is euclidian distance between achieved_goal and desired_goal
        
#         x_start ~ 0
#         y_start ~ 0
#         z_start ~0.2
        
#         x_min, x_max = -1 / 0.21
#         y_min, y_max = -0.8 / 0.8
#         z_min, z_max=   0  / 0.83
        
#         goal stats (sample size: 105)
#         mean_point  = [~0,  0.014,  0.15]
#         xmin,xmax = (-0.145, 0.15)
#         ymin,ymax = (-0.144, 0.15)
#         zmin,zmax = (0, 0.3)
#     '''

#     obs = envs.reset()
    
#     frames = []
#     rewards = []
#     Term = []
#     cur_pos = []
#     cur_vel = []
#     end_goal = []
#     actions = []
#     for _ in range(1000):
#         current_velocity = obs["observation"][:, 3:]  # velocity ?
#         current_position = obs["observation"][:, :3]
#         desired_position = obs["desired_goal"]
#         action = 3.0 * (desired_position - current_position)
#         #action_zy_pos = np.array([0,0,-20]).reshape([1,3])
#         #actions.append(action_zy_pos)   
#         #action = np.array([[0,0,0]])
#         frames.append(envs.render( mode="rgb_array")) 
#         obs,  reward, terminated, truncated  = envs.step(action)
#         rewards.append(reward)
#         Term.append(terminated)
#         cur_pos.append(np.squeeze(current_position))
#         cur_vel.append(np.squeeze(current_velocity))
#         end_goal.append(np.squeeze(desired_position))

#     envs.close()




# # Action distributions
# st = np.stack(Term,axis=0)
# First_episode_step = np.argmax(st)
# num_ep = st.sum()

# sa = np.stack(actions,axis=0)
# sa = np.squeeze(sa)
# fig, ax = plt.subplots(1,3, figsize = (15,5))
# ax[0].hist(sa[:,0], bins=20)
# ax[1].hist(sa[:,1],bins=20)
# ax[2].hist(sa[:,2],bins = 20)
# fig.suptitle(f"x,y,z distribution of actions over {num_ep} successful episodes")

# fig, ax = plt.subplots(1,3, figsize = (15,5))
# ax[0].hist(sa[:First_episode_step,0], bins=20)
# ax[1].hist(sa[:First_episode_step,1],bins=20)
# ax[2].hist(sa[:First_episode_step,2],bins = 20)
# fig.suptitle("x,y,z distribution of actions over 1st episode")




# S = 0  #8 to see end of episode
# E = 4   #12 to see end of episode

# Col = 2
# fig, ax = plt.subplots(2,Col, figsize = (25,20))
# for ix, (f,r,t,cp,cv,eg) in enumerate(zip(frames[S:E], rewards[S:E], Term[S:E], 
#                                           cur_pos[S:E],cur_vel[S:E],end_goal[S:E] )):
#     ax[ix // Col, ix % Col].imshow(f)
#     ax[ix // Col, ix % Col].set_title(f"r{r[0]:.2f} | t{t}\ncpx{cp[0]:.3f}\ncpy{cp[1]:.3f} | cpz{cp[2]:.3f}\
#                                       \nvpx{cv[0]:.3f} | vpy{cv[1]:.3f} | vpz{cv[2]:.3f}\
#                                       \negx{eg[0]:.3f} | egy{eg[1]:.3f} | egz{eg[2]:.3f}")
# fig.suptitle("dones and rewards after taking actions in the image shown", size=25)


#%%   
class AC(nn.Module):
    """
    role: actor or critic
    """
    def __init__(self, role):  #input = 48x48 for Pixelcopter

        super().__init__()
        
        HL = 50
        
        self.fc1 = nn.Linear(9, HL)
        self.fc2 = nn.Linear(HL, HL)
        self.fc2_bis = nn.Linear(HL, HL)
        self.fc3_actor  = nn.Linear(HL, 6) #3 actions X two arguments (μ(s) and σ(s) for a Gaussian)
        self.fc3_critic = nn.Linear(HL, 1) #value function        
        self.role = role
        
        self.fc3_actor.bias.data.fill_(0.0)  # Initialize biases to be near 0
        self.fc3_actor.weight.data.uniform_(-0.05, 0.05)  # Small weights for small outputs

        
        
    def forward(self, x):
        #print("DEBUG x shape", x.shape)
        #print("DEBUG x type", type(x))
        #print("DEBUG x first item", x[0])
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2_bis(x))
        if self.role == 'actor':
            return self.fc3_actor(x)
        else:
            return self.fc3_critic(x)


class actor_NN:
    def __init__(self, lr, device, std_constraint=10):
        self.model = AC(role = 'actor').to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        #self.scheduler = MultiStepLR(self.optimizer, milestones=[50000], gamma = 0.5)
        self.scheduler = CosineAnnealingLR(self.optimizer, args.hp_total_timesteps, 
                                           eta_min=args.hp_minlr_actor, last_epoch=-1)
        self.std_constraint = std_constraint
        
    def act(self, state, global_step, mode = "train"):
        assert state.ndim == 2, f"state tensor expected (N,6), got {state.shape}" 
        state = torch.from_numpy(state).float().to(device)
        if mode == 'eval':
            self.model.eval()
            with torch.no_grad(): 
                action_parameters = self.model(state).cpu()      
        else:            
            self.model.train() 
            action_parameters = self.model(state).cpu()
            
        mus = torch.tanh(action_parameters[:, :3])
        sigmas = torch.nn.functional.softplus(action_parameters[:, 3:]) 
        sigmas =  sigmas / self.std_constraint
        normal_distributions = distributions.Normal(mus, sigmas)
        actions = normal_distributions.sample()
        return actions, mus, sigmas

    
    def update(self, actions, mus, sigmas, TD, I, prev_actions):
        self.model.train()
        normal_distributions = distributions.Normal(mus, sigmas)
        log_probs = normal_distributions.log_prob(actions).sum(dim=1)  #sum across action dimensions
        actor_loss = -torch.mean(I * TD * log_probs) #mean across num environments
        
        Loss_act_diff = args.hp_reg_term * torch.sum(torch.square(actions-prev_actions),axis=1)
        Loss_act_diff = Loss_act_diff.mean()
        
        actor_loss = actor_loss + Loss_act_diff
        self.optimizer.zero_grad()
        actor_loss.backward()
        
        if args.debug_mode and args.TB_log:         
            grads = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grads.append(param.grad.view(-1))
            all_grads = torch.cat(grads)
        else:
            all_grads = None
        
        utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        self.scheduler.step()
        return all_grads, log_probs

class critic_NN:
    def __init__(self, lr, device):
        self.model = AC(role = 'critic').to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        #self.scheduler = MultiStepLR(self.optimizer, milestones=[50000], gamma = 0.5)
        self.scheduler = CosineAnnealingLR(self.optimizer, args.hp_total_timesteps, 
                                           eta_min=args.hp_minlr_critic, last_epoch=-1, verbose=False)
        self.loss_fn = nn.MSELoss()

    def value(self, state, mode = 'eval'):
        assert state.ndim == 2, f"state tensor expected (N,2), got {state.shape}"
        state = torch.from_numpy(state).float().to(device)
        if mode == 'eval':
            self.model.eval()
            with torch.no_grad():
                value =  self.model(state).cpu()
        elif mode == 'train':
            self.model.train()
            value =  self.model(state).cpu()
        return torch.squeeze(value)
    
    def update(self, state, target):
        cur_state_value = self.value(state, mode = 'train')
        loss = self.loss_fn(cur_state_value.flatten(), target) #Flatten to get shape [1] instead of [0]
        
        self.optimizer.zero_grad()
        loss.backward()
        
        if args.debug_mode and args.TB_log:         
            grads = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grads.append(param.grad.view(-1))
            all_grads = torch.cat(grads)
        else:
            all_grads = None
        
        utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        self.scheduler.step()
        
        return cur_state_value.detach(), all_grads   #for TD error, needed for actor update, 
                                          # detach because we do not want to re-use the critic's graph


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
    
    

#%%


if __name__ == '__main__':
    args = parse_args()
    os.chdir("/home/had/Python works/RL - hugging face/AC/")
    os.makedirs(f"./videos/{args.env_id}/", exist_ok=True)
    run_name = f"{args.exp_name}__{int(time.time())}"
    
    # Set up and initialization
    eps = np.finfo(np.float32).eps.item()
    np.random.seed(args.hp_seed)
    torch.manual_seed(args.hp_seed)
    torch.backends.cudnn.deterministic = args.hp_torch_deterministic #if True, cuDNN will only run deterministic algorithm, always producing the same results, but less performant
    torch.autograd.set_detect_anomaly(True)
    rand_generator = np.random.RandomState(args.hp_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    actor = actor_NN(args.hp_learning_rate_actor, device)
    critic = critic_NN(args.hp_learning_rate_critic, device)
    
    
    
    env = make_vec_env(args.env_id, n_envs=args.hp_num_envs, seed=args.hp_seed, vec_env_cls =  SubprocVecEnv,
                       env_kwargs={"renderer": "Tiny"})
    
    #envs = SelectiveRenderWrapper(env, render_idx=0)
    envs = VecNormalize(env, norm_obs=False, norm_reward=False, clip_obs=10)
    
        
    #Wrap the DummyVecEnv with VecVideoRecorder wrapper
    if args.capture_vid:
        video_dir = f"./videos/{args.env_id}/{run_name}/" 
        record_video_trigger = lambda x: x % args.vid_every == True 
        envs = VecVideoRecorder(envs, video_dir, record_video_trigger = record_video_trigger, video_length = args.vid_length)
    
    
    #tensorboard
    if args.TB_log:
        os.makedirs(f"./runs/{args.env_id}/", exist_ok = True)
    
        #Creating parent HP writer only if it does not already exist
        files = glob.glob(f"./runs/{args.env_id}/*")
        if sum(["events.out.tfevents" in f for f in files]) == 0:
            print("Creating parent HP writer")
            parent_HP_writer = tf.summary.create_file_writer(f"./runs/{args.env_id}/")
            with parent_HP_writer.as_default():
                arg_value_list = [(name, getattr(args, name)) for name in vars(args) if "hp_" in name]
                hp.hparams_config([hp.HParam(h, display_name=h.split("hp_")[-1]) for h,v in arg_value_list],
                                  [hp.Metric('losses/rolling_av_rewards',display_name = 'rolling_av_rewards')])
            
        summary_writer = tf.summary.create_file_writer(f"./runs/{args.env_id}/{run_name}")
        with summary_writer.as_default():
            TB_log_hparams(run_name)
    
    
    #TODO: investigate norm_obs, norm_reward, clip_obs etc.
    obs = envs.reset()
    states = np.concatenate([obs["observation"], obs["desired_goal"]], axis=-1)
    print("Initial states shape", states.shape)
    
    reward_metric = deque(maxlen=args.log_every)
    rewards_per_episode = [[] for x in range(args.hp_num_envs)]
    rolling_av_rewards = 0
    
    #DEBUG
    if args.debug_mode:
        saved_states_debug = []
        saved_rewards_debug    = []
        saved_terms_debug      = []
        saved_epsteps_debug    = []
        saved_I_debug  = []
        saved_Gains_debug  = []
        saved_infos_debug = []
        saved_T_debug = []
    
    nsteps_rewards    = deque(maxlen=args.hp_critic_nstep) # stores last n-step rewards
    nsteps_states     = deque(maxlen=args.hp_critic_nstep) # stores last n-step states to update
    nsteps_actions    = deque(maxlen=args.hp_critic_nstep) # stores last n-step actions
    nsteps_terms      = deque(maxlen=args.hp_critic_nstep) # stores last n-step dones
    nsteps_Is         = deque(maxlen=args.hp_critic_nstep)
    nsteps_mus        = deque(maxlen=args.hp_critic_nstep)
    nsteps_stds       = deque(maxlen=args.hp_critic_nstep)
    nsteps_gamma  = np.array([[args.hp_gamma**x,] for x in range(args.hp_critic_nstep)])
    
    ep_steps   = np.zeros(args.hp_num_envs)
    I          = np.ones(args.hp_num_envs)
    
    prev_action = torch.zeros((args.hp_num_envs,3))
    total_episodes = 0
    Count_Ep_termination = 0
    for global_step in range(1, args.hp_total_timesteps+1):
        
        
        
        if global_step %  1000 == 0: 
            stepk = int(global_step/1000)
            model_folder = f"./models/{args.env_id}/{run_name}" 
            os.makedirs(model_folder, exist_ok = True)
            torch.save(actor.model.state_dict(), model_folder + f"/PandasReach_{stepk}K.pt")
        
        
        
        # #Sample action only 
        # if global_step % 3 == 1:
        #     actions_tensor, mus, stds = actor.act(states, global_step)
        actions_tensor, mus, stds = actor.act(states, global_step)
        action_numpy = actions_tensor.detach().cpu().numpy()
        obs,  rewards, Terms, truncated  = envs.step(action_numpy)
        next_states = np.concatenate([obs["observation"], obs["desired_goal"]], axis=-1)
        
        
        #Reward scaling (rewards are naturally very small)
        #rewards = rewards * 10
        
        #Metrics
        for i in range(args.hp_num_envs):
            rewards_per_episode[i].append(envs.get_original_reward()[i])
        
        #N-step TD operations
        nsteps_rewards.append(rewards)
        nsteps_states.append(states)
        nsteps_actions.append(actions_tensor)
        nsteps_terms.append(Terms)
        nsteps_Is.append(I)
        nsteps_mus.append(mus)
        nsteps_stds.append(stds)
        ep_steps += 1
        
        
        #For debbug
        if args.debug_mode:
            saved_states_debug.append(states)
            saved_rewards_debug.append(envs.get_original_reward())  
            saved_terms_debug.append(Terms)  
            saved_epsteps_debug.append(ep_steps)
            saved_I_debug.append(I)
        

        Term_last_n_steps = np.stack(nsteps_terms,axis=0).sum(axis = 0)
        first_T   = np.argmax(np.stack(nsteps_terms,axis=0) == True, axis=0)
        rewards_steps = np.stack(nsteps_rewards, axis=0)
        
        
        up_gains       = np.array([])
        up_terms       = np.array([])
        up_states      = np.zeros([1,states.shape[-1]])
        up_next_states = np.zeros([1,states.shape[-1]])
        up_I           = np.array([])
        up_actions     = []
        up_mus         = []
        up_stds        = []
        
        if global_step >= args.hp_critic_nstep:
            for env_id in range(args.hp_num_envs):
                if ep_steps[env_id] >= args.hp_critic_nstep:
                    env_reward         = rewards_steps[:, env_id]
                    env_gain           = np.sum(env_reward * nsteps_gamma.T)
                    
                    up_gains       = np.concatenate([up_gains, [env_gain]])
                    up_terms       = np.concatenate([up_terms, [nsteps_terms[-1][env_id]]])
                    up_states      = np.concatenate([up_states, [nsteps_states[0][env_id]]], axis=0)
                    up_next_states = np.concatenate([up_next_states, [next_states[env_id]]], axis=0)
                    up_I           = np.concatenate([up_I, [nsteps_Is[0][env_id]]])            
                    up_actions.append(nsteps_actions[0][env_id])
                    up_mus.append(nsteps_mus[0][env_id])
                    up_stds.append(nsteps_stds[0][env_id])
    
    
                if Term_last_n_steps[env_id] > 0 and ep_steps[env_id] < args.hp_critic_nstep: 
                    env_first_T     = first_T[env_id] + 1 #used as a slice hence +1
                    env_rewards     =  rewards_steps[:, env_id][:env_first_T]
                    env_gain        = np.sum(env_rewards * np.squeeze(nsteps_gamma[:env_first_T]))
    
                    up_gains       = np.concatenate([up_gains, [env_gain]])
                    up_terms       = np.concatenate([up_terms, [True]])
                    up_states      = np.concatenate([up_states, [nsteps_states[0][env_id]]], axis=0)
                    up_next_states = np.concatenate([up_next_states, [next_states[env_id]]], axis=0)           
                    up_I           = np.concatenate([up_I, [nsteps_Is[0][env_id]]])                  
                    up_actions.append(nsteps_actions[0][env_id])
                    up_mus.append(nsteps_mus[0][env_id])
                    up_stds.append(nsteps_stds[0][env_id])              
            
            up_actions = torch.stack(up_actions, dim=0)
            up_mus = torch.stack(up_mus, dim=0)
            up_stds = torch.stack(up_stds, dim=0)
            assert up_actions.shape == (args.hp_num_envs, actions_tensor.shape[-1]) #TODO: remove
            assert up_mus.shape == (args.hp_num_envs, actions_tensor.shape[-1])
            assert up_stds.shape == (args.hp_num_envs, actions_tensor.shape[-1])
            
            
            
            if args.debug_mode:
                saved_Gains_debug.append(up_gains)
            
            # Compute metric: av rewards per episode
            # When Terms == True, the last ep reward is recorded and accounted for in the sum
            if np.sum(Terms) >= 1:
                for id_Term in np.where(Terms)[0]:
                    ep_cumR = np.sum(rewards_per_episode[id_Term])
                    Count_Ep_termination += 1
                    rolling_av_rewards += (ep_cumR - rolling_av_rewards) / Count_Ep_termination
                    rewards_per_episode[id_Term] = []
    
    
            # Update step if at least 1 gain has been computed         
            if len(up_gains) >= 1 and global_step >= args.hp_critic_nstep:
                
                #remove the initialized zeros
                up_states = up_states[1:,:]
                up_next_states = up_next_states[1:,:]
                
                assert up_next_states.shape[0] == up_states.shape[0] == len(up_terms)  #TODO: remove
                
    
                # flag_non_terminal -> so that value of next step if Terminal is not computed 
                # in target for critics update, since SubprocVecEnv automatically reset env
                up_flag_nonT = torch.Tensor((up_terms == False) * 1)   # =0 where episode is Terminal           
     
                # update critic
                up_targets = torch.Tensor(up_gains) + args.hp_gamma**(args.hp_critic_nstep) * up_flag_nonT * critic.value(up_next_states, mode = 'eval')
                up_cur_state_val, critic_grads = critic.update(up_states, up_targets)            
                
                #Update actor
                TD  = up_targets - up_cur_state_val
                actor_grads, act_logprob = actor.update(up_actions, up_mus, up_stds, TD, torch.Tensor(up_I), prev_action)
    
            
            #End of loop operations
            I = I*args.hp_gamma   
            I[np.where(Terms == True)] = 1      
            states = next_states.copy()        
            prev_action = actions_tensor.detach()
            ep_steps = ep_steps * (Terms == False)
            
            total_episodes += np.sum(Terms)

        
        if global_step %  args.log_every ==0:
            av_episode_per_agent = total_episodes / args.hp_num_envs
            print('tsp {}\tep {:.1f}\t Av Rewards per episode last {} steps: {:.2f}'.format( global_step, av_episode_per_agent, args.log_every, rolling_av_rewards))
            
            if args.TB_log:
                with summary_writer.as_default():
                    tf.summary.scalar(f"losses/av_reward_per_ep_{args.log_every}_steps", rolling_av_rewards, global_step)
                    tf.summary.scalar("LR/critics_lr", critic.scheduler.get_last_lr()[0], step=global_step)
                    tf.summary.scalar("LR/actor_lr", actor.scheduler.get_last_lr()[0], step=global_step)
                    
                    if args.debug_mode:
                        tf.summary.scalar("Actor/actions_x", action_numpy[:,0][0], step=global_step)
                        tf.summary.scalar("Actor/actions_y", action_numpy[:,1][0], step=global_step)
                        tf.summary.scalar("Actor/actions_z", action_numpy[:,2][0], step=global_step)
                        tf.summary.histogram("Grads/actor", actor_grads.cpu().detach().numpy(), step=global_step)
                        tf.summary.histogram("Grads/critic", critic_grads.cpu().detach().numpy(), step=global_step)
                        tf.summary.scalar("A_values/mus_x", mus.detach().numpy()[:,0][0], step=global_step)
                        tf.summary.scalar("A_values/mus_y", mus.detach().numpy()[:,1][0], step=global_step)
                        tf.summary.scalar("A_values/mus_z", mus.detach().numpy()[:,2][0], step=global_step)
                        tf.summary.scalar("A_values/sigmas_x", stds.detach().numpy()[:,0][0], step=global_step)
                        tf.summary.scalar("A_values/sigmas_y", stds.detach().numpy()[:,1][0], step=global_step)
                        tf.summary.scalar("A_values/sigmas_z", stds.detach().numpy()[:,2][0], step=global_step)
                        tf.summary.scalar("C_values/critics_values", up_cur_state_val[0], step=global_step)
                        tf.summary.scalar("Update/TD", TD[0], step=global_step)
                        tf.summary.scalar("Update/target", up_targets[0], step=global_step)
                        tf.summary.scalar("Update/logprobs", act_logprob.detach().numpy().sum(), step=global_step)  
                        tf.summary.scalar("Update/reward", rewards[0], step=global_step)   
                        
            rolling_av_rewards = 0
            Count_Ep_termination = 0
            
            
    envs.close()



    #%%  DEBUG


# import pandas as pd
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.width', 400)

# #temp = [] #temp = [np.array([-99,-99])]*3
# #temp.extend(saved_Gains_debug)


# IXES = [0,2]
# deb_rewards = list(np.round(np.stack(saved_rewards_debug, axis=0)[:,IXES[0]:IXES[-1]],2))
# deb_terms   = list(np.stack(saved_terms_debug, axis=0)[:,IXES[0]:IXES[-1]])
# deb_I       = list(np.round(np.stack(saved_I_debug, axis=0)[:,IXES[0]:IXES[-1]],2))
# deb_epsteps = list(np.stack(saved_epsteps_debug, axis=0)[:,IXES[0]:IXES[-1]])
# deb_gains   = list(np.round(np.stack(saved_Gains_debug, axis=0)[:,IXES[0]:IXES[-1]],2))

# debug = pd.DataFrame.from_dict({"rewards":deb_rewards,
#                                 "terms":deb_terms,
#                                 "I":deb_I,
#                                 "epsteps":deb_epsteps,
#                                 "Gains": deb_gains})

# debug



#%%  Evaluate and send to HF
actor= actor_NN(0.001, device, std_constraint = 1000)
actor.model.load_state_dict(torch.load('./models/PandaReachDense-v3/for_hf__1711998531/PandasReach_18K.pt'))



hug.package_to_hub(
    repo_id=args.repo_id,
    model= actor,
    hyperparameters=args,
    eval_env=make_vec_env("PandaReachDense-v3", n_envs=1, seed=123),
    logs=f"runs/{run_name}"
)