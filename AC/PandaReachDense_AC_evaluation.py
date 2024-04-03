import os
import time
import numpy as np
import argparse
from distutils.util import strtobool

import gymnasium as gym
import panda_gym

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env


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

from huggingface_hub import HfApi, upload_folder
from huggingface_hub.repocard import metadata_eval_result, metadata_save

from pathlib import Path
import datetime
import tempfile
import json
import shutil
import imageio
import torch
import numpy as np
from wasabi import Printer

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
        
    def act(self, state, mode = "train"):
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


if __name__ == '__main__':
    
    # Set up and initialization
    run_name = 'Eval'
    args = parse_args()
    os.chdir("/home/had/Python works/RL - hugging face/AC/")
    os.makedirs(f"./videos/{args.env_id}/", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    

    #Load model
    actor= actor_NN(0.001, device, std_constraint = 1000)
    actor.model.load_state_dict(torch.load('./models/PandaReachDense-v3/for_hf__1711998531/PandasReach_18K.pt'))

    
    #Create environment
    env = make_vec_env(args.env_id, n_envs=1, seed=args.hp_seed, vec_env_cls =  SubprocVecEnv,
                       env_kwargs={"renderer": "Tiny"})
    envs = VecNormalize(env, norm_obs=False, norm_reward=False, clip_obs=10)

    obs = envs.reset()
    images = [envs.render(mode='rgb_array')]
    states = np.concatenate([obs["observation"], obs["desired_goal"]], axis=-1)
    print("Initial states shape", states.shape)
    
    
    rewards_per_episode = []
    for ep_num in range(10):
        Terms = False
        ep_step = 0
        reward_ep = 0
        while Terms == False:
            ep_step += 1
            actions_tensor, mus, stds = actor.act(states)
            action_numpy = actions_tensor.detach().cpu().numpy()
            obs,  rewards, Terms, truncated  = envs.step(action_numpy)
            next_states = np.concatenate([obs["observation"], obs["desired_goal"]], axis=-1)
            states = next_states
            reward_ep += rewards
            if ep_num < 5:
                img = envs.render(mode='rgb_array')
                images.append(img)
        print(f"Episode {ep_num} finished in {ep_step} steps | rewards = {reward_ep}")
        rewards_per_episode.append(reward_ep)
        
    imageio.mimsave('./evaluation/PandaReachDense-v3/replay.mp4', images, fps=30)
    
    
    
    #Publish to HF Hub
    msg = Printer()
    repo_id = args.repo_id
    hyperparameters=args
    logs=f"runs/{run_name}"
    token = None
    commit_message="Push agent to the Hub"
    
    # Step 1: Clone or create the repo
    repo_url = HfApi().create_repo(
        repo_id=repo_id,
        token=token,
        private=False,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)

        # Step 2: Save the model
        #torch.save(model.state_dict(), tmpdirname / "model.pt")
        model_scripted = torch.jit.script(actor.model) # Export to TorchScript
        model_scripted.save(tmpdirname / "model_scripted.pt") # Save
        
        # Step 3: Evaluate the model and build JSON
        mean_reward, std_reward = np.mean(rewards_per_episode), np.std(rewards_per_episode)

        # First get datetime
        eval_datetime = datetime.datetime.now()
        eval_form_datetime = eval_datetime.isoformat()
        
        
        evaluate_data = {
            "env_id": hyperparameters.env_id,
            "mean_reward": str(mean_reward),
            "std_reward": str(std_reward),
            "n_evaluation_episodes": 10,
            "eval_datetime": eval_form_datetime,
        }

        # Write a JSON file
        with open(tmpdirname / "results.json", "w") as outfile:
            json.dump(evaluate_data, outfile)

        # Step 4: copy video to model card temp folder
        shutil.copyfile('./evaluation/PandaReachDense-v3/replay.mp4', tmpdirname / 'replay.mp4')


        # Step 5: Generate the model card
        generated_model_card, metadata = hug._generate_model_card(
            "Actor-Critic", hyperparameters.env_id, mean_reward, std_reward, hyperparameters
        )
        hug._save_model_card(tmpdirname, generated_model_card, metadata)

        # Step 6: Add logs if needed
        if logs:
            hug._add_logdir(tmpdirname, Path(logs))

        msg.info(f"Pushing repo {repo_id} to the Hugging Face Hub")

        repo_url = upload_folder(
            repo_id=repo_id,
            folder_path=tmpdirname,
            path_in_repo="",
            commit_message=commit_message,
            token=token,
        )

        msg.info(f"Your model is pushed to the Hub. You can view your model here: {repo_url}")

