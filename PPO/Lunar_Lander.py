import argparse
import os
import random
import time
from distutils.util import strtobool
import glob

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.distributions.categorical import Categorical

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

import PPO.hugging_hub as hug


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default='Async_FT', #os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--hp-seed", type=int, default=5678,
        help="seed of the experiment")
    parser.add_argument("--TB_log", type=bool, default=True,
        help="Wheter to log to Tensorboard")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="LunarLander-v2",
        help="the id of the environment")
    parser.add_argument("--hp-total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--hp-learning-rate", type=float, default=1e-3,
        help="the learning rate of the optimizer")
    parser.add_argument("--hp-min-learning-rate", type=float, default=1.5e-6, 
        help="minimum LR rate")
    parser.add_argument("--hp-num-envs", type=int, default=12,
        help="the number of parallel game environments")
    parser.add_argument("--hp-num-steps", type=int, default=256,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--hp-anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--hp-gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--hp-gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--hp-gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--hp-minibatch-size", type=int, default=256,
        help="Size of mini-batch")
    parser.add_argument("--hp-update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--hp-norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--hp-clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--hp-clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--hp-ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--hp-vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--hp-max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--hp-target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    # Adding HuggingFace argument
    parser.add_argument("--repo-id", type=str, default="Farbum/Lunar_lander", help="id of the model repository from the Hugging Face Hub {username/repo_name}")

    args = parser.parse_args()
    args.batch_size = int(args.hp_num_envs * args.hp_num_steps)
    args.num_minibatches = int(args.batch_size // args.hp_minibatch_size)
    assert args.batch_size % args.hp_minibatch_size == 0, f"Choose hp_num_steps so that bsize = ({args.batch_size}) is a multiple of minibatch size (args.hp_minibatch_size)"
    return args

#%%

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")

        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}",
                                               episode_trigger=lambda x: x % int(50*4/args.hp_num_envs) == 0)      
        env.action_space.seed(args.hp_seed)
        env.observation_space.seed(args.hp_seed)
        return env

    return thunk

#%%
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(int(np.array(envs.single_observation_space.shape).prod()), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(int(np.array(envs.single_observation_space.shape).prod()), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, int(envs.single_action_space.n)), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

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
    os.chdir("/home/had/Python works/RL - hugging face/PPO/")
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.hp_seed}__{int(time.time())}"
        
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
                                  [hp.Metric('losses/policy_loss',display_name = 'policy_loss')])
            
        summary_writer = tf.summary.create_file_writer(f"./runs/{args.env_id}/{run_name}")
        with summary_writer.as_default():
            TB_log_hparams(run_name)
            
    
    #Record reward progression
    rolling_ep_reward = 0
    rolling_ep_count = 0
    rolling_ep_length = 0
    best_reward = 0
    
    # Seeding
    random.seed(args.hp_seed)
    np.random.seed(args.hp_seed)
    torch.manual_seed(args.hp_seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.hp_num_envs)]
    # )
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.hp_num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.hp_learning_rate, eps=1e-5)
    scheduler = CosineAnnealingLR(optimizer, args.hp_total_timesteps, 
                                           eta_min=args.hp_min_learning_rate, last_epoch=-1)
 
    
    
    # ALGO Logic: Storage setup
    obs = torch.zeros((args.hp_num_steps, args.hp_num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.hp_num_steps, args.hp_num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.hp_num_steps, args.hp_num_envs)).to(device)
    rewards = torch.zeros((args.hp_num_steps, args.hp_num_envs)).to(device)
    dones = torch.zeros((args.hp_num_steps, args.hp_num_envs)).to(device)
    values = torch.zeros((args.hp_num_steps, args.hp_num_envs)).to(device)

    # start the game
    global_step = 0
    start_time = time.time()
    next_obs, next_info = envs.reset(seed = args.hp_seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.hp_num_envs).to(device)
    num_updates = args.hp_total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.hp_anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.hp_learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.hp_num_steps):
            global_step += 1 * args.hp_num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # execute the game and log data.
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = terminated + truncated
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            
            
            # Capturing and logging average episode return/length
            if len(info) > 0:
                ep_stats  = [x['episode'] for x in info['final_info'][info['_final_info']]]
                ep_cum_reward, ep_length = [x['r'] for x in ep_stats], [x['l'] for x in ep_stats]
                rolling_ep_reward += np.sum(ep_cum_reward)
                rolling_ep_length += np.sum(ep_length)
                rolling_ep_count += len(ep_cum_reward)               
                with summary_writer.as_default():
                    tf.summary.scalar("charts/episodic_return", np.mean(ep_cum_reward), global_step)
                    tf.summary.scalar("charts/episodic_length", np.mean(ep_length), global_step)
                    

            if global_step % 1000 == 0 and rolling_ep_count > 0:
                print(f"------------------\nglobal_step={global_step}\
                      \n   -episodic_return = {round(rolling_ep_reward/rolling_ep_count,1)}\
                      \n   -episode_length = {round(rolling_ep_length/rolling_ep_count,1)}\n")
                      
                # Save if best model
                if rolling_ep_reward > best_reward:
                    best_reward = rolling_ep_reward
                    model_folder = f"./models/{args.env_id}/{run_name}" 
                    os.makedirs(model_folder, exist_ok = True)
                    torch.save(agent.state_dict(), model_folder + f"/Lulander_st{global_step}_rw{int(rolling_ep_reward)}.pt")
                    #model_scripted = torch.jit.script(agent) # Export to TorchScript
                    #model_scripted.save(model_folder + f"/Lulander_ep{rolling_ep_count}_rw{int(rolling_ep_reward)}.pt") # Save

                rolling_ep_reward = 0
                rolling_ep_count = 0
                rolling_ep_length = 0
                

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.hp_gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.hp_num_steps)):
                    if t == args.hp_num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.hp_gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.hp_gamma * args.hp_gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.hp_num_steps)):
                    if t == args.hp_num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.hp_gamma * nextnonterminal * next_return
                advantages = returns - values
        
        
        
        
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.hp_update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.hp_minibatch_size):
                end = start + args.hp_minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.hp_clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.hp_norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.hp_clip_coef, 1 + args.hp_clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.hp_clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.hp_clip_coef,
                        args.hp_clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.hp_ent_coef * entropy_loss + v_loss * args.hp_vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.hp_max_grad_norm)
                optimizer.step()
                scheduler.step()

            if args.hp_target_kl is not None:
                if approx_kl > args.hp_target_kl:
                    break
   
                                                                                            
   
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # log training info
        with summary_writer.as_default():
            tf.summary.scalar("charts/learning_rate", scheduler.get_last_lr()[0], global_step)
            tf.summary.scalar("losses/value_loss", v_loss.item(), global_step)
            tf.summary.scalar("losses/policy_loss", pg_loss.item(), global_step)
            tf.summary.scalar("losses/entropy", entropy_loss.item(), global_step)
            tf.summary.scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            tf.summary.scalar("losses/approx_kl", approx_kl.item(), global_step)
            tf.summary.scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            tf.summary.scalar("losses/explained_variance", explained_var, global_step)
            tf.summary.scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            

    envs.close()



#%%

#Create the evaluation environment

model =  Agent(envs).to(device)
model.load_state_dict(torch.load('./models/LunarLander-v2/LunarLander-v2__Async_FT__5678__1701844137/Lulander_st405000_rw1929.pt'))


eval_env = gym.make(args.env_id)

hug.package_to_hub(
    repo_id=args.repo_id,
    model= model,
    hyperparameters=args,
    eval_env=gym.make(args.env_id, render_mode="rgb_array"),
    logs=f"runs/{run_name}",
    seed = 2
)