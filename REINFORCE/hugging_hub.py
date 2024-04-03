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

msg = Printer()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def package_to_hub(
    repo_id,
    model,
    hyperparameters,
    eval_env,
    video_fps=30,
    commit_message="Push agent to the Hub",
    token=None,
    logs=None,
    seed = 1234
):
    """
    Evaluate, Generate a video and Upload a model to Hugging Face Hub.
    This method does the complete pipeline:
    - It evaluates the model
    - It generates the model card
    - It generates a replay video of the agent
    - It pushes everything to the hub
    :param repo_id: id of the model repository from the Hugging Face Hub
    :param model: trained model
    :param eval_env: environment used to evaluate the agent
    :param fps: number of fps for rendering the video
    :param commit_message: commit message
    :param logs: directory on local machine of tensorboard logs you'd like to upload
    """
    msg.info(
        "This function will save, evaluate, generate a video of your agent, "
        "create a model card and push everything to the hub. "
        "It might take up to 1min. \n "
        "This is a work in progress: if you encounter a bug, please open an issue."
    )
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
        model_scripted = torch.jit.script(model) # Export to TorchScript
        model_scripted.save(tmpdirname / "model_scripted.pt") # Save
        
        # Step 3: Evaluate the model and build JSON
        mean_reward, std_reward = _evaluate_agent(eval_env, 10, model, seed=seed)

        # First get datetime
        eval_datetime = datetime.datetime.now()
        eval_form_datetime = eval_datetime.isoformat()

        evaluate_data = {
            "env_id": hyperparameters.env_id,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "n_evaluation_episodes": 10,
            "eval_datetime": eval_form_datetime,
        }

        # Write a JSON file
        with open(tmpdirname / "results.json", "w") as outfile:
            json.dump(evaluate_data, outfile)

        # Step 4: Generate a video
        video_path = tmpdirname / "replay.mp4"
        record_video(eval_env, model, video_path, video_fps, seed=seed)

        # Step 5: Generate the model card
        generated_model_card, metadata = _generate_model_card(
            "SARSA_QLEARNING", hyperparameters.env_id, mean_reward, std_reward, hyperparameters
        )
        _save_model_card(tmpdirname, generated_model_card, metadata)

        # Step 6: Add logs if needed
        if logs:
            _add_logdir(tmpdirname, Path(logs))

        msg.info(f"Pushing repo {repo_id} to the Hugging Face Hub")

        repo_url = upload_folder(
            repo_id=repo_id,
            folder_path=tmpdirname,
            path_in_repo="",
            commit_message=commit_message,
            token=token,
        )

        msg.info(f"Your model is pushed to the Hub. You can view your model here: {repo_url}")
    return repo_url


def _evaluate_agent(env, n_eval_episodes, model, seed = 1234):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param policy: The agent
    """
    


    episode_rewards = []
    for episode in range(n_eval_episodes):
        state = env.reset()
        #state, info = env.reset(seed = seed)
        done = False
        total_rewards_ep = 0

        while done is False:
            #state = torch.Tensor(state).to(device)
            action, _ = model.act(state)
            new_state, reward, done, info = env.step(action[0])
            total_rewards_ep += reward
            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"Result eval:\n    mean reward: {mean_reward:.2f}\n    std: {std_reward:.2f}")

    return mean_reward, std_reward


def record_video(env, model, out_directory, fps=30, seed = 1234):
    images = []
    done = False
    state = env.reset()
    #state, info = env.reset(seed = seed)
    img = env.render(mode='rgb_array')
    images.append(img)
    while not done:
        #state = torch.Tensor(state).to(device)
        action, _ = model.act(state)
        state, reward, done, info = env.step(action[0])
        img = env.render(mode='rgb_array')
        images.append(img)

    imageio.mimsave(out_directory, images, fps=fps)
    #imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)


def _generate_model_card(model_name, env_id, mean_reward, std_reward, hyperparameters):
    """
    Generate the model card for the Hub
    :param model_name: name of the model
    :env_id: name of the environment
    :mean_reward: mean reward of the agent
    :std_reward: standard deviation of the mean reward of the agent
    :hyperparameters: training arguments
    """
    # Step 1: Select the tags
    metadata = generate_metadata(model_name, env_id, mean_reward, std_reward)

    # Transform the hyperparams namespace to string
    converted_dict = vars(hyperparameters)
    converted_str = str(converted_dict)
    converted_str = converted_str.split(", ")
    converted_str = "\n".join(converted_str)

    # Step 2: Generate the model card
    hyperparams = '<br />'.join([name + ": " + str(getattr(hyperparameters, name)) for name in vars(hyperparameters) if 'hp_' in name])
    
    model_card = f"""
  # REINFORCE Agent Playing {env_id}

  This is a trained model of a REINFORCE agent playing {env_id}.

  # Hyperparameters
  {hyperparams}
  """
    return model_card, metadata


def generate_metadata(model_name, env_id, mean_reward, std_reward):
    """
    Define the tags for the model card
    :param model_name: name of the model
    :param env_id: name of the environment
    :mean_reward: mean reward of the agent
    :std_reward: standard deviation of the mean reward of the agent
    """
    metadata = {}
    metadata["tags"] = [
        env_id,
        "ppo",
        "deep-reinforcement-learning",
        "reinforcement-learning",
        "custom-implementation",
        "deep-rl-course",
    ]

    # Add metrics
    eval = metadata_eval_result(
        model_pretty_name=model_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
        dataset_pretty_name=env_id,
        dataset_id=env_id,
    )

    # Merges both dictionaries
    metadata = {**metadata, **eval}

    return metadata


def _save_model_card(local_path, generated_model_card, metadata):
    """Saves a model card for the repository.
    :param local_path: repository directory
    :param generated_model_card: model card generated by _generate_model_card()
    :param metadata: metadata
    """
    readme_path = local_path / "README.md"
    readme = ""
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
            readme = f.read()
    else:
        readme = generated_model_card

    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme)

    # Save our metrics to Readme metadata
    metadata_save(readme_path, metadata)


def _add_logdir(local_path: Path, logdir: Path):
    """Adds a logdir to the repository.
    :param local_path: repository directory
    :param logdir: logdir directory
    """
    if logdir.exists() and logdir.is_dir():
        # Add the logdir to the repository under new dir called logs
        repo_logdir = local_path / "logs"

        # Delete current logs if they exist
        if repo_logdir.exists():
            shutil.rmtree(repo_logdir)

        # Copy logdir into repo logdir
        shutil.copytree(logdir, repo_logdir)