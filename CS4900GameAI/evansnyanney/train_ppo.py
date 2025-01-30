import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback
import Box2D

from utils import record_video, show_videos

def main():
    # Configuration
    config = {
        "policy_type": "MlpPolicy",  # Standard MLP policy
        "total_timesteps": 500_000,
        "env_name": "LunarLander-v3",
        "project_name": "sb3-ppo-lunar-lander-evans-nyanney",
    }

    # Initialize WandB
    wandb.init(
        project=config["project_name"],
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("videos", exist_ok=True)

    # Create the environment
    env = DummyVecEnv([lambda: Monitor(gym.make(config["env_name"], render_mode="rgb_array"))])

    # Initialize the PPO model
    model = PPO(
        config["policy_type"],
        env,
        verbose=1,
        tensorboard_log=f"logs/{wandb.run.id}",
    )

    # Define evaluation callback
    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"models/{wandb.run.id}",
        log_path=f"logs/{wandb.run.id}",
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    # Start training
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[WandbCallback(), eval_callback],
    )

    # Save the trained model
    model_path = f"models/lunar_lander_ppo_{wandb.run.id}"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Record a video of the trained agent
    record_video(
        env_id=config["env_name"],
        model=model,
        video_length=1000,
        prefix="ppo-lunarlander",
        video_folder="videos/",
    )

    # Display the recorded video
    video_html = show_videos(video_path="videos", prefix="ppo-lunarlander")
    display(video_html)

    wandb.finish()

if __name__ == "__main__":
    main()
