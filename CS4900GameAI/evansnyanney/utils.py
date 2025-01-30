import os
import base64
from pathlib import Path
from IPython.display import HTML
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

def record_video(env_id, model, video_length=500, prefix="", video_folder="videos/"):
    """
    Records a video of the trained agent.

    Args:
        env_id (str): The environment ID.
        model: The trained RL model.
        video_length (int): Number of steps to record.
        prefix (str): Prefix for the video file.
        video_folder (str): Directory to save videos.
    """
    os.makedirs(video_folder, exist_ok=True)

    # Create evaluation environment
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(env_id, render_mode="rgb_array"))])
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder=video_folder,
        record_video_trigger=lambda step: step == 0,  # Record at the beginning
        video_length=video_length,
        name_prefix=prefix,
    )

    # Reset environment
    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = eval_env.step(action)

    eval_env.close()
    print(f"Video recorded and saved to {video_folder}")

def show_videos(video_path="videos", prefix=""):
    """
    Displays the first video that matches the prefix.

    Args:
        video_path (str): Directory where videos are stored.
        prefix (str): Prefix of the video file to display.

    Returns:
        HTML: HTML object to display the video in Jupyter notebooks.
    """
    videos = list(Path(video_path).glob(f"{prefix}*.mp4"))
    if len(videos) > 0:
        video_path = videos[0]
        video_b64 = base64.b64encode(Path(video_path).read_bytes()).decode("utf-8")
        return HTML(
            f"""
            <video width="640" height="480" controls>
                <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
            </video>
            """
        )
    else:
        return "No videos found."
