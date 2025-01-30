# RL4SE: Reinforcement Learning for Software Engineering

![RL4SE Banner](https://github.com/evansnyanney/RL4SE/blob/main/images/banner.png?raw=true)

## Description

RL4SE is a reinforcement learning project focused on applying the **Proximal Policy Optimization (PPO)** algorithm to the **LunarLander-v3** environment from OpenAI's Gymnasium. This project leverages **Stable Baselines3** for model implementation and **Weights & Biases (WandB)** for experiment tracking and visualization. Additionally, it incorporates **Git Large File Storage (LFS)** to manage large video recordings of agent performance.

## Features

- **Standard PPO Implementation:** Utilizes the PPO algorithm from Stable Baselines3 for training agents.
- **Experiment Tracking:** Integrates with WandB to monitor training progress, visualize metrics, and save code snapshots.
- **Video Recording:** Records and displays videos of the trained agent's performance.
- **Model Saving:** Saves trained models for future use and evaluation.
- **Git LFS Integration:** Manages large video files efficiently using Git Large File Storage.
- **Modular Code Structure:** Organized scripts and utilities for maintainability and scalability.

## Training Video Demo

![RL4SE Agent Training Demo](https://github.com/evansnyanney/RL4SE/blob/main/images/training-demo.gif?raw=true)

## Installation

### Prerequisites

- **Python 3.7 or higher**
- **Git**
- **Git LFS**
- **OpenAI Gymnasium Environment**

### Clone the Repository

```bash
git clone https://github.com/evansnyanney/RL4SE.git
cd RL4SE
