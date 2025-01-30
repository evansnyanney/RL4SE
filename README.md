# RL4SE: Reinforcement Learning for AIGAME (LunarLander-v3)

<!-- WandB Badge -->
[![WandB Project](https://img.shields.io/badge/Weights%20%26%20Biases-Project-blue?logo=wandb)](https://wandb.ai/evansnyanney-ohio-university/sb3-lunar-lander-Evans%20Nyanney?nw=nwuserevansnyanney)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/evansnyanney/RL4SE/main/LICENSE)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![GitHub Repo size](https://img.shields.io/github/repo-size/evansnyanney/RL4SE)
![GitHub stars](https://img.shields.io/github/stars/evansnyanney/RL4SE?style=social)
![GitHub forks](https://img.shields.io/github/forks/evansnyanney/RL4SE?style=social)

**RL4SE** stands for **Reinforcement Learning for Software Engineering**. This project leverages cutting-edge reinforcement learning algorithms to optimize and enhance software engineering processes. By applying the **Proximal Policy Optimization (PPO)** algorithm to the **LunarLander-v3** environment from OpenAI's Gymnasium, RL4SE demonstrates the practical applications of reinforcement learning in complex, real-world scenarios.

## 📝 Table of Contents

- [📈 Training Demo](#training-demo)
- [🔍 Description](#description)
- [✨ Features](#features)
- [🚀 Installation](#installation)
  - [🔧 Prerequisites](#prerequisites)
  - [📥 Clone the Repository](#clone-the-repository)
  - [🐍 Setup Virtual Environment](#setup-virtual-environment)
  - [📦 Install Dependencies](#install-dependencies)
- [🎮 Usage](#usage)
  - [🔄 Training the PPO Model](#training-the-ppo-model)
  - [📹 Viewing the Training Video](#viewing-the-training-video)
- [📊 Experiment Tracking](#experiment-tracking)
  - [WandB Dashboard](#wandb-dashboard)
  - [Key Performance Metrics](#key-performance-metrics)
- [⚙️ Configuration](#configuration)
- [📚 Dependencies](#dependencies)
- [🤝 Contributing](#contributing)
- [📜 License](#license)
- [🙏 Acknowledgements](#acknowledgements)
- [📬 Contact](#contact)

## 📈 Training Demo

## 📈 Training Demo

<img src="https://github.com/evansnyanney/RL4SE/blob/main/training-demo.gif?raw=true" alt="RL4SE Agent Training Demo" width="1000"/>

*Watch the RL4SE agent successfully land on the lunar surface using the PPO algorithm.*



*Watch the RL4SE agent successfully land on the lunar surface using the PPO algorithm.*


## 📊 Key Performance Metrics

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/evansnyanney/RL4SE/blob/main/performance_metrics/eval1.png?raw=true" alt="Eval 1" width="250"/><br/>
      <sub>**Eval 1:** Initial evaluation metric showing baseline performance.</sub>
    </td>
    <td align="center">
      <img src="https://github.com/evansnyanney/RL4SE/blob/main/performance_metrics/eval2.png?raw=true" alt="Eval 2" width="250"/><br/>
      <sub>**Eval 2:** Secondary evaluation metric indicating progress.</sub>
    </td>
    <td align="center">
      <img src="https://github.com/evansnyanney/RL4SE/blob/main/performance_metrics/rollout2.png?raw=true" alt="Rollout 2" width="250"/><br/>
      <sub>**Rollout 2:** Rollout metrics during training phases.</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/evansnyanney/RL4SE/blob/main/performance_metrics/train_entropy_loss.png?raw=true" alt="Train Entropy Loss" width="250"/><br/>
      <sub>**Train Entropy Loss:** Measures the randomness of the policy over time.</sub>
    </td>
    <td align="center">
      <img src="https://github.com/evansnyanney/RL4SE/blob/main/performance_metrics/train_learning_rate.png?raw=true" alt="Train Learning Rate" width="250"/><br/>
      <sub>**Train Learning Rate:** Adaptive learning rate schedule during training.</sub>
    </td>
    <td align="center">
      <img src="https://github.com/evansnyanney/RL4SE/blob/main/performance_metrics/train_loss.png?raw=true" alt="Train Loss" width="250"/><br/>
      <sub>**Train Loss:** Overall training loss decreasing over epochs.</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/evansnyanney/RL4SE/blob/main/performance_metrics/train_value_loss.png?raw=true" alt="Train Value Loss" width="250"/><br/>
      <sub>**Train Value Loss:** Value loss metric showing accurate value estimation.</sub>
    </td>
    <td align="center">
      <img src="https://github.com/evansnyanney/RL4SE/blob/main/performance_metrics/train_approx_kl.png?raw=true" alt="Train Approx KL" width="250"/><br/>
      <sub>**Train Approx KL:** Approximate KL divergence during training.</sub>
    </td>
    <td align="center">
      <img src="https://github.com/evansnyanney/RL4SE/blob/main/performance_metrics/train_clip_fraction%20.png?raw=true" alt="Train Clip Fraction" width="250"/><br/>
      <sub>**Train Clip Fraction:** Fraction of policy updates that were clipped.</sub>
    </td>
  </tr>
</table>



## 🔍 Description

RL4SE is a **reinforcement learning** project focused on applying the **Proximal Policy Optimization (PPO)** algorithm to the **LunarLander-v3** environment from OpenAI's Gymnasium. This project leverages **Stable Baselines3** for model implementation and **Weights & Biases (WandB)** for experiment tracking and visualization. Additionally, it incorporates **Git Large File Storage (LFS)** to manage large video recordings of agent performance.

**Key Objectives:**

- Demonstrate the effectiveness of PPO in complex environments.
- Track and visualize training metrics using WandB.
- Manage large media files efficiently with Git LFS.
- Provide a modular and scalable codebase for future enhancements.

## ✨ Features

- **Standard PPO Implementation:** Utilizes the PPO algorithm from Stable Baselines3 for training agents.
- **Experiment Tracking:** Integrates with WandB to monitor training progress, visualize metrics, and save code snapshots.
- **Video Recording:** Records and displays videos of the trained agent's performance.
- **Model Saving:** Saves trained models for future use and evaluation.
- **Git LFS Integration:** Manages large video files efficiently using Git Large File Storage.
- **Modular Code Structure:** Organized scripts and utilities for maintainability and scalability.
- **Configuration Flexibility:** Easily adjustable hyperparameters and environment settings.
- **Comprehensive Documentation:** Detailed instructions and explanations for ease of use.

## 🚀 Installation

### 🔧 Prerequisites

Before you begin, ensure you have met the following requirements:

- **Operating System:** Windows, macOS, or Linux
- **Python:** Version 3.7 or higher
- **Git:** Installed on your system
- **Git LFS:** Installed and configured ([Installation Guide](https://git-lfs.github.com/))
- **OpenAI Gymnasium Environment:** Installed as part of the dependencies

### 📥 Clone the Repository

```bash
git clone https://github.com/evansnyanney/RL4SE.git
cd RL4SE
