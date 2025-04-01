# DQN-implementation
A basic Deep Q-Learning algorithm implementation in PyTorch

## Overview
This project implements a Deep Q-Network (DQN) agent that can learn to play various Gymnasium environments. It includes both standard feed-forward and CNN model architectures.

## Features
- Standard DQN with Experience Replay
- Target Network for stable learning
- Support for both discrete observation spaces (CartPole) and image-based environments (Atari)
- Configurable hyperparameters
- Training visualization with matplotlib

## Requirements

Python 3.x PyTorch Gymnasium Matplotlib NumPy tqdm
```sh
pip3 install -r requirements.txt
```
## Project Structure
- `DQN.py` - Main DQN agent implementation
- `Models.py` - Neural network architectures (FFN and CNN)
- `Example.py` - CartPole training example
- `Example2.py` - Atari Pong training example

## Usage
To train on CartPole:
```python
python Example.py
```
To train on Pong:
```
python Example2.py (Need to be fixed)
```

## Implementation Details

The DQN implementation includes:

- Double network architecture (policy and target networks)
- Epsilon-greedy exploration strategy
- Experience replay buffer
- Periodic target network updates
- Support for both MLP and CNN architectures