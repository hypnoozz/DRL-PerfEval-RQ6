# On-Policy vs Off-Policy DRL Algorithm Comparison

This repository evaluates the performance differences between on-policy and off-policy deep reinforcement learning algorithms across different task types.

## Research Question

**Do on-policy and off-policy DRL algorithms perform differently across task types, such as discrete vs. continuous actions, or sparse vs. dense rewards?**

This project analyzes the performance of on-policy algorithms (PPO, A2C) versus off-policy algorithms (DQN, DDPG, SAC) across environments with different reward structures and action spaces.

**Hypothesis:** Off-policy methods will perform better in sparse reward environments (e.g., MountainCarContinuous-v0, BipedalWalker-v3), as their replay buffer enables better utilization of limited feedback. In contrast, on-policy methods may perform better in dense reward environments due to their ability to consistently update policies with fresh and aligned data.

## Environments & Algorithms

### Environments Tested

| Environment | Action Space | Reward Density |
|-------------|--------------|----------------|
| CartPole-v1 | Discrete | Dense |
| LunarLander-v3 | Discrete | Dense |
| MountainCarContinuous-v0 | Continuous | Sparse |
| Pendulum-v1 | Continuous | Dense |
| BipedalWalker-v3 | Continuous | Sparse |

### Algorithms Evaluated

| Algorithm | Type |
|-----------|------|
| PPO | On-Policy |
| A2C | On-Policy |
| DQN | Off-Policy |
| DDPG | Off-Policy |
| SAC | Off-Policy |

## Setup and Usage

### Requirements
```
gymnasium
stable-baselines3
torch
pandas
matplotlib
seaborn
numpy
tqdm
```

### Running Experiments
```
# Run a specific algorithm on an environment
python run_experiment.py --algo DQN --env CartPole-v1

# Run all experiments
python run_experiment.py

# Generate analysis
python analysis.py
```

## Project Structure
```
├── config.py           # Configuration parameters
├── utils.py            # Utility functions for logging and evaluation
├── run_experiment.py   # Main script for running experiments
├── analysis.py         # Data analysis and visualization
├── results/            # Experiment results
└── analysis_figures/   # Generated analysis plots
```

## Key Findings

The analysis focuses on several key metrics:

1. **Performance by Action Space**: How on-policy vs off-policy algorithms perform in discrete vs continuous action spaces
2. **Performance by Reward Density**: Comparison of algorithm types in sparse vs dense reward settings
3. **Algorithm Stability**: Variance analysis across multiple seeds
4. **Learning Curves**: Sample efficiency comparison across different environments

The results help identify which algorithm type (on-policy or off-policy) is better suited for specific environment characteristics, providing practical guidance for reinforcement learning practitioners.

## Acknowledgements

This project uses Stable Baselines 3 implementations of reinforcement learning algorithms and Gymnasium for environments.
