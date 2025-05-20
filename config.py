import gymnasium as gym
from typing import Dict, List, Tuple
import torch

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 随机种子配置
SEEDS: List[int] = [1, 2, 3, 4, 5]

# 算法配置
ALGORITHMS = ['DQN', 'PPO', 'A2C', 'SAC', 'DDPG']

# 环境配置
ENVIRONMENTS: List[str] = [
    'CartPole-v1',
    'Pendulum-v1',
    'MountainCarContinuous-v0',
    'LunarLander-v3',
    'BipedalWalker-v3'
]

# 训练配置
USE_UNIFIED_TRAINING_STEPS = True  # 强制使用统一步数
UNIFIED_TRAINING_STEPS = 200_000

# 环境特定训练步数配置
ENV_TRAINING_STEPS = {
    'CartPole-v1': 200_000,
    'Pendulum-v1': 200_000,
    'MountainCarContinuous-v0': 200_000,
    'LunarLander-v3': 200_000,
    'BipedalWalker-v3': 200_000,
}

# 通用网络配置
NETWORK_CONFIG = {
    'hidden_sizes': [256, 256],    # 隐藏层大小
    'activation': 'ReLU',          # 激活函数
    'learning_rate': 3e-4,         # 学习率
}

# 算法特定配置
ALGO_SPECIFIC_CONFIG = {
    "DQN": {
        'buffer_size': 100_000,
        'learning_starts': 1000,
        'train_freq': 4,
        'batch_size': 32,
        'gamma': 0.99,
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.05,
        'exploration_fraction': 0.1,
        'target_update_interval': 1000,
        'tau': 1.0,
        'learning_rate': 1e-3,
    },
    'PPO': {
        'clip_range': 0.2,
        'gae_lambda': 0.95,
        'ent_coef': 0.01,
        'n_epochs': 10,
    },
    'A2C': {
        'gae_lambda': 0.95,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
    },
    'SAC': {
        'buffer_size': 1_000_000,
        'tau': 0.005,
        'ent_coef': 'auto',
    },
    'DDPG': {
        'buffer_size': 1_000_000,
        'tau': 0.005,
    },
}

# 日志配置
LOG_CONFIG = {
    'log_dir': 'logs',
    'model_dir': 'models',
    'results_dir': 'results',  # 结果将保存在 results/算法/环境/ 目录下
}

# 实验结果保存的指标
METRICS = [
    'episode_reward',
    'episode_length',
    'success_rate',
    'training_time',
    'average_value_loss',
    'average_policy_loss',
    'raw_rewards',  # 原始奖励
    'smoothed_rewards',  # 平滑后的奖励
    'final_reward_distribution',  # 最终奖励分布
    'early_phase_rewards',  # 早期阶段奖励
]

UNIFIED_TRAINING_CONFIG = {
    'total_timesteps': 200_000,      # 固定为200k步
    'num_seeds': 5,                  # 5个随机种子
    'gamma': 0.99,                   # 折扣因子
    'learning_rate': 3e-4,           # 学习率
    'optimizer': 'Adam',             # 优化器
    'batch_size': 64,                # batch大小
    'eval_frequency': 10_000,        # 每1万步评估一次
    'reward_normalization': True,    # 启用奖励归一化
    'env_wrapper': 'GymMonitor+CustomLogger',
    'early_stopping': False,         # 不使用早停(固定步数)
    'early_phase_steps': 50_000,     # 早期阶段步数
    'smoothing_window': 100,         # 奖励平滑窗口
}

def get_env_config(env_id: str) -> Dict:
    """获取特定环境的配置"""
    base_config = {
        'CartPole-v1': {
            'success_threshold': 195.0,
            'total_timesteps': ENV_TRAINING_STEPS['CartPole-v1'],
        },
        'Pendulum-v1': {
            'success_threshold': -200.0,
            'total_timesteps': ENV_TRAINING_STEPS['Pendulum-v1'],
        },
        'MountainCarContinuous-v0': {
            'success_threshold': -110.0,
            'total_timesteps': ENV_TRAINING_STEPS['MountainCarContinuous-v0'],
        },
        'LunarLander-v3': {
            'success_threshold': 200.0,
            'total_timesteps': ENV_TRAINING_STEPS['LunarLander-v3'],
        },
        'BipedalWalker-v3': {
            'success_threshold': 300.0,
            'total_timesteps': ENV_TRAINING_STEPS['BipedalWalker-v3'],
        },
    }
    config = base_config.get(env_id, {}).copy()
    if USE_UNIFIED_TRAINING_STEPS:
        config['total_timesteps'] = UNIFIED_TRAINING_STEPS
    return config 