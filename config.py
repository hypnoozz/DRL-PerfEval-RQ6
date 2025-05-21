import gymnasium as gym
from typing import Dict, List, Tuple
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS: List[int] = [1, 2, 3, 4, 5]

ALGORITHMS = ['DQN', 'PPO', 'A2C', 'SAC', 'DDPG']

ENVIRONMENTS: List[str] = [
    'CartPole-v1',
    'Pendulum-v1',
    'MountainCarContinuous-v0',
    'LunarLander-v3',
    'BipedalWalker-v3'
]

USE_UNIFIED_TRAINING_STEPS = True 
UNIFIED_TRAINING_STEPS = 200_000

NETWORK_CONFIG = {
    'hidden_sizes': [256, 256], 
    'activation': 'ReLU',      
    'learning_rate': 3e-4,      
}


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


LOG_CONFIG = {
    'log_dir': 'logs',
    'model_dir': 'models',
    'results_dir': 'results', 
}

METRICS = [
    'episode_reward',
    'episode_length',
    'success_rate',
    'training_time',
    'average_value_loss',
    'average_policy_loss',
    'raw_rewards',  
    'smoothed_rewards',  
    'final_reward_distribution', 
    'early_phase_rewards',  
]

UNIFIED_TRAINING_CONFIG = {
    'total_timesteps': 200_000,      
    'num_seeds': 5,                
    'gamma': 0.99,                   
    'learning_rate': 3e-4,           
    'optimizer': 'Adam',            
    'batch_size': 64,                
    'eval_frequency': 10_000,       
    'reward_normalization': True,    
    'env_wrapper': 'GymMonitor+CustomLogger',
    'early_stopping': False,         
    'early_phase_steps': 50_000,     
    'smoothing_window': 100,         
}

def get_env_config(env_id: str) -> Dict:
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
