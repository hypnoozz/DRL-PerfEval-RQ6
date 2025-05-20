import os
import json
import time
import torch
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

class ExperimentLogger:
    def __init__(self, algo_name: str, env_name: str, seed: int):
        """初始化实验记录器"""
        self.algo_name = algo_name
        self.env_name = env_name
        self.seed = seed
        
        # 创建实验目录
        self.results_dir = os.path.join("results", algo_name.lower(), env_name)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 创建tensorboard目录
        self.tensorboard_dir = os.path.join(self.results_dir, "tensorboard")
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tensorboard_dir)
        
        # 初始化结果字典
        self.results = {
            'algo_name': algo_name,
            'env_name': env_name,
            'seed': seed,
            'episodes': [],
            'rewards': [],
            'lengths': [],
            'times': [],
            'success_rate': [],
            'value_losses': [],
            'policy_losses': [],
            'raw_rewards': [],  # 原始奖励
            'smoothed_rewards': [],  # 平滑后的奖励
            'timesteps': [],  # 时间步
        }
        
        self.start_time = time.time()
        self.total_timesteps = 0
        self.total_episodes = 0
        self.max_reward = float('-inf')
        self.min_reward = float('inf')
        
    def log_episode(self, episode: int, reward: float, length: int,
                   value_loss: float = None, policy_loss: float = None,
                   success: bool = None, timesteps: int = None, **kwargs):
        """记录每个episode的结果"""
        self.total_episodes += 1
        if timesteps is not None:
            self.total_timesteps = timesteps
        
        self.results['episodes'].append(episode)
        self.results['rewards'].append(reward)
        self.results['lengths'].append(length)
        self.results['times'].append(time.time() - self.start_time)
        self.results['raw_rewards'].append(reward)
        self.results['timesteps'].append(self.total_timesteps)
        
        # 更新最大/最小奖励
        self.max_reward = max(self.max_reward, reward)
        self.min_reward = min(self.min_reward, reward)
        
        # 计算平滑奖励
        if len(self.results['raw_rewards']) >= 100:
            smoothed_reward = np.mean(self.results['raw_rewards'][-100:])
            self.results['smoothed_rewards'].append(smoothed_reward)
        
        if value_loss is not None:
            self.results['value_losses'].append(value_loss)
            self.writer.add_scalar('Loss/value', value_loss, self.total_timesteps)
            
        if policy_loss is not None:
            self.results['policy_losses'].append(policy_loss)
            self.writer.add_scalar('Loss/policy', policy_loss, self.total_timesteps)
            
        if success is not None:
            self.results['success_rate'].append(float(success))
            
        # 记录到tensorboard
        self.writer.add_scalar('Reward/episode', reward, self.total_timesteps)
        self.writer.add_scalar('Length/episode', length, self.total_timesteps)
        
        # 记录其他自定义指标
        for key, value in kwargs.items():
            if key not in self.results:
                self.results[key] = []
            self.results[key].append(value)
            self.writer.add_scalar(f'Custom/{key}', value, self.total_timesteps)
            
    def save_results(self):
        """保存实验结果"""
        # 计算最终统计信息
        final_avg_reward = np.mean(self.results['rewards'][-100:]) if self.results['rewards'] else 0
        final_avg_length = np.mean(self.results['lengths'][-100:]) if self.results['lengths'] else 0
        final_avg_loss = np.mean(self.results['value_losses'][-100:]) if self.results['value_losses'] else 0
        
        # 创建summary字典
        summary = {
            'env_name': self.env_name,
            'algorithm': self.algo_name,
            'seed': self.seed,
            'total_episodes': int(self.total_episodes),
            'total_timesteps': int(self.total_timesteps),
            'training_time_seconds': float(time.time() - self.start_time),
            'steps_per_second': float(self.total_timesteps / (time.time() - self.start_time)),
            'final_100_episodes': {
                'mean_reward': float(final_avg_reward),
                'std_reward': float(np.std(self.results['rewards'][-100:]) if self.results['rewards'] else 0),
                'mean_length': float(final_avg_length)
            },
            'overall': {
                'max_reward': float(self.max_reward),
                'min_reward': float(self.min_reward),
                'mean_reward': float(np.mean(self.results['rewards'])) if self.results['rewards'] else 0,
                'std_reward': float(np.std(self.results['rewards'])) if self.results['rewards'] else 0
            },
            'convergence': {
                'solved': bool(final_avg_reward >= 195.0),  # CartPole-v1 标准
                'solved_at_step': self._find_solving_step(195.0) if final_avg_reward >= 195.0 else None,
                'final_avg_loss': float(final_avg_loss) if self.results['value_losses'] else None
            }
        }
        
        # 保存CSV日志
        df_dict = {
            'timestep': self.results['timesteps'],
            'reward': self.results['raw_rewards'],
            'length': self.results['lengths'],
        }
        
        # 只添加长度一致的字段
        if len(self.results['smoothed_rewards']) == len(self.results['raw_rewards']):
            df_dict['smoothed_reward'] = self.results['smoothed_rewards']
        if len(self.results['value_losses']) == len(self.results['raw_rewards']):
            df_dict['value_loss'] = self.results['value_losses']
        if len(self.results['policy_losses']) == len(self.results['raw_rewards']):
            df_dict['policy_loss'] = self.results['policy_losses']
        if len(self.results['success_rate']) == len(self.results['raw_rewards']):
            df_dict['success_rate'] = self.results['success_rate']
        
        df = pd.DataFrame(df_dict)
        csv_path = os.path.join(self.results_dir, f"seed{self.seed}_log.csv")
        df.to_csv(csv_path, index=False)
        
        return summary
        
    def _find_solving_step(self, threshold):
        """找到首次达到解决标准的时间步"""
        if not self.results['rewards']:
            return None
            
        window_size = 100
        for i in range(len(self.results['rewards']) - window_size + 1):
            if np.mean(self.results['rewards'][i:i+window_size]) >= threshold:
                return self.results['timesteps'][i]
        return None
        
    def _generate_plots(self):
        """生成并保存可视化图表"""
        # 1. Reward vs Timestep - Raw Curve
        plt.figure(figsize=(10, 6))
        plt.plot(self.results['timesteps'], self.results['raw_rewards'])
        plt.title(f'{self.algo_name} on {self.env_name} (Seed {self.seed})')
        plt.xlabel('Timesteps')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, f'reward_raw_seed{self.seed}.png'))
        plt.close()
        
        # 2. Reward vs Timestep - Smoothed Curve
        if len(self.results['smoothed_rewards']) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(self.results['timesteps'][-len(self.results['smoothed_rewards']):], 
                    self.results['smoothed_rewards'])
            plt.title(f'{self.algo_name} on {self.env_name} - Smoothed (Seed {self.seed})')
            plt.xlabel('Timesteps')
            plt.ylabel('Smoothed Reward')
            plt.grid(True)
            plt.savefig(os.path.join(self.results_dir, f'reward_smooth_seed{self.seed}.png'))
            plt.close()
        
        # 3. Early Phase Zoom-in (First 50K steps)
        early_idx = [i for i, t in enumerate(self.results['timesteps']) if t <= 50_000]
        if early_idx:
            plt.figure(figsize=(10, 6))
            plt.plot(
                [self.results['timesteps'][i] for i in early_idx],
                [self.results['raw_rewards'][i] for i in early_idx]
            )
            plt.title(f'{self.algo_name} on {self.env_name} - Early Phase (Seed {self.seed})')
            plt.xlabel('Timesteps')
            plt.ylabel('Reward')
            plt.grid(True)
            plt.savefig(os.path.join(self.results_dir, f'reward_zoom_early_seed{self.seed}.png'))
            plt.close()
            
def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def create_env(env_name: str):
    """创建环境实例"""
    return gym.make(env_name)

def evaluate_policy(env, policy, n_episodes=10, max_steps=1000):
    """评估策略的性能"""
    rewards = []
    lengths = []
    successes = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()  # gymnasium返回(state, info)
        total_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < max_steps:
            action = policy(state)
            if isinstance(action, np.ndarray):
                action = action.item()  # 转换为标量
            next_state, reward, terminated, truncated, info = env.step(action)  # gymnasium返回5个值
            done = terminated or truncated
            total_reward += reward
            episode_length += 1
            state = next_state
            
        rewards.append(total_reward)
        lengths.append(episode_length)
        # 某些环境可能在info中提供是否成功的信息
        if 'is_success' in info:
            successes.append(info['is_success'])
            
    results = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
    }
    
    if successes:
        results['success_rate'] = np.mean(successes)
        
    return results 

def aggregate_results(algo: str, env_name: str, all_results: list):
    """聚合多个种子的结果并生成汇总图表和统计信息"""
    results_dir = os.path.join("results", algo.lower(), env_name)

    # 1. 读取所有种子的 eval_curve.csv
    eval_curves = []
    for result in all_results:
        eval_curve_path = result.get("eval_curve_csv")
        if eval_curve_path and os.path.exists(eval_curve_path):
            df = pd.read_csv(eval_curve_path)
            eval_curves.append(df)
    if not eval_curves:
        print("No eval curves found.")
        return

    # 2. 对齐所有曲线（以最短的为准）
    min_len = min(len(df) for df in eval_curves)
    timesteps = eval_curves[0]["timestep"][:min_len]
    rewards = np.stack([df["reward"][:min_len].values for df in eval_curves])
    lengths = np.stack([df["length"][:min_len].values for df in eval_curves])

    mean_rewards = rewards.mean(axis=0)
    std_rewards = rewards.std(axis=0)
    mean_lengths = lengths.mean(axis=0)
    std_lengths = lengths.std(axis=0)

    # 3. 学习曲线
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_rewards, label="Mean Reward", color="blue")
    plt.fill_between(timesteps, mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.2, color="blue")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title(f"{algo} on {env_name} - Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "learning_curve.png"))
    plt.close()

    # 4. reward分布直方图（最后一个评估点）
    final_rewards = rewards[:, -1]
    plt.figure(figsize=(10, 6))
    plt.hist(final_rewards, bins=10, alpha=0.7, color="blue")
    plt.axvline(np.mean(final_rewards), color="red", linestyle="dashed", label="Mean")
    plt.title(f"{algo} on {env_name} - Final Reward Distribution")
    plt.xlabel("Final Average Reward")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "final_reward_hist.png"))
    plt.close()

    # 5. 最终表现柱状图
    plt.figure(figsize=(6, 6))
    plt.bar(['Final Performance'], [np.mean(final_rewards)], yerr=[np.std(final_rewards)],
            capsize=5, alpha=0.7, color='blue')
    plt.title(f"{algo} on {env_name} - Final Performance")
    plt.ylabel("Reward (Mean ± Std)")
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "final_performance_bar.png"))
    plt.close()

    # 6. 保存汇总统计
    summary = {
        'environment': env_name,
        'algorithm': algo,
        'seeds': len(all_results),
        'performance': {
            'final_reward_mean': float(np.mean(final_rewards)),
            'final_reward_std': float(np.std(final_rewards)),
            'final_reward_min': float(np.min(final_rewards)),
            'final_reward_max': float(np.max(final_rewards))
        }
    }
    with open(os.path.join(results_dir, 'experiment_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"图表和汇总已保存到 {results_dir}") 