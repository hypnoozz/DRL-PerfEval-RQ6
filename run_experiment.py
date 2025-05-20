import os
import argparse
import numpy as np
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from config import *
from utils import set_seed, create_env, aggregate_results
import pandas as pd
from stable_baselines3 import DQN, PPO, A2C, SAC, DDPG
from stable_baselines3.common.callbacks import EvalCallback

SB3_ALGO_MAP = {
    "DQN": DQN,
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
    "DDPG": DDPG,
}

def parse_args():
    parser = argparse.ArgumentParser(description='RL Algorithm Experiment Runner')
    parser.add_argument('--algo', type=str, choices=ALGORITHMS,
                      help='RL algorithm to use')
    parser.add_argument('--env', type=str, choices=ENVIRONMENTS,
                      help='Environment to train on')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed (default: None runs all seeds)')
    return parser.parse_args()

def run_single_experiment(algo: str, env_name: str, seed: int):
    result_dir = os.path.join("results", algo.lower(), env_name)
    summary_path = os.path.join(result_dir, f"seed{seed}_summary.json")
    if os.path.exists(summary_path):
        print(f"[已存在] {algo} on {env_name} (seed {seed}) 已有 summary，跳过。")
        return None

    from gymnasium import spaces
    env = create_env(env_name)
    is_discrete = isinstance(env.action_space, spaces.Discrete)
    is_continuous = isinstance(env.action_space, spaces.Box)
    if algo == 'DQN' and not is_discrete:
        print(f"[跳过] {algo} 不支持连续动作空间环境 {env_name}，已跳过。")
        return None
    if algo in ['SAC', 'DDPG'] and not is_continuous:
        print(f"[跳过] {algo} 只适用于连续动作空间环境，{env_name} 为离散动作空间，已跳过。")
        return None

    set_seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    model_class = SB3_ALGO_MAP[algo]
    model = model_class(
        "MlpPolicy", env, verbose=0, seed=seed,
        tensorboard_log=os.path.join("results", algo.lower(), env_name, "tb"),
        **ALGO_SPECIFIC_CONFIG[algo]
    )

    eval_env = create_env(env_name)
    eval_env.reset(seed=seed+1000)
    eval_log_dir = os.path.join("results", algo.lower(), env_name, f"seed{seed}_eval")
    os.makedirs(eval_log_dir, exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=eval_log_dir,
        log_path=eval_log_dir,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    total_timesteps = get_env_config(env_name)['total_timesteps']
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # 导出评估过程为csv
    eval_data_path = os.path.join(eval_log_dir, "evaluations.npz")
    if os.path.exists(eval_data_path):
        data = np.load(eval_data_path)
        timesteps = data["timesteps"]
        rewards = data["results"]
        lengths = data["ep_lengths"]
        avg_rewards = rewards.mean(axis=1)
        avg_lengths = lengths.mean(axis=1)
        df = pd.DataFrame({
            "timestep": timesteps,
            "reward": avg_rewards,
            "length": avg_lengths
        })
        df.to_csv(os.path.join(eval_log_dir, "eval_curve.csv"), index=False)

    # 只保存 summary，指向 eval_curve.csv
    results = {
        "algo": algo,
        "env": env_name,
        "seed": seed,
        "eval_curve_csv": os.path.join(eval_log_dir, "eval_curve.csv")
    }
    os.makedirs(result_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    return results

def run_single_seed(args):
    """运行单个种子的实验（用于并行处理）"""
    algo, env_name, seed = args
    return run_single_experiment(algo, env_name, seed)

def get_unfinished_args():
    args_list = []
    for algo in ALGORITHMS:
        for env_name in ENVIRONMENTS:
            for seed in SEEDS:
                result_dir = os.path.join("results", algo.lower(), env_name)
                summary_path = os.path.join(result_dir, f"seed{seed}_summary.json")
                if not os.path.exists(summary_path):
                    args_list.append((algo, env_name, seed))
    return args_list

def main():
    args = parse_args()
    if args.algo and args.env:
        # 检查 experiment_summary.json 是否存在
        result_dir = os.path.join("results", args.algo.lower(), args.env)
        exp_summary_path = os.path.join(result_dir, "experiment_summary.json")
        if os.path.exists(exp_summary_path):
            print(f"[已存在] {args.algo} on {args.env} 所有种子已完成，跳过。")
            return
        seeds = [args.seed] if args.seed is not None else SEEDS
        parallel_args = [(args.algo, args.env, seed) for seed in seeds]
        num_processes = min(cpu_count(), len(seeds))
        print(f"Using {num_processes} processes to run {len(seeds)} seeds in parallel")
        with Pool(processes=num_processes) as pool:
            all_results = list(tqdm(
                pool.imap(run_single_seed, parallel_args),
                total=len(seeds),
                desc=f"Running {args.algo} on {args.env}"
            ))
        valid_results = [r for r in all_results if r is not None]
        if valid_results:
            aggregate_results(args.algo, args.env, valid_results)
        else:
            print(f"[跳过聚合] {args.algo} on {args.env} 没有有效实验结果，跳过聚合。")
    else:
        for algo in ALGORITHMS:
            for env_name in ENVIRONMENTS:
                result_dir = os.path.join("results", algo.lower(), env_name)
                exp_summary_path = os.path.join(result_dir, "experiment_summary.json")
                if os.path.exists(exp_summary_path):
                    print(f"[已存在] {algo} on {env_name} 所有种子已完成，跳过。")
                    continue
                print(f"\nRunning {algo} on {env_name}")
                seeds = SEEDS
                parallel_args = [(algo, env_name, seed) for seed in seeds]
                num_processes = min(cpu_count(), len(seeds))
                print(f"Using {num_processes} processes to run {len(seeds)} seeds in parallel")
                with Pool(processes=num_processes) as pool:
                    all_results = list(tqdm(
                        pool.imap(run_single_seed, parallel_args),
                        total=len(seeds),
                        desc=f"Running {algo} on {env_name}"
                    ))
                valid_results = [r for r in all_results if r is not None]
                if valid_results:
                    aggregate_results(algo, env_name, valid_results)
                else:
                    print(f"[跳过聚合] {algo} on {env_name} 没有有效实验结果，跳过聚合。")

if __name__ == "__main__":
    main()