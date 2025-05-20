import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from matplotlib.gridspec import GridSpec

# 设置更好看的图表样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12

# 从config.py中导入算法和环境列表
from config import ALGORITHMS, ENVIRONMENTS

# 创建输出目录
os.makedirs("analysis_figures", exist_ok=True)

# 读取所有算法和环境的实验结果
all_results = []

for algo in ALGORITHMS:
    for env in ENVIRONMENTS:
        summary_path = os.path.join("results", algo.lower(), env, "experiment_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                # 添加算法和环境信息
                summary['algorithm'] = algo
                summary['environment'] = env
                all_results.append(summary)

# 转换为DataFrame
df = pd.DataFrame([
    {
        "Algorithm": result["algorithm"],
        "Environment": result["environment"],
        "FinalReward": result["performance"]["final_reward_mean"],
        "StdReward": result["performance"]["final_reward_std"],
        "MinReward": result["performance"]["final_reward_min"],
        "MaxReward": result["performance"]["final_reward_max"],
        "Seeds": result["seeds"]
    }
    for result in all_results
])

# 添加环境特征
environment_types = {
    "CartPole-v1": {"ActionSpace": "Discrete", "RewardDensity": "Dense"},
    "LunarLander-v3": {"ActionSpace": "Discrete", "RewardDensity": "Dense"},
    "MountainCarContinuous-v0": {"ActionSpace": "Continuous", "RewardDensity": "Sparse"},
    "Pendulum-v1": {"ActionSpace": "Continuous", "RewardDensity": "Dense"},
    "BipedalWalker-v3": {"ActionSpace": "Continuous", "RewardDensity": "Sparse"}
}

df["ActionSpace"] = df["Environment"].map(lambda env: environment_types[env]["ActionSpace"])
df["RewardDensity"] = df["Environment"].map(lambda env: environment_types[env]["RewardDensity"])

# 算法类别
on_policy_algorithms = ["PPO", "A2C"]
off_policy_algorithms = ["DQN", "DDPG", "SAC"]
df["PolicyType"] = df["Algorithm"].map(lambda algo: "On-Policy" if algo in on_policy_algorithms else "Off-Policy")

# 组合所有环境的final reward柱状图到一个大图中
def plot_combined_final_rewards():
    fig = plt.figure(figsize=(12, 15))
    
    # 创建子图网格
    gs = GridSpec(5, 1, figure=fig, height_ratios=[1, 1, 1, 1, 1], hspace=0.4)
    
    environments = df["Environment"].unique()
    for i, env in enumerate(environments):
        ax = fig.add_subplot(gs[i])
        env_data = df[df["Environment"] == env].copy()
        
        # 设置颜色
        colors = ['blue' if algo in on_policy_algorithms else 'orange' for algo in env_data["Algorithm"]]
        
        # 处理MountainCar的特殊尺度
        if env == "MountainCarContinuous-v0":
            env_data["FinalReward"] = env_data["FinalReward"] * 1000
            env_data["StdReward"] = env_data["StdReward"] * 1000
            y_label = "Final Reward (×10⁻³)"
        else:
            y_label = "Final Reward"
        
        # 为Pendulum和MountainCar反转y轴
        invert_y = env in ["Pendulum-v1", "MountainCarContinuous-v0"]
        
        # 绘制柱状图
        bars = ax.bar(
            env_data["Algorithm"],
            env_data["FinalReward"],
            yerr=env_data["StdReward"],
            capsize=5,
            color=colors,
            alpha=0.7
        )
        
        # 设置图表标题和标签
        action_space = environment_types[env]["ActionSpace"]
        reward_density = environment_types[env]["RewardDensity"]
        ax.set_title(f"{env} ({action_space}, {reward_density} Rewards)")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 如果是负向奖励值的环境，反转y轴
        if invert_y:
            ax.invert_yaxis()
        
        # 为每个条形添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom' if not invert_y else 'top',
                fontsize=9
            )
    
    plt.tight_layout()
    plt.savefig("analysis_figures/combined_final_rewards.png", dpi=300, bbox_inches='tight')
    plt.close()

# 组合所有环境的学习曲线到一个大图中
def plot_combined_learning_curves():
    fig = plt.figure(figsize=(15, 18))
    
    # 创建子图网格
    gs = GridSpec(5, 1, figure=fig, height_ratios=[1, 1, 1, 1, 1], hspace=0.4)
    
    environments = df["Environment"].unique()
    for i, env in enumerate(environments):
        ax = fig.add_subplot(gs[i])
        
        # 为每个算法绘制学习曲线
        for algo in ALGORITHMS:
            # 检查该算法-环境组合是否有结果
            if not os.path.exists(os.path.join("results", algo.lower(), env)):
                continue
                
            # 查找所有种子的eval_curve.csv文件
            eval_curves = []
            seed_paths = glob(os.path.join("results", algo.lower(), env, "seed*_eval"))
            
            for seed_path in seed_paths:
                csv_path = os.path.join(seed_path, "eval_curve.csv")
                if os.path.exists(csv_path):
                    df_seed = pd.read_csv(csv_path)
                    eval_curves.append(df_seed)
            
            if not eval_curves:
                continue
                
            # 对齐所有曲线（以最短的为准）
            min_len = min(len(df_curve) for df_curve in eval_curves)
            timesteps = eval_curves[0]["timestep"][:min_len]
            rewards = np.stack([df_curve["reward"][:min_len].values for df_curve in eval_curves])
            
            mean_rewards = rewards.mean(axis=0)
            std_rewards = rewards.std(axis=0)
            
            # 绘制学习曲线
            color = 'blue' if algo in on_policy_algorithms else 'orange'
            ax.plot(timesteps, mean_rewards, label=f"{algo}", color=color)
            ax.fill_between(timesteps, mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.2, color=color)
        
        # 设置图表标题和标签
        action_space = environment_types[env]["ActionSpace"]
        reward_density = environment_types[env]["RewardDensity"]
        ax.set_title(f"Learning Curves for {env} ({action_space}, {reward_density} Rewards)")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Average Reward")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig("analysis_figures/combined_learning_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

# 修改后的按动作空间类型比较性能函数
def plot_by_action_space():
    # 创建一个空的列表来存储归一化后的性能数据
    normalized_data_list = []
    
    # 对每个环境进行归一化处理
    for env in df['Environment'].unique():
        env_data = df[df['Environment'] == env].copy()
        
        # 处理负值环境（如Pendulum），将最小值映射到0，最大值映射到1
        if env in ["Pendulum-v1", "MountainCarContinuous-v0"]:
            min_val = env_data['FinalReward'].min()
            max_val = env_data['FinalReward'].max()
            if max_val != min_val:
                env_data['NormalizedPerformance'] = (max_val - env_data['FinalReward']) / (max_val - min_val)
            else:
                env_data['NormalizedPerformance'] = 1.0
        else:
            # 正常环境归一化
            min_val = env_data['FinalReward'].min()
            max_val = env_data['FinalReward'].max()
            if max_val != min_val:
                env_data['NormalizedPerformance'] = (env_data['FinalReward'] - min_val) / (max_val - min_val)
            else:
                env_data['NormalizedPerformance'] = 1.0
        
        # 添加到列表中
        normalized_data_list.append(env_data[['Algorithm', 'Environment', 'NormalizedPerformance', 'PolicyType', 'ActionSpace']])
    
    # 一次性合并所有数据
    normalized_data = pd.concat(normalized_data_list, ignore_index=True) if normalized_data_list else pd.DataFrame()
    
    # 按动作空间和策略类型计算平均归一化性能
    action_space_data = normalized_data.groupby(['ActionSpace', 'PolicyType']).agg({
        'NormalizedPerformance': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(10, 6))
    
    # 设置x位置
    x_pos = np.arange(len(action_space_data['ActionSpace'].unique()))
    width = 0.35
    
    # 确保顺序一致
    action_spaces = sorted(action_space_data['ActionSpace'].unique())
    
    # 绘制分组柱状图
    for i, policy_type in enumerate(['On-Policy', 'Off-Policy']):
        policy_data = action_space_data[action_space_data['PolicyType'] == policy_type]
        policy_data = policy_data.set_index('ActionSpace').reindex(action_spaces).reset_index()
        
        bars = plt.bar(
            x_pos + (width/2 if i == 0 else -width/2),
            policy_data['NormalizedPerformance'],
            width=width,
            color='blue' if i == 0 else 'orange',
            label=policy_type,
            alpha=0.7
        )
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.02,
                f'{height:.2f}',
                ha='center',
                fontsize=9
            )
    
    plt.xlabel('Action Space Type')
    plt.ylabel('Normalized Performance')
    plt.title('Performance by Action Space')
    plt.xticks(x_pos, action_spaces)
    plt.ylim(0, 0.8)  # 设置y轴限制，使图表更清晰
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("analysis_figures/normalized_performance_by_action_space.png", dpi=300, bbox_inches='tight')
    plt.close()

# 修改后的按奖励密度比较性能函数
def plot_by_reward_density():
    # 创建一个空的列表来存储归一化后的性能数据
    normalized_data_list = []
    
    # 对每个环境进行归一化处理
    for env in df['Environment'].unique():
        env_data = df[df['Environment'] == env].copy()
        
        # 处理负值环境（如Pendulum），将最小值映射到0，最大值映射到1
        if env in ["Pendulum-v1", "MountainCarContinuous-v0"]:
            min_val = env_data['FinalReward'].min()
            max_val = env_data['FinalReward'].max()
            if max_val != min_val:
                env_data['NormalizedPerformance'] = (max_val - env_data['FinalReward']) / (max_val - min_val)
            else:
                env_data['NormalizedPerformance'] = 1.0
        else:
            # 正常环境归一化
            min_val = env_data['FinalReward'].min()
            max_val = env_data['FinalReward'].max()
            if max_val != min_val:
                env_data['NormalizedPerformance'] = (env_data['FinalReward'] - min_val) / (max_val - min_val)
            else:
                env_data['NormalizedPerformance'] = 1.0
        
        # 添加到列表中
        normalized_data_list.append(env_data[['Algorithm', 'Environment', 'NormalizedPerformance', 'PolicyType', 'RewardDensity']])
    
    # 一次性合并所有数据
    normalized_data = pd.concat(normalized_data_list, ignore_index=True) if normalized_data_list else pd.DataFrame()
    
    # 按奖励密度和策略类型计算平均归一化性能
    reward_density_data = normalized_data.groupby(['RewardDensity', 'PolicyType']).agg({
        'NormalizedPerformance': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(10, 6))
    
    # 设置x位置
    x_pos = np.arange(len(reward_density_data['RewardDensity'].unique()))
    width = 0.35
    
    # 确保顺序一致
    reward_densities = sorted(reward_density_data['RewardDensity'].unique())
    
    # 绘制分组柱状图
    for i, policy_type in enumerate(['On-Policy', 'Off-Policy']):
        policy_data = reward_density_data[reward_density_data['PolicyType'] == policy_type]
        policy_data = policy_data.set_index('RewardDensity').reindex(reward_densities).reset_index()
        
        bars = plt.bar(
            x_pos + (width/2 if i == 0 else -width/2),
            policy_data['NormalizedPerformance'],
            width=width,
            color='blue' if i == 0 else 'orange',
            label=policy_type,
            alpha=0.7
        )
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.02,
                f'{height:.2f}',
                ha='center',
                fontsize=9
            )
    
    plt.xlabel('Reward Density')
    plt.ylabel('Normalized Performance')
    plt.title('Performance by Reward Density')
    plt.xticks(x_pos, reward_densities)
    plt.ylim(0, 0.6)  # 设置y轴限制，使图表更清晰
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("analysis_figures/normalized_performance_by_reward_density.png", dpi=300, bbox_inches='tight')
    plt.close()

# 按算法稳定性比较性能
def plot_by_algorithm_stability():
    # 将标准差归一化为变异系数 (Coefficient of Variation)
    df['CV'] = df['StdReward'].abs() / df['FinalReward'].abs().replace(0, np.nan)
    
    # 按算法和策略类型计算平均变异系数
    stability_data = df.groupby(['Algorithm', 'PolicyType']).agg({
        'CV': 'mean',
        'FinalReward': 'mean'
    }).reset_index()
    
    # 排序以便于比较
    stability_data = stability_data.sort_values('CV')
    
    plt.figure(figsize=(12, 6))
    
    # 设置颜色
    colors = ['blue' if algo in on_policy_algorithms else 'orange' for algo in stability_data["Algorithm"]]
    
    # 绘制柱状图
    bars = plt.bar(
        stability_data["Algorithm"],
        stability_data["CV"],
        color=colors,
        alpha=0.7
    )
    
    plt.xlabel('Algorithm')
    plt.ylabel('Coefficient of Variation (Lower is Better)')
    plt.title('Algorithm Stability Comparison')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加性能标注
    for i, bar in enumerate(bars):
        perf = stability_data["FinalReward"].iloc[i]
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height()/2,
            f'Avg Perf: {perf:.2f}',
            ha='center',
            color='white',
            fontsize=9,
            fontweight='bold'
        )
    
    plt.tight_layout()
    plt.savefig("analysis_figures/algorithm_stability.png", dpi=300, bbox_inches='tight')
    plt.close()

# 算法性能热图
def plot_performance_heatmap():
    # 创建算法-环境性能矩阵
    pivot_data = df.pivot_table(
        values='FinalReward', 
        index='Algorithm',
        columns='Environment', 
        aggfunc='mean'
    )
    
    # 对每个环境列进行归一化处理
    normalized_pivot = pivot_data.copy()
    
    for col in normalized_pivot.columns:
        # 对Pendulum和MountainCar反转，因为负值越大越好
        if col in ["Pendulum-v1", "MountainCarContinuous-v0"]:
            col_min = pivot_data[col].min()
            col_max = pivot_data[col].max()
            if col_max != col_min:
                normalized_pivot[col] = (col_max - pivot_data[col]) / (col_max - col_min)
            else:
                normalized_pivot[col] = 1
        else:
            col_min = pivot_data[col].min()
            col_max = pivot_data[col].max()
            if col_max != col_min:
                normalized_pivot[col] = (pivot_data[col] - col_min) / (col_max - col_min)
            else:
                normalized_pivot[col] = 1
    
    # 创建热力图
    plt.figure(figsize=(14, 8))
    
    # 将NaN值替换为-1，这样它们在热力图中会显示为白色
    heatmap_data = normalized_pivot.fillna(-1)
    
    # 定义一个自定义的colormap，其中NaN值为白色
    cmap = plt.cm.YlOrRd
    cmap.set_under('white')
    
    # 在每列上方添加环境属性
    col_labels = [f"{col}\n{environment_types[col]['ActionSpace']}, {environment_types[col]['RewardDensity']}"
                 for col in heatmap_data.columns]
    
    # 绘制热力图
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        cmap=cmap,
        mask=heatmap_data < 0,  # 屏蔽NaN值（现在是-1）
        vmin=0,
        vmax=1,
        fmt=".2f",
        linewidths=.5,
        cbar_kws={"label": "Normalized Performance"},
        xticklabels=col_labels
    )
    
    plt.title("Normalized Algorithm Performance Across Environments")
    plt.ylabel("Algorithm")
    plt.tight_layout()
    plt.savefig("analysis_figures/performance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

# 创建组合图表函数 - 将动作空间和奖励密度比较图放在一起
def plot_combined_normalized_performance():
    # 获取按动作空间和奖励密度归一化的数据
    action_space_data_list = []
    reward_density_data_list = []
    
    # 对每个环境进行归一化处理
    for env in df['Environment'].unique():
        env_data = df[df['Environment'] == env].copy()
        
        # 处理负值环境（如Pendulum），将最小值映射到0，最大值映射到1
        if env in ["Pendulum-v1", "MountainCarContinuous-v0"]:
            min_val = env_data['FinalReward'].min()
            max_val = env_data['FinalReward'].max()
            if max_val != min_val:
                env_data['NormalizedPerformance'] = (max_val - env_data['FinalReward']) / (max_val - min_val)
            else:
                env_data['NormalizedPerformance'] = 1.0
        else:
            # 正常环境归一化
            min_val = env_data['FinalReward'].min()
            max_val = env_data['FinalReward'].max()
            if max_val != min_val:
                env_data['NormalizedPerformance'] = (env_data['FinalReward'] - min_val) / (max_val - min_val)
            else:
                env_data['NormalizedPerformance'] = 1.0
        
        # 添加到动作空间和奖励密度列表
        action_space_data_list.append(env_data[['Algorithm', 'Environment', 'NormalizedPerformance', 'PolicyType', 'ActionSpace']])
        reward_density_data_list.append(env_data[['Algorithm', 'Environment', 'NormalizedPerformance', 'PolicyType', 'RewardDensity']])
    
    # 合并数据
    action_space_data = pd.concat(action_space_data_list, ignore_index=True) if action_space_data_list else pd.DataFrame()
    reward_density_data = pd.concat(reward_density_data_list, ignore_index=True) if reward_density_data_list else pd.DataFrame()
    
    # 计算平均值
    action_space_avg = action_space_data.groupby(['ActionSpace', 'PolicyType']).agg({
        'NormalizedPerformance': 'mean'
    }).reset_index()
    
    reward_density_avg = reward_density_data.groupby(['RewardDensity', 'PolicyType']).agg({
        'NormalizedPerformance': 'mean'
    }).reset_index()
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # 动作空间子图
    x_pos = np.arange(len(action_space_avg['ActionSpace'].unique()))
    width = 0.35
    action_spaces = sorted(action_space_avg['ActionSpace'].unique())
    
    for i, policy_type in enumerate(['On-Policy', 'Off-Policy']):
        policy_data = action_space_avg[action_space_avg['PolicyType'] == policy_type]
        policy_data = policy_data.set_index('ActionSpace').reindex(action_spaces).reset_index()
        
        bars = ax1.bar(
            x_pos + (width/2 if i == 0 else -width/2),
            policy_data['NormalizedPerformance'],
            width=width,
            color='blue' if i == 0 else 'orange',
            label=policy_type,
            alpha=0.7
        )
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.02,
                f'{height:.2f}',
                ha='center',
                fontsize=9
            )
    
    ax1.set_xlabel('Action Space Type')
    ax1.set_ylabel('Normalized Performance')
    ax1.set_title('Performance by Action Space')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(action_spaces)
    ax1.set_ylim(0, 0.8)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 奖励密度子图
    x_pos = np.arange(len(reward_density_avg['RewardDensity'].unique()))
    reward_densities = sorted(reward_density_avg['RewardDensity'].unique())
    
    for i, policy_type in enumerate(['On-Policy', 'Off-Policy']):
        policy_data = reward_density_avg[reward_density_avg['PolicyType'] == policy_type]
        policy_data = policy_data.set_index('RewardDensity').reindex(reward_densities).reset_index()
        
        bars = ax2.bar(
            x_pos + (width/2 if i == 0 else -width/2),
            policy_data['NormalizedPerformance'],
            width=width,
            color='blue' if i == 0 else 'orange',
            label=policy_type,
            alpha=0.7
        )
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.02,
                f'{height:.2f}',
                ha='center',
                fontsize=9
            )
    
    ax2.set_xlabel('Reward Density')
    ax2.set_ylabel('Normalized Performance')
    ax2.set_title('Performance by Reward Density')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(reward_densities)
    ax2.set_ylim(0, 0.6)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("analysis_figures/combined_normalized_performance.png", dpi=300, bbox_inches='tight')
    plt.close()

# 运行所有图表生成函数
plot_combined_final_rewards()
plot_combined_learning_curves()
plot_by_action_space()
plot_by_reward_density()
plot_by_algorithm_stability()
plot_performance_heatmap()
plot_combined_normalized_performance()  # 添加新的组合图表函数

print("所有图表已生成并保存到 analysis_figures/ 目录下")