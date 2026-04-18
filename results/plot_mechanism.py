import argparse
import wandb
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib as mpl

# --- 1. 科研绘图风格设置 ---
try:
    import scienceplots
    plt.style.use(['science', 'ieee', 'no-latex'])
    # 如果系统有 Times New Roman，可以取消下面注释
    # plt.rcParams['font.family'] = 'Times New Roman'
except ImportError:
    print("[提示] 未检测到 SciencePlots，将使用默认风格")
    plt.style.use('seaborn-v0_8-paper')

# --- 2. 中文字体设置 ---
# 解决负号显示问题
mpl.rcParams['axes.unicode_minus'] = False

from sys import platform
if platform == "win32":
    # Windows 优先使用微软雅黑或黑体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Times New Roman']
elif platform == "darwin":
    # Mac 优先使用苹方或 Arial Unicode
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
else:
    # Linux 优先使用文泉驿
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']

# --- 3. 配置常量 ---
# [修改] 图表上显示的文字改成中文
MECH_LABELS = ['均匀缺失', 'Beta分布', '马尔可夫', '标签相关']

# WandB 和 Bat 文件中的 mechanism 对应值 (不要改这里，这是数据索引键)
MECH_KEYS = ['uniform', 'beta', 'markov', 'label']

def get_metric_stats(api, project_path, base_filters, method, mechanism, target_rate):
    """
    获取特定配置下的统计数据 (均值, 标准差, 样本数)
    """
    filters = base_filters.copy()
    filters["config.method"] = method
    filters["config.mechanism"] = mechanism
    
    try:
        runs = api.runs(project_path, filters=filters)
        values = []
        
        for run in runs:
            # 优先尝试获取带后缀的精确 key
            keys_to_try = [
                f"final_test_acc_{target_rate}", 
                f"test_acc_{target_rate}",
                "final_test_acc", 
                "test_acc"
            ]
            
            val = None
            # 1. 尝试从 summary 获取
            for k in keys_to_try:
                if k in run.summary:
                    val = run.summary[k]
                    break
            
            # 2. 尝试从 history 获取
            if val is None:
                for k in keys_to_try:
                    hist = run.history(keys=[k], pandas=False)
                    if hist and len(hist) > 0 and k in hist[-1]:
                        val = hist[-1][k]
                        break

            if val is not None:
                # 统一转为百分比 (0-100)
                if val <= 1.0: val *= 100
                values.append(val)
        
        n = len(values)
        if n == 0:
            return 0.0, 0.0, 0
            
        avg_val = np.mean(values)
        std_val = np.std(values)
        return avg_val, std_val, n

    except Exception as e:
        print(f"[错误] 查询 WandB 失败 ({method}-{mechanism}): {e}")
        return 0.0, 0.0, 0

def plot_charts(results, args, save_dir):
    """
    results: 包含 'local', 'ensemble', 'laser' 的字典，每个存有 {'mean': [], 'std': []}
    """
    # 提取数据
    means_loc = results['local']['mean']
    stds_loc = results['local']['std']
    
    means_ens = results['ensemble']['mean']
    stds_ens = results['ensemble']['std']
    
    means_las = results['laser']['mean']
    stds_las = results['laser']['std']

    # 检查数据有效性
    if sum(means_loc) + sum(means_las) == 0:
        print("[错误] 数据全为 0，跳过绘图。请检查 WandB 项目名和参数配置。")
        return

    # === 图1: 分组柱状图 (带误差棒) ===
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    x = np.arange(len(MECH_LABELS))
    width = 0.25

    # 颜色风格
    c_loc = '#D3D3D3'     # 浅灰
    c_ens = '#87CEFA'     # 浅蓝
    c_las = '#FF6347'     # 番茄红

    # [修改] 绘制柱子 + 误差棒 + 中文图例标签
    ax.bar(x - width, means_loc, width, yerr=stds_loc, label='Local (本地基准)', 
           color=c_loc, edgecolor='black', linewidth=0.8, capsize=4)
    
    ax.bar(x, means_ens, width, yerr=stds_ens, label='Ensemble (集成方法)', 
           color=c_ens, edgecolor='black', linewidth=0.8, capsize=4)
    
    ax.bar(x + width, means_las, width, yerr=stds_las, label='LASER (本文方法)', 
           color=c_las, edgecolor='black', linewidth=0.8, hatch='//', capsize=4)

    # [修改] 坐标轴与标题中文设置
    ax.set_ylabel('测试集准确率 (%)', fontsize=12)
    ax.set_xlabel(f'特征缺失机制 (缺失率={args.p_miss_train})', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(MECH_LABELS, fontsize=11)
    ax.set_title(f"{args.task_name} 上的性能表现", fontsize=13, pad=10)

    # 动态 Y 轴范围
    all_vals = means_loc + means_ens + means_las
    valid_vals = [v for v in all_vals if v > 10]
    if valid_vals:
        ax.set_ylim(min(valid_vals) - 5, max(valid_vals) + 5)

    # 图例
    ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
              ncol=3, frameon=False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # 保存柱状图
    f_bar = f"Bar_CN_{args.task_name}_Rate{args.p_miss_train}"
    plt.savefig(save_dir / f"{f_bar}.png", dpi=300)
    plt.savefig(save_dir / f"{f_bar}.pdf", format='pdf')
    print(f"[绘图成功] 柱状图 (中文): {f_bar}.png")

    # === 图2: 雷达图 ===
    labels = MECH_LABELS
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1] # 闭合

    # 闭合数据
    v_loc = means_loc + means_loc[:1]
    v_ens = means_ens + means_ens[:1]
    v_las = means_las + means_las[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), dpi=300)
    
    # [修改] 中文图例标签
    # Local
    ax.plot(angles, v_loc, color='#808080', linewidth=1.5, linestyle='--', label='Local (本地)')
    ax.fill(angles, v_loc, color='#808080', alpha=0.1)

    # Ensemble
    ax.plot(angles, v_ens, color='#005EB8', linewidth=2, linestyle='-', label='Ensemble (集成)')
    ax.fill(angles, v_ens, color='#005EB8', alpha=0.1)

    # LASER
    ax.plot(angles, v_las, color='#d62728', linewidth=2, linestyle='-', marker='o', label='LASER (本文)')
    ax.fill(angles, v_las, color='#d62728', alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12) # 这里使用的是上面定义的中文 MECH_LABELS
    ax.tick_params(axis='y', labelsize=9)
    
    if valid_vals:
        ax.set_ylim(min(valid_vals)-5, max(valid_vals)+2)

    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=10)
    plt.tight_layout()
    
    f_radar = f"Radar_CN_{args.task_name}_Rate{args.p_miss_train}"
    plt.savefig(save_dir / f"{f_radar}.png", dpi=300)
    plt.savefig(save_dir / f"{f_radar}.pdf", format='pdf')
    print(f"[绘图成功] 雷达图 (中文): {f_radar}.png")

def main(args):
    api = wandb.Api()
    if args.entity is None: 
        args.entity = api.default_entity
    project_path = f"{args.entity}/{args.project_name}"
    
    print(f"\n{'='*50}")
    print(f" 开始可视化 (中文版): {args.task_name} | Rate: {args.p_miss_train} | Clients: {args.num_clients}")
    print(f" 项目路径: {project_path}")
    print(f"{'='*50}")

    # 基础过滤条件
    base_filters = {
        "config.task_name": args.task_name,
        "config.p_miss_train": args.p_miss_train,
        "config.num_clients": args.num_clients
    }

    # 数据容器
    results = {
        'local': {'mean': [], 'std': []},
        'ensemble': {'mean': [], 'std': []},
        'laser': {'mean': [], 'std': []}
    }
    methods = ['local', 'ensemble', 'laser']

    print(f"{'Mechanism':<12} | {'Method':<10} | {'Runs':<5} | {'Acc (Mean ± Std)'}")
    print("-" * 60)

    for mech in MECH_KEYS:
        for method in methods:
            avg, std, n = get_metric_stats(api, project_path, base_filters, method, mech, args.p_miss_train)
            
            results[method]['mean'].append(avg)
            results[method]['std'].append(std)
            
            print(f"{mech:<12} | {method:<10} | {n:<5} | {avg:.2f} ± {std:.2f}")
    
    print("-" * 60)

    # 绘图
    out_dir = Path("./results/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_charts(results, args, out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--project_name', default='laser', help='WandB 项目名')
    parser.add_argument('--entity', default=None, help='WandB 用户名')
    
    parser.add_argument('--task_name', default='diabetes5', help='Task Name')
    parser.add_argument('--num_clients', type=int, default=4, help='Client Num')
    parser.add_argument('--p_miss_train', type=float, default=0.5, help='Missing Rate')
    
    args = parser.parse_args()
    main(args)