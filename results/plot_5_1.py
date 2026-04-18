import argparse
import wandb
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib as mpl
import sys
import pandas as pd
import matplotlib.font_manager as fm  # 引入字体管理模块

# ================= 1. 终极字体修复方案 =================

def configure_fonts():
    """
    自动检测系统字体并配置 Matplotlib，彻底解决中文乱码
    """
    # 1. 先加载样式 (如果有的化)
    try:
        import scienceplots
        # 关键：必须加 'no-latex'，否则 LaTeX 引擎渲染中文极其复杂
        plt.style.use(['science', 'ieee', 'no-latex'])
    except ImportError:
        print("[提示] 未检测到 SciencePlots，将使用默认风格")
        plt.style.use('ggplot')

    # 2. 定义常用的中文字体候选列表 (优先级从高到低)
    # Windows: 微软雅黑, 黑体, 宋体
    # Mac: 苹方, 黑体-简
    # Linux: 文泉驿, Noto Sans CJK
    cjk_candidates = [
        'Microsoft YaHei', 'SimHei', 'SimSun',      # Windows
        'PingFang SC', 'Heiti TC', 'STHeiti',       # Mac
        'WenQuanYi Micro Hei', 'Noto Sans CJK SC'   # Linux
    ]
    
    # 3. 扫描系统已安装的字体
    system_fonts = set(f.name for f in fm.fontManager.ttflist)
    found_fonts = [f for f in cjk_candidates if f in system_fonts]
    
    if found_fonts:
        print(f"[字体] 检测到系统可用中文字体: {found_fonts}")
        # 将找到的中文字体插入到 sans-serif 列表的最前面 (最高优先级)
        plt.rcParams['font.sans-serif'] = found_fonts + plt.rcParams['font.sans-serif']
    else:
        # 如果没找到，强制指定常见字体试试 (保底)
        print("[警告] 未检测到常用中文字体，尝试强制配置...")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] + plt.rcParams['font.sans-serif']

    # 4. 核心修复：强制 Matplotlib 使用 sans-serif 族
    # scienceplots 默认可能会用 serif (衬线体)，导致我们设置的 sans-serif 不生效
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 5. 解决负号显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False

# ================= 2. 数据获取与绘图逻辑 =================

def get_stats_for_dim(api, project_path, base_filters, seeds, metric_type='acc', p_miss_test=0.5):
    """ 获取统计数据 """
    values = []
    
    if metric_type == 'acc':
        metric_keys = [f"final_test_acc_{p_miss_test}", "final_test_acc", "test_acc"]
    else:
        metric_keys = [f"final_test_f1_{p_miss_test}", "final_test_f1", "test_f1"]

    for s in seeds:
        filters = base_filters.copy()
        filters["config.seed"] = s
        
        try:
            runs = api.runs(project_path, filters=filters, order="-created_at")
            if not runs: continue
            
            run = runs[0]
            val = None
            
            for key in metric_keys:
                if key in run.summary:
                    val = run.summary[key]
                    break
            
            if val is not None:
                if metric_type == 'acc' and val <= 1.0: val *= 100
                if metric_type == 'f1' and val <= 1.0: val *= 100
                values.append(val)
                
        except Exception as e:
            print(f"    [Warn] Seed {s} 读取异常: {e}")
            continue

    if not values: return None, None
    return np.mean(values), np.std(values)

def plot_metric(ax, x_dims, means, stds, color, p_miss):
    """ 绘图函数 """
    label_cn = f'LASER (测试缺失率 $p_{{test}}={p_miss}$)'
    
    # 绘制折线
    ax.plot(x_dims, means, marker='D', markersize=6, linestyle='-', linewidth=2, color=color, label=label_cn, zorder=10)
    
    # 绘制阴影
    ax.fill_between(x_dims, np.array(means) - np.array(stds), np.array(means) + np.array(stds), 
                    color=color, alpha=0.2, zorder=5, label='标准差范围')
    
    ax.set_xscale('log', base=2)
    ax.set_xticks(x_dims)
    ax.set_xticklabels(x_dims, fontsize=10)
    ax.grid(True, which='major', linestyle='--', alpha=0.5)
    
    # 动态调整 Y 轴
    y_min, y_max = np.min(np.array(means) - np.array(stds)), np.max(np.array(means) + np.array(stds))
    margin = (y_max - y_min) * 0.3 if (y_max - y_min) > 0 else 5.0
    # 确保下限不小于 0，上限不超 100
    ax.set_ylim(max(0, y_min - margin), min(100, y_max + margin))

def main(args):
    # [第一步] 应用字体修复
    configure_fonts()

    api = wandb.Api()
    if args.entity is None: args.entity = api.default_entity
    project_path = f"{args.entity}/{args.project_name}"
    
    seeds = [5, 6, 7, 8, 9]
    x_dims = [2, 4, 8, 16, 32, 64, 128]

    print(f"--- 敏感性分析: 通信维度 vs 性能 ---")
    
    data_records = []

    # 画布布局：一行两列
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    # 配置：(类型, 坐标轴, Y轴标签, 颜色, 标题)
    metrics_config = [
        ('acc', axes[0], '测试准确率 (%)', '#d62728', '准确率 (Accuracy)'),
        ('f1', axes[1], '测试 F1-Score (%)', '#1f77b4', 'F1 分数 (F1-Score)')
    ]

    for m_type, ax, ylabel, color, title_cn in metrics_config:
        print(f"\n正在处理指标: {m_type.upper()} ...")
        means, stds, valid_x = [], [], []
        
        for dim in x_dims:
            base_filters = {
                "config.task_name": args.task_name,
                "config.method": args.method,
                "config.num_clients": args.num_clients,
                "config.p_miss_train": args.p_miss_train,
                "config.cut_dim": dim,
                "state": "finished"
            }
            if args.mechanism: base_filters["config.mechanism"] = args.mechanism

            mean, std = get_stats_for_dim(api, project_path, base_filters, seeds, m_type, args.p_miss)
            
            if mean is not None:
                print(f"  Dim={dim}: {mean:.2f} ± {std:.2f}")
                means.append(mean)
                stds.append(std)
                valid_x.append(dim)
                
                if m_type == 'acc': 
                    data_records.append({'Dim': dim, 'Acc_Mean': mean, 'Acc_Std': std})
                else:
                    for item in data_records:
                        if item['Dim'] == dim:
                            item['F1_Mean'] = mean
                            item['F1_Std'] = std
            else:
                print(f"  Dim={dim}: [无数据]")

        if valid_x:
            plot_metric(ax, valid_x, means, stds, color, args.p_miss)
            ax.set_xlabel('通信维度 ($d_{cut}$)')
            ax.set_ylabel(ylabel)
            ax.set_title(title_cn)
            # 使用 prop 参数强制图例使用中文字体
            ax.legend(prop={'size': 10})
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    
    # 保存
    out_dir = Path("./results/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    f_name = f"Sensitivity_CutDim_{args.task_name}_P{args.p_miss}_CN_Fixed"
    
    plt.savefig(out_dir / f"{f_name}.png", bbox_inches='tight')
    plt.savefig(out_dir / f"{f_name}.pdf", bbox_inches='tight')
    
    # 导出数据
    df = pd.DataFrame(data_records)
    csv_path = out_dir / f"{f_name}.csv"
    df.to_csv(csv_path, index=False, float_format='%.2f')
    
    print(f"\n[完成] 中文修复版图片已保存: {out_dir / f'{f_name}.png'}")
    print(f"[完成] 数据已导出: {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', default='laser')
    parser.add_argument('--entity', default=None)
    parser.add_argument('--task_name', default='diabetes5')
    parser.add_argument('--method', default='laser')
    parser.add_argument('--num_clients', type=int, default=4)
    parser.add_argument('--p_miss_train', type=float, default=0.5)
    parser.add_argument('--p_miss', type=float, default=0.5)
    parser.add_argument('--mechanism', default='uniform')
    args = parser.parse_args()
    main(args)