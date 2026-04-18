import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.ticker as ticker
from pathlib import Path

# ================= 1. 全局样式与字体设置 =================

# 使用 Seaborn 默认的白底网格风格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# --- 核心：中文字体配置 ---
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 自动选择系统中的中文字体
if sys.platform.startswith('win'):
    # Windows 优先尝试这些字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
else:
    # Mac/Linux 备用
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'DejaVu Sans']

# --- 字体大小微调 ---
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,     # (a) (b) 标题大小
    'axes.labelsize': 14,     # 坐标轴名称大小
    'xtick.labelsize': 12,    # 刻度数字大小
    'ytick.labelsize': 12,
    'legend.fontsize': 11,    # 图例文字大小
    'figure.titlesize': 18
})

def get_wandb_data(entity, project, task_name, methods, train_rates, test_rates, seeds):
    """
    (逻辑保持不变，仅用于获取数据)
    """
    api = wandb.Api(timeout=60)
    path = f"{entity}/{project}" if entity else project
    print(f"正在连接 WandB 项目: {path} ...")
    
    filters = {"config.task_name": task_name, "state": "finished"}
    runs = api.runs(path, filters=filters, order="-created_at")
    
    print(f"正在解析数据...")
    rows = []
    seen_configs = set()
    
    # 预期总数
    total_expected = len(methods) * len(train_rates) * len(seeds)
    print(f"目标：需收集 {total_expected} 个唯一配置的实验数据...")
    
    for run in runs:
        cfg = run.config
        m = cfg.get('method')
        tr = cfg.get('p_miss_train')
        s = cfg.get('seed')
        
        if m not in methods: continue
        if s not in seeds: continue
        if tr is not None: 
            try: tr = round(float(tr), 1)
            except: continue
        if tr not in train_rates: continue
        
        config_key = (m, tr, s)
        if config_key in seen_configs: continue
            
        summary = run.summary
        has_data = False
        
        for te in test_rates:
            key = f"final_test_acc_{te}"
            if key in summary:
                acc = summary[key]
                if acc <= 1.0: acc *= 100
                rows.append({
                    "Method": m,
                    "Train Rate": tr,
                    "Test Rate": te,
                    "Accuracy": acc,
                    "Seed": s
                })
                has_data = True
        
        if has_data:
            seen_configs.add(config_key)
            print(f"\r[进度] 已提取: {len(seen_configs)}/{total_expected}", end="", flush=True)
            
        if len(seen_configs) >= total_expected:
            print("\n\n✅ 找齐了所有数据！停止扫描。")
            break
            
    print(f"数据提取完成，共获取 {len(rows)} 条绘图点。")
    return pd.DataFrame(rows)

def main():
    # ================= 配置区域 =================
    PROJECT = "laser"      
    ENTITY = None          
    TASK = "diabetes2"
    SEEDS = [10, 11, 12, 13, 14]
    TRAIN_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    METHODS = ['local', 'combinatorial', 'laser']
    ALL_TEST_RATES = [0.0, 0.3, 0.5, 0.7]

    # ================= 获取数据 =================
    df = get_wandb_data(ENTITY, PROJECT, TASK, METHODS, TRAIN_RATES, ALL_TEST_RATES, SEEDS)

    if df.empty:
        print("错误：未获取到数据。")
        return

    print("正在绘图 (中文版)...")

    # ================= 开始绘图 =================
    # 创建画布 (15:6 比例)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300, constrained_layout=True)
    
    colors = {'local': '#7f7f7f', 'combinatorial': '#1f77b4', 'laser': '#d62728'}
    
    # ------------------ 左图 (a): 方法对比 ------------------
    ax1 = axes[0]
    target_test = 0.3
    df1 = df[df['Test Rate'] == target_test]

    if not df1.empty:
        sns.lineplot(
            data=df1, x='Train Rate', y='Accuracy', hue='Method', style='Method',
            markers=True, dashes=False, 
            linewidth=2.5, markersize=9, 
            ax=ax1, palette=colors, errorbar='sd'
        )
        
        # --- [关键修改] 图例汉化 ---
        handles, _ = ax1.get_legend_handles_labels()
        # 英文 -> 中文 映射字典
        labels_map = {
            'local': 'Local (基准)', 
            'combinatorial': 'Combinatorial (对比)', 
            'laser': 'LASER (本文方法)'
        }
        new_labels = [labels_map.get(h.get_label(), h.get_label()) for h in handles]
        
        # --- [关键修改] 标题和坐标轴汉化 ---
        ax1.set_title(f"(a) 不同方法的鲁棒性对比 ($p_{{test}}={target_test}$)", fontweight='bold', pad=10)
        ax1.set_xlabel("训练集缺失率 ($p_{train}$)")
        ax1.set_ylabel("测试准确率 (%)")
        
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        
        # 图例设置
        ax1.legend(handles=handles, labels=new_labels, title="方法", 
                   loc='best', frameon=True, framealpha=0.9, edgecolor='#ccc')

    # ------------------ 右图 (b): LASER 性能全景 ------------------
    ax2 = axes[1]
    df2 = df[df['Method'] == 'laser']
    
    if not df2.empty:
        sns.lineplot(
            data=df2, x='Train Rate', y='Accuracy', hue='Test Rate', style='Test Rate',
            markers=True, dashes=False, 
            linewidth=2.5, markersize=9, 
            ax=ax2, palette='viridis_r', errorbar=None
        )

        # --- [关键修改] 标题和坐标轴汉化 ---
        ax2.set_title(f"(b) LASER 在不同测试场景下的表现", fontweight='bold', pad=10)
        ax2.set_xlabel("训练集缺失率 ($p_{train}$)")
        ax2.set_ylabel("测试准确率 (%)")
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        
        # --- [关键修改] 图例标题汉化 ---
        ax2.legend(title="测试集缺失率 ($p_{test}$)", 
                   loc='best', frameon=True, framealpha=0.9, edgecolor='#ccc')

    # ================= 保存 =================
    out_file = "results/plots/paper_plots_chinese.png"
    plt.savefig(out_file, dpi=300)
    print(f"\n[完美] 中文绘图完成！已保存至: {out_file}")
    
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    main()