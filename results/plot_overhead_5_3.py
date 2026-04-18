import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.font_manager as fm
from pathlib import Path
from datetime import datetime, timedelta

# ================= 1. 样式与字体 =================
def configure_style():
    try:
        import scienceplots
        plt.style.use(['science', 'ieee', 'no-latex'])
    except ImportError:
        plt.style.use('seaborn-v0_8-whitegrid')

    plt.rcParams['axes.unicode_minus'] = False
    cjk_fonts = ['Microsoft YaHei', 'SimHei', 'PingFang SC']
    sys_fonts = set(f.name for f in fm.fontManager.ttflist)
    found = [f for f in cjk_fonts if f in sys_fonts]
    if found:
        plt.rcParams['font.sans-serif'] = found + plt.rcParams['font.sans-serif']
        plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams.update({'font.size': 12})

# ================= 2. 理论通信量计算器 =================
def calculate_communication(method, num_clients=4, cut_dim=32, num_samples=70000, epochs=100):
    bytes_per_number = 4
    if method == 'local':
        return 0.0
    elif method == 'ensemble':
        total_bytes = num_clients * 2 * num_samples * epochs * bytes_per_number * 2
        return total_bytes / (1024 * 1024)
    elif method in ['svfl', 'combinatorial', 'laser']:
        # SVFL, Combinatorial 和 LASER 的客户端通信量完全相等！
        # 客户端只负责传 cut_dim 维度的 embedding，组合计算均在服务端。
        total_bytes = num_clients * cut_dim * num_samples * epochs * bytes_per_number * 2
        return total_bytes / (1024 * 1024)
    return 0.0

# ================= 3. 数据拉取 =================
def fetch_overhead_data(api, entity, project):
    path = f"{entity}/{project}"
    start_date = (datetime.now() - timedelta(days=1)).isoformat()
    
    filters = {
        "config.task_name": "diabetes5",
        "state": "finished",
        "created_at": {"$gt": start_date}
    }
    runs = api.runs(path, filters=filters)
    
    raw_data = {
        "Local": {'time': [], 'acc': []},
        "Ensemble": {'time': [], 'acc': []},
        "SVFL": {'time': [], 'acc': []},
        "Combinatorial": {'time': [], 'acc': []},
        "LASER (Ours)": {'time': [], 'acc': []}
    }
    
    for run in runs:
        if "5.3" not in run.name: continue
        
        method = "Unknown"
        if "local" in run.name: method = "Local"
        elif "ensemble" in run.name: method = "Ensemble"
        elif "svfl" in run.name: method = "SVFL"
        elif "combinatorial" in run.name: method = "Combinatorial"
        elif "laser" in run.name: method = "LASER (Ours)"
        
        if method == "Unknown": continue

        runtime = run.summary.get('_runtime', 0)
        acc = run.summary.get('final_test_acc_0.0', run.summary.get('test_acc', 0))
        if isinstance(acc, list): acc = acc[0]
        if acc <= 1.0: acc *= 100
            
        raw_data[method]['time'].append(runtime)
        raw_data[method]['acc'].append(acc)
        
    results = {}
    for m in raw_data:
        times = raw_data[m]['time']
        accs = raw_data[m]['acc']
        if not times: continue
        
        results[m] = {
            'Runtime (s)': np.mean(times),
            'Runtime_std': np.std(times),
            'Accuracy (%)': np.mean(accs),
            'Comm (MB)': calculate_communication(m.split()[0].lower())
        }
        
    df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    df.rename(columns={'index': 'Method'}, inplace=True)
    
    order = ["Local", "Ensemble", "SVFL", "Combinatorial", "LASER (Ours)"]
    df['Method'] = pd.Categorical(df['Method'], categories=order, ordered=True)
    return df.sort_values('Method')

# ================= 4. 绘制双轴柱状图 =================
def plot_overhead(df, output_dir):
    fig, ax1 = plt.subplots(figsize=(11, 6), dpi=300)
    
    x = np.arange(len(df['Method']))
    width = 0.35
    
    color1 = '#4A7EBB'
    bars1 = ax1.bar(x - width/2, df['Runtime (s)'], width, yerr=df['Runtime_std'], capsize=5,
                    color=color1, edgecolor='black', label='平均训练耗时 (s)')
    ax1.set_ylabel('训练耗时 (秒)', color=color1, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = '#D9534F'
    bars2 = ax2.bar(x + width/2, df['Comm (MB)'], width, color=color2, edgecolor='black', label='理论通信总量 (MB)')
    ax2.set_ylabel('通信总量 (MB)', color=color2, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Method'], fontweight='bold')
    
    # 移除了在柱子上标注 Accuracy 的代码段

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.title('计算耗时与通信开销评估', fontsize=15, pad=25, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    # 调整 Y 轴上限
    ax1.set_ylim(0, max(df['Runtime (s)']) * 1.20)
    ax2.set_ylim(0, max(df['Comm (MB)']) * 1.20 if max(df['Comm (MB)']) > 0 else 10)
    
    plt.tight_layout()
    save_path = output_dir / "Efficiency_Overhead_5_3.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n[成功] 5.3 节开销评估图表已保存至: {save_path}")
    plt.show()

def main():
    ENTITY = "jun19357384004-"
    PROJECT = "laser"
    
    configure_style()
    api = wandb.Api()
    
    print("正在拉取 5.3 节的 25 条实验记录并计算平均值...")
    df = fetch_overhead_data(api, ENTITY, PROJECT)
    
    if not df.empty:
        print("\n--- 开销评估数据表 ---")
        print(df[['Method', 'Runtime (s)', 'Runtime_std', 'Comm (MB)', 'Accuracy (%)']].to_string(index=False))
        
        out_dir = Path("results/plots")
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "Overhead_Table_5_3.csv", index=False)
        plot_overhead(df, out_dir)
    else:
        print("[错误] 未拉取到数据。")

if __name__ == "__main__":
    main()