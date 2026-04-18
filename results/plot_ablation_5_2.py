import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.font_manager as fm
from pathlib import Path
from datetime import datetime, timedelta

def configure_style():
    try:
        import scienceplots
        plt.style.use(['science', 'ieee', 'no-latex'])
    except ImportError:
        plt.style.use('seaborn-v0_8-whitegrid')

    plt.rcParams['axes.unicode_minus'] = False
    cjk_candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'PingFang SC']
    system_fonts = set(f.name for f in fm.fontManager.ttflist)
    found_fonts = [f for f in cjk_candidates if f in system_fonts]
    if found_fonts:
        plt.rcParams['font.sans-serif'] = found_fonts + plt.rcParams['font.sans-serif']
        plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams.update({'font.size': 12, 'figure.titlesize': 16})

def fetch_data_multi(api, entity, project, task_name, acc_key, f1_key, days_back=7):
    path = f"{entity}/{project}"
    print(f"正在连接 WandB 项目: {path} ...")
    start_date = (datetime.now() - timedelta(days=days_back)).isoformat()
    filters = {"config.task_name": task_name, "state": "finished", "created_at": {"$gt": start_date}} 
    
    print(f"-> 正在拉取数据...")
    runs = api.runs(path, filters=filters, order="-created_at")

    # [核心修改] 匹配全新的架构演进逻辑
    variants_map = {
        'abl_1': 'SVFL\n(无缺失应对)',
        'abl_2': 'Ensemble\n(无任务采样)',
        'abl_3': 'Combinatorial\n(无参数共享)',
        'abl_4': 'LASER\n(完整版)'
    }
    
    data_acc = {label: [] for label in variants_map.values()}
    data_f1 = {label: [] for label in variants_map.values()}
    data_time = {label: [] for label in variants_map.values()}
    seen_run_names = set()
    
    for run in runs:
        matched_label = None
        for key, label in variants_map.items():
            if key in run.name:
                matched_label = label
                break
        if not matched_label: continue
        if run.name in seen_run_names: continue
        seen_run_names.add(run.name)

        if acc_key in run.summary:
            val = run.summary[acc_key]
            if val <= 1.0: val *= 100
            data_acc[matched_label].append(val)
        if f1_key in run.summary:
            val = run.summary[f1_key]
            if val <= 1.0: val *= 100
            data_f1[matched_label].append(val)
        if '_runtime' in run.summary:
            data_time[matched_label].append(run.summary['_runtime'] / 60.0)

    def to_df(data_dict):
        records = []
        for label, vals in data_dict.items():
            if vals:
                records.append({'Variant': label, 'Mean': np.mean(vals), 'Std': np.std(vals)})
            else:
                records.append({'Variant': label, 'Mean': 0, 'Std': 0})
        return pd.DataFrame(records)

    return to_df(data_acc).rename(columns={'Mean': 'Mean_Acc', 'Std': 'Std_Acc'}), \
           to_df(data_f1).rename(columns={'Mean': 'Mean_F1', 'Std': 'Std_F1'}), \
           to_df(data_time).rename(columns={'Mean': 'Mean_Runtime(min)', 'Std': 'Std_Runtime'})

def plot_dual_chart(df_acc, df_f1, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    colors = ['#A9A9A9', '#8FAADC', '#8FAADC', '#D62728'] 
    
    metrics_config = [
        (axes[0], df_acc, 'Mean_Acc', 'Std_Acc', '测试准确率 (%)', '(a) 架构演进 - 准确率'),
        (axes[1], df_f1, 'Mean_F1', 'Std_F1', '测试 F1-Score (%)', '(b) 架构演进 - F1 分数')
    ]

    for ax, df, mean_col, std_col, ylabel, title in metrics_config:
        bars = ax.bar(df['Variant'], df[mean_col], yerr=df[std_col], capsize=8, 
                      color=colors, alpha=0.9, width=0.6, edgecolor='black', linewidth=1)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                        f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        valid_means = df[df[mean_col] > 0][mean_col]
        if not valid_means.empty:
            ax.set_ylim(max(0, valid_means.min() - 10), min(100, valid_means.max() + 5))
            
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
        ax.set_title(title, fontsize=14, pad=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.tick_params(axis='x', labelsize=11)

    plt.tight_layout()
    save_path = output_dir / "Architecture_Evolution_5_2.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n[完成] 架构演进柱状图已保存至: {save_path}")
    plt.show()

def main():
    ENTITY, PROJECT, TASK = "jun19357384004-", "laser", "diabetes5"
    ACC_KEY, F1_KEY = "final_test_acc_0.5", "final_test_f1_0.5"
    
    configure_style()
    api = wandb.Api()
    if not ENTITY: ENTITY = api.default_entity
        
    df_acc, df_f1, df_time = fetch_data_multi(api, ENTITY, PROJECT, TASK, ACC_KEY, F1_KEY, days_back=0.5)
    
    if df_acc['Mean_Acc'].sum() > 0:
        out_dir = Path("results/plots")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        merged_df = pd.merge(df_acc, df_f1, on="Variant")
        merged_df = pd.merge(merged_df, df_time, on="Variant")
        
        print_df = merged_df.copy()
        print_df['Variant'] = print_df['Variant'].str.replace('\n', ' ')
        
        print("\n=================================================================================")
        print("                 [ 架构演进完整数据表 (ACC, F1, 耗时) ]                ")
        print("=================================================================================")
        print(print_df.round(2).to_string(index=False, justify='center'))
        print("=================================================================================\n")
        
        plot_dual_chart(df_acc, df_f1, out_dir)
    else:
        print("\n[错误] 未找到数据。")

if __name__ == "__main__":
    main()