import wandb
import numpy as np
import argparse
import time
import socket
from requests.exceptions import RequestException

def get_runs_safe(api, project_name, filters, max_retries=10):
    """
    带重试机制的获取 Run 列表函数，专门应对网络不稳定
    """
    for attempt in range(max_retries):
        try:
            # 尝试获取 run 迭代器并转换为列表（触发网络请求）
            runs = api.runs(project_name, filters=filters)
            return list(runs)
        except (wandb.errors.CommError, ConnectionError, RequestException, socket.timeout, Exception) as e:
            if attempt == max_retries - 1:
                print(f"  [Error] Failed after {max_retries} attempts. Error: {e}")
                return []
            
            wait_time = (attempt + 1) * 2 # 递增等待时间 2s, 4s, 6s...
            print(f"  [Network unstable] Connection failed, retrying in {wait_time}s... ({attempt+1}/{max_retries})")
            time.sleep(wait_time)
            
            # 重新初始化 API 以防 Session 断开
            try:
                api = wandb.Api(timeout=60)
            except:
                pass
    return []

def main(project_name, run_names, metric_name):
    
    # 初始化 API，设置超时时间长一点
    try:
        api = wandb.Api(timeout=60)
    except Exception as e:
        print("  [Init Error] Could not initialize WandB API. Check internet connection.")
        return

    metric_values = []
    
    for run_name in run_names:
        # 使用带重试机制的函数获取数据
        run_list = get_runs_safe(api, project_name, filters={"display_name": run_name})
        
        if not run_list:
            # print(f"  [Warning] No run found with name '{run_name}'")
            continue
        
        run = run_list[0]
        
        # 获取指标
        metric_value = run.summary.get(metric_name)
        if metric_value is not None:
            metric_values.append(metric_value)
        else:
            print(f"  [Metric Missing] '{metric_name}' not found in run '{run_name}'")

    if metric_values:
        if "f1" in metric_name.lower():
            metric_values = [v * 100 for v in metric_values]
        
        mean_val, std_val = np.mean(metric_values), np.std(metric_values)
        print(f"{metric_name}: {mean_val:.1f} ± {std_val:.1f} (Found {len(metric_values)}/{len(run_names)} runs)")
    else:
        print(f"{metric_name}: N/A (No runs found)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', default='laser') # 您的项目名
    parser.add_argument('--task_name', choices=['diabetes1', 'diabetes2', 'diabetes3', 'diabetes4', 'diabetes5', 'diabetes6'], required=True)
    parser.add_argument('--method', choices=['local', 'svfl', 'ensemble', 'combinatorial', 'plug', 'laser'], required=True)
    parser.add_argument('--num_clients', type=int, default=4)
    parser.add_argument('--metric', choices=['acc', 'f1'], required=True)
    parser.add_argument('--p_miss_train', nargs='+', default=['0.0', '0.1', '0.5']) 
    parser.add_argument('--p_miss_test', nargs='+', default=['0.0', '0.1', '0.5'])
    
    args = parser.parse_args()

    # 自动判断种子数量
    num_seeds = 1 if args.task_name == 'diabetes6' else 5

    for p_miss_train in args.p_miss_train:
        print(f"p_miss_train {p_miss_train}")
        for p_miss_test in args.p_miss_test:
            metric_name = f"final_test_{args.metric}_{p_miss_test}"
            run_names = [f"{args.task_name}_{args.method}_K{args.num_clients}_p_miss_train{p_miss_train}_s{i}" for i in range(num_seeds)]
            
            main(args.project_name, run_names, metric_name)