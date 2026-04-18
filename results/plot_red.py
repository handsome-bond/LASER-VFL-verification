import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ================= 配置区域 =================
# 请确保路径正确
FILE_PATH_1 = './data/diabetes1/Diabetes_Final_Data_V2.csv' 
FILE_PATH_2 = './data/diabetes2/diabetes2.csv'

# 图片保存目录
OUTPUT_DIR = 'results/plots'
# ===========================================

# ================= 关键：设置中文字体 =================
plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows 常用中文字体
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号
# ====================================================

def visualize_dataset_complexity_separate(file_path, dataset_name, cn_name):
    print(f"\n>>> 正在分析: {dataset_name} ({cn_name}) ...")
    
    # 0. 检查并创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 清理文件名中的特殊字符，用于保存文件
    file_prefix = dataset_name.replace(" ", "_").replace("(", "").replace(")", "")

    # 1. 读取数据
    if not os.path.exists(file_path):
        print(f"  [错误] 文件不存在: {file_path}")
        return
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"  [错误] 读取失败: {e}")
        return

    # 2. 预处理
    target_col = df.columns[-1] 
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 填充数值列
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].mean())
        else:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    # -----------------------------------------------------------
    # 图 1: t-SNE 数据分布
    # -----------------------------------------------------------
    print("  -> [1/3] 生成 t-SNE 分布图...")
    plt.figure(figsize=(6, 5)) # 单张图的大小
    
    X_scaled = StandardScaler().fit_transform(X)
    if len(X_scaled) > 2000:
        indices = np.random.choice(len(X_scaled), 2000, replace=False)
        X_tsne_input = X_scaled[indices]
        y_tsne = y[indices]
    else:
        X_tsne_input = X_scaled
        y_tsne = y

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X_tsne_input)

    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_tsne, cmap='coolwarm', alpha=0.6, s=20)
    plt.title(f"{cn_name}\n数据分布可视化 (t-SNE)", fontsize=14, fontweight='bold')
    plt.xlabel("维度 1", fontsize=12)
    plt.ylabel("维度 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 图例
    handles, _ = scatter.legend_elements()
    plt.legend(handles, ["类别 0 (健康)", "类别 1 (患病)"], title="样本类别", loc='best')
    
    # 保存
    save_path_1 = os.path.join(OUTPUT_DIR, f"{file_prefix}_1_tSNE.png")
    plt.tight_layout()
    plt.savefig(save_path_1, dpi=300)
    plt.close() # 关闭画布，释放内存
    print(f"     已保存: {save_path_1}")


    # -----------------------------------------------------------
    # 图 2: 特征相关性条形图
    # -----------------------------------------------------------
    print("  -> [2/3] 生成特征相关性图...")
    plt.figure(figsize=(6, 5))
    
    full_data = pd.DataFrame(X, columns=X.columns)
    full_data['Target'] = y
    corr = full_data.corr()['Target'].drop('Target').abs().sort_values(ascending=False)
    
    top_n = min(10, len(corr))
    # 使用 viridis 配色，颜色越深代表相关性越高
    sns.barplot(x=corr.values[:top_n], y=corr.index[:top_n], palette='viridis')
    
    plt.title(f"{cn_name}\n前 {top_n} 大特征相关性", fontsize=14, fontweight='bold')
    plt.xlabel("与目标变量的相关性 (绝对值)", fontsize=12)
    plt.xlim(0, 1.0) 
    plt.grid(axis='x', linestyle='--', alpha=0.3)

    save_path_2 = os.path.join(OUTPUT_DIR, f"{file_prefix}_2_Correlation.png")
    plt.tight_layout()
    plt.savefig(save_path_2, dpi=300)
    plt.close()
    print(f"     已保存: {save_path_2}")


    # -----------------------------------------------------------
    # 图 3: 最强特征密度图
    # -----------------------------------------------------------
    print("  -> [3/3] 生成最强特征分离度图...")
    plt.figure(figsize=(6, 5))
    
    top_feature = corr.index[0]
    
    sns.kdeplot(data=full_data, x=top_feature, hue='Target', fill=True, 
                palette='coolwarm', alpha=0.5, common_norm=False)
    
    plt.title(f"{cn_name}\n最强特征的类别分离度\n(特征: '{top_feature}')", fontsize=14, fontweight='bold')
    plt.xlabel(f"特征值: {top_feature}", fontsize=12)
    plt.ylabel("密度 (Density)", fontsize=12)
    plt.legend(title='类别', labels=['患病 (1)', '健康 (0)'])
    plt.grid(True, linestyle='--', alpha=0.3)

    save_path_3 = os.path.join(OUTPUT_DIR, f"{file_prefix}_3_Separation.png")
    plt.tight_layout()
    plt.savefig(save_path_3, dpi=300)
    plt.close()
    print(f"     已保存: {save_path_3}")
    
    print(f"  -> {dataset_name} 全部图表生成完毕。\n")

# ================= 运行 =================
if __name__ == "__main__":
    visualize_dataset_complexity_separate(FILE_PATH_1, "Mendeley Data1", "简单数据集")
    visualize_dataset_complexity_separate(FILE_PATH_2, "Mendeley Data2", "困难数据集")