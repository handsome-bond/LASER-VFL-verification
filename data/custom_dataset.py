import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torchvision import transforms
# --- Mask Generator ---
def generate_mask(num_samples, num_clients, mechanism, p_miss, labels=None, seed=42):
    if p_miss is None: p_miss = 0.0
    rng = np.random.RandomState(seed)
    mask = np.ones((num_samples, num_clients), dtype=bool)

    if mechanism == 'uniform':
        rand_matrix = rng.rand(num_samples, num_clients)
        mask = rand_matrix >= p_miss
    elif mechanism == 'beta':
        alpha = 2.0
        beta_param = alpha * (1 - p_miss) / p_miss if p_miss > 0 else 2.0
        client_rates = rng.beta(alpha, beta_param, num_clients)
        for c in range(num_clients):
            mask[:, c] = rng.rand(num_samples) >= client_rates[c]
    return mask

# --- Batch Collate ---
def collate_fn(batch):
    items = list(zip(*batch))
    features_list_of_tuples = items[0]
    num_clients = len(features_list_of_tuples[0])
    
    stacked_features = []
    for c in range(num_clients):
        client_batch = [sample[c] for sample in features_list_of_tuples]
        stacked_features.append(torch.stack(client_batch))
        
    labels = torch.tensor(items[1], dtype=torch.long)
    masks = torch.stack(items[2])
    return (*stacked_features, labels, masks)

# --- Wrapper ---
class CustomDataset(Dataset):
    def __init__(self, base_dataset, batch_size, num_clients, p_miss, mechanism='uniform', seed=42):
        self.dataset = base_dataset
        self.num_clients = num_clients
        self.classes = getattr(base_dataset, 'classes', [0, 1]) 
        
        all_labels = None
        if hasattr(base_dataset, 'y'): all_labels = base_dataset.y
        elif hasattr(base_dataset, 'labels'): all_labels = base_dataset.labels
        
        self.masks = generate_mask(len(base_dataset), num_clients, mechanism, p_miss, all_labels, seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if isinstance(data, (tuple, list)):
            x, y = data[0], data[1]
        else:
            raise TypeError(f"Unexpected data type: {type(data)}")

        if isinstance(x, torch.Tensor):
            # [核心修复] 识别图像数据 (3维: C, H, W)，不进行盲目切分
            if x.dim() == 3:
                features = [x for _ in range(self.num_clients)]
            else:
                feat_dim = x.shape[-1]
                if feat_dim < self.num_clients:
                    features = [x for _ in range(self.num_clients)]
                else:
                    chunk_size = feat_dim // self.num_clients
                    features = []
                    for i in range(self.num_clients):
                        start = i * chunk_size
                        end = (i + 1) * chunk_size if i != self.num_clients - 1 else feat_dim
                        features.append(x[..., start:end])
        else:
            features = list(x)

        mask = torch.tensor(self.masks[idx], dtype=torch.bool)
        return (tuple(features), y, mask)

# --- Diabetes 1 (Standardized) ---
class DiabetesDataset1(Dataset):
    def __init__(self, csv_file, num_clients, is_train=True, train_ratio=0.8):
        data = pd.read_csv(csv_file)
        X = data.drop(columns=['diabetic']).values.astype(np.float32)
        y = data['diabetic'].values.astype(np.int64)
        
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        if is_train: self.X, self.y = X_train, y_train
        else: self.X, self.y = X_test, y_test
        self.num_clients = num_clients
        self.classes = np.unique(self.y)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): 
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# --- Diabetes 2 (Standardized) ---
class DiabetesDataset2(Dataset):
    def __init__(self, csv_file, num_clients, is_train=True, train_ratio=0.8):
        data = pd.read_csv(csv_file)
        X = data.drop(columns=['outcome']).values.astype(np.float32)
        y = data['outcome'].values.astype(np.int64)
        
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        if is_train: self.X, self.y = X_train, y_train
        else: self.X, self.y = X_test, y_test
        self.num_clients = num_clients
        self.classes = np.unique(self.y)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): 
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# --- Diabetes 3 (Standardized) ---
class DiabetesDataset3(Dataset):
    def __init__(self, csv_file, num_clients, is_train=True, train_ratio=0.8):
        data = pd.read_csv(csv_file)
        X = data.drop(columns=['class']).values.astype(np.float32)
        y = data['class'].values.astype(np.int64)
        
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        if is_train: self.X, self.y = X_train, y_train
        else: self.X, self.y = X_test, y_test
        self.num_clients = num_clients
        self.classes = np.unique(self.y)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): 
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# --- Diabetes 4 (Standardized) ---
class DiabetesDataset4(Dataset):
    def __init__(self, csv_file, num_clients, is_train=True, train_ratio=0.8):
        data = pd.read_csv(csv_file)
        for col in data.columns:
            if data[col].dtype == 'object': data[col] = data[col].astype('category').cat.codes
        X = data.drop(columns=['diabetes']).values.astype(np.float32)
        y = data['diabetes'].values.astype(np.int64)
        
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        if is_train: self.X, self.y = X_train, y_train
        else: self.X, self.y = X_test, y_test
        self.num_clients = num_clients
        self.classes = np.unique(self.y)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): 
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# --- Diabetes 5 (NHANES - Corrected) ---
class DiabetesDataset5(Dataset):
    def __init__(self, csv_path, num_clients, is_train=True, train_ratio=0.8):
        self.num_clients = num_clients
        
        df = None
        for sep in [',', ';', '\t']:
            try:
                temp_df = pd.read_csv(csv_path, sep=sep)
                if temp_df.shape[1] > 5:
                    df = temp_df
                    if is_train: print(f"[Dataset] Diabetes5 read with '{sep}'. Shape: {df.shape}")
                    break
            except: continue
        
        if df is None:
            df = pd.read_csv(csv_path, sep=None, engine='python')

        drop_cols = ['LBXGH', 'SEQN', 'Unnamed: 0', 'id', 'ID'] 
        existing_drop = [c for c in drop_cols if c in df.columns]
        
        if 'LBXGH' in df.columns:
            y = (df['LBXGH'] >= 6.5).astype(int).values
            X_df = df.drop(columns=existing_drop, errors='ignore')
        else:
            X_df = df.iloc[:, :-1]
            y = df.iloc[:, -1].values

        X = X_df.fillna(0).values.astype(np.float32)
        
        feat_dim = X.shape[1]
        if feat_dim % num_clients != 0:
            pad_len = num_clients - (feat_dim % num_clients)
            padding = np.zeros((X.shape[0], pad_len), dtype=np.float32)
            X = np.hstack([X, padding])
            if is_train: print(f"[Dataset] Padding features from {feat_dim} to {X.shape[1]} for {num_clients} clients")
        
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        if is_train:
            self.X = torch.tensor(X_train, dtype=torch.float32)
            self.y = torch.tensor(y_train, dtype=torch.long)
        else:
            self.X = torch.tensor(X_test, dtype=torch.float32)
            self.y = torch.tensor(y_test, dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]
# --- Diabetes 6 (Image Data) ---
class DiabetesDataset6(Dataset):
    def __init__(self, img_dir, label_csv, num_clients, transform=None):
        self.num_clients = num_clients
        self.img_dir = img_dir
        self.transform = transform
        df = pd.read_csv(label_csv)
        if 'Image name' in df.columns: self.img_names = df['Image name'].values
        else: self.img_names = df.iloc[:, 0].values
        if 'Retinopathy grade' in df.columns: grades = df['Retinopathy grade'].values
        else: grades = df.iloc[:, 1].values
        self.labels = (grades > 1).astype(int)
        
        print(f"[Dataset] 正在预处理并缓存图片到内存...")
        self.cached_images = []
        # [极致优化]：在存入内存前就将图片缩小，极大节省内存和 CPU Resize 时间
        base_resize = transforms.Resize((224, 224)) 
        
        for img_name in self.img_names:
            name = str(img_name)
            if not name.lower().endswith(('.jpg', '.jpeg', '.png')): name += '.jpg'
            img_path = os.path.join(self.img_dir, name)
            try: 
                with Image.open(img_path) as img:
                    img_rgb = img.convert('RGB')
                    img_resized = base_resize(img_rgb) # 提前缩小
                    self.cached_images.append(img_resized.copy())
            except FileNotFoundError:
                raise FileNotFoundError(f"Image not found: {img_path}")
        print(f"[Dataset] 成功缓存 {len(self.cached_images)} 张图片！")

    def __len__(self): return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.cached_images[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.transform: image = self.transform(image)
        return image, label