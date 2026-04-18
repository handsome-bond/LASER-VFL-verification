from pathlib import Path
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from data.custom_dataset import (CustomDataset, collate_fn)

def get_dataloaders(args, config, p_miss_test=0.0):
    dataset, batch_size, num_workers = config["dataset"], config["batch_size"], config["num_workers"]
    data_dir = Path(__file__).absolute().parent.parent / 'data' / dataset

    train_ld, test_ld = None, None

    # --- Diabetes 1-4 ---
    if dataset == 'diabetes1':
        from data.custom_dataset import DiabetesDataset1
        train_ds = DiabetesDataset1(data_dir / 'Diabetes_Final_Data_V2.csv', args.num_clients, is_train=True)
        test_ds  = DiabetesDataset1(data_dir / 'Diabetes_Final_Data_V2.csv', args.num_clients, is_train=False)
    
    elif dataset == 'diabetes2':
        from data.custom_dataset import DiabetesDataset2
        train_ds = DiabetesDataset2(data_dir / 'diabetes2.csv', args.num_clients, is_train=True)
        test_ds  = DiabetesDataset2(data_dir / 'diabetes2.csv', args.num_clients, is_train=False)
        
    elif dataset == 'diabetes3':
        from data.custom_dataset import DiabetesDataset3
        train_ds = DiabetesDataset3(data_dir / 'diabetes3.csv', args.num_clients, is_train=True)
        test_ds  = DiabetesDataset3(data_dir / 'diabetes3.csv', args.num_clients, is_train=False)
        
    elif dataset == 'diabetes4':
        from data.custom_dataset import DiabetesDataset4
        train_ds = DiabetesDataset4(data_dir / 'diabetes4.csv', args.num_clients, is_train=True)
        test_ds  = DiabetesDataset4(data_dir / 'diabetes4.csv', args.num_clients, is_train=False)

    # --- Diabetes 5 (NHANES) ---
    elif dataset == 'diabetes5':
        from data.custom_dataset import DiabetesDataset5
        csv_path = data_dir / 'nhanes_merged.csv'
        train_ds = DiabetesDataset5(csv_path, args.num_clients, is_train=True)
        test_ds  = DiabetesDataset5(csv_path, args.num_clients, is_train=False)

    # --- Diabetes 6 (IDRiD 图像数据) ---
    elif dataset == 'diabetes6':
        from data.custom_dataset import DiabetesDataset6
        
        # 1. 定义增强和预处理
        # [极致优化] 移除了 Resize，因为在 custom_dataset.py 缓存时已经提前缩放过了
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 测试集只做标准化
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 2. 设置路径
        base_img_path = data_dir / '1. Original Images'
        base_lbl_path = data_dir / '2. Groundtruths'

        # 图片路径
        train_img_dir = base_img_path / 'a. Training Set'
        test_img_dir = base_img_path / 'b. Testing Set'
        
        # CSV 文件名
        train_csv = base_lbl_path / 'a. IDRiD_Disease Grading_Training Labels.csv'
        test_csv = base_lbl_path / 'b. IDRiD_Disease Grading_Testing Labels.csv'

        # 3. 检查文件是否存在
        if not train_img_dir.exists():
            raise FileNotFoundError(f"找不到训练图片目录: {train_img_dir}")
        if not train_csv.exists():
            raise FileNotFoundError(f"找不到训练CSV: {train_csv}")
        if not test_img_dir.exists():
            raise FileNotFoundError(f"找不到测试图片目录: {test_img_dir}")
        if not test_csv.exists():
            raise FileNotFoundError(f"找不到测试CSV: {test_csv}")

        # 4. 初始化 Dataset
        train_ds = DiabetesDataset6(train_img_dir, train_csv, args.num_clients, transform=train_transform)
        test_ds  = DiabetesDataset6(test_img_dir, test_csv, args.num_clients, transform=test_transform)

    if train_ds and test_ds:
        train_ld = create_data_loader(train_ds, batch_size, args.num_clients, args.p_miss_train, 
                                      mechanism=getattr(args, 'mechanism', 'uniform'), 
                                      seed=args.seed,
                                      num_workers=num_workers)
                                      
        test_ld  = create_data_loader(test_ds, batch_size, args.num_clients, p_miss_test, 
                                      mechanism=getattr(args, 'mechanism', 'uniform'), 
                                      seed=args.seed,
                                      num_workers=num_workers)

    return train_ld, test_ld

def create_data_loader(base_dataset, batch_size, num_clients, p_miss, mechanism='uniform', seed=42, num_workers=0, drop_last=False, sampler=None):
    # 传递 mechanism 和 seed
    wrapped_dataset = CustomDataset(base_dataset, batch_size, num_clients, p_miss, mechanism=mechanism, seed=seed)
    
    # [极速优化 2] 加入 pin_memory=True，大幅加速 CPU 锁页内存到 GPU 显存的数据传输
    return DataLoader(wrapped_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, 
                      drop_last=drop_last, sampler=sampler, pin_memory=True)