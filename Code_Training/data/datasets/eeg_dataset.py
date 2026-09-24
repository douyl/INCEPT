# Code_Training/data/datasets/eeg_dataset.py

import torch
import numpy as np
import os
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root
        self.file_paths = []

        print(f"Scanning files in {data_root}... This might take a minute.")
        
        # Walk through the directory tree to find all .npy files
        # Structure: data_root/000/record_folder/segment_x.npy
        for root, _, files in os.walk(data_root):
            for file in files:
                if file.endswith('.npy'):
                    self.file_paths.append(os.path.join(root, file))

        print(f"Dataset loaded: Found {len(self.file_paths)} files.")

    def __len__(self):
        return len(self.file_paths)

    def _normalize_per_channel(self, x):
        """
        Normalize each channel independently.
        Input x shape: (C, N, T) -> (19, 30, 250)
        """
        # Calculate mean/std along (N, T) dimensions (axis 1 and 2)
        mean = x.mean(axis=(1, 2), keepdims=True)
        std = x.std(axis=(1, 2), keepdims=True)
        
        # Avoid division by zero
        return np.divide(x - mean, std, out=np.zeros_like(x), where=std > 1e-8)

    def __getitem__(self, index):
        path = self.file_paths[index]
        
        # 1. Load Data
        # Shape: (C, N, T) -> (19, 30, 250)
        data_np = np.load(path)

        # 2. Per-Channel Normalization
        data = self._normalize_per_channel(data_np)
        data = torch.from_numpy(data).float()
         
        # 3. Permute for model input
        # Current: (C, N, T)
        # Target: (T, C, N) -> (250, 19, 30)
        data = data.permute(2, 0, 1) 

        return {"data": data}