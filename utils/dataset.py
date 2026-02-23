
# utils/dataset.py
import torch
from torch.utils.data import Dataset

class FinancialDataset(Dataset):
    def __init__(self, df, feature_cols, target_cols, use_minute_features=True):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols.copy()
        self.target_cols = target_cols
        
        if use_minute_features:
            self.feature_cols += ["minute_sin", "minute_cos"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(row[self.feature_cols].values, dtype=torch.float32)
        if self.target_cols:
            y = torch.tensor(row[self.target_cols].values, dtype=torch.float32)
            return x, y
        return x  # 用于测试集