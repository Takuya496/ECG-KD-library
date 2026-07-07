"""
PyTorch Dataset定義（④⑤⑥共通）
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class TeacherEmbeddingDataset(Dataset):
    """④ HeadOnly用: 事前計算済みembeddingから教師を学習"""
    def __init__(self, indices, raw_embeddings, all_samples):
        self.data = [
            (torch.tensor(raw_embeddings[i]),
             torch.tensor(all_samples[i]['y_ar']),
             torch.tensor(all_samples[i]['y_val']))
            for i in indices
        ]

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


class TeacherWaveformDataset(Dataset):
    """⑤⑥ PartialUnfreeze/FullFT用: 生波形をbatch毎にMOMENTへ入力"""
    def __init__(self, indices, all_samples):
        self.data = [
            (torch.tensor(all_samples[i]['x_tea']),
             torch.tensor(all_samples[i]['y_ar']),
             torch.tensor(all_samples[i]['y_val']))
            for i in indices
        ]

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


class StudentDataset(Dataset):
    """④⑤⑥共通: ECG特徴量 + 教師予測キャッシュで生徒を学習"""
    def __init__(self, indices, all_samples, mean_ecg, std_ecg, t_ar_cache, t_val_cache):
        self.data = []
        for idx in indices:
            s = all_samples[idx]
            ecg_feat = (s['ecg_feat_raw'] - mean_ecg) / std_ecg
            self.data.append((
                torch.tensor(ecg_feat),
                torch.tensor(t_ar_cache[idx]),
                torch.tensor(t_val_cache[idx]),
                torch.tensor(s['y_ar']),
                torch.tensor(s['y_val']),
            ))

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


class AllSubjectsStudentDataset(Dataset):
    """allsubjects用: ECG特徴量 + MOMENT生波形（教師予測をonline計算）"""
    def __init__(self, all_samples, mean_ecg, std_ecg):
        self.data = []
        for s in all_samples:
            ecg_feat = (s['ecg_feat_raw'] - mean_ecg) / std_ecg
            self.data.append((
                torch.tensor(ecg_feat),
                torch.tensor(s['x_tea']),
                torch.tensor(s['y_ar']),
                torch.tensor(s['y_val']),
            ))

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]
