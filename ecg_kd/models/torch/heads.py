"""
PyTorchモデル定義（④⑤⑥共通）
"""
import torch.nn as nn
import torch.nn.functional as F


class TeacherRegressionHead(nn.Module):
    """MOMENT embedding(1024次元) → Arousal/Valence"""
    def __init__(self, in_dim=1024, hidden1=256, hidden2=64):
        super().__init__()
        self.fc1     = nn.Linear(in_dim, hidden1)
        self.act1    = nn.ELU()
        self.drop    = nn.Dropout(0.3)
        self.fc2     = nn.Linear(hidden1, hidden2)
        self.act2    = nn.ELU()
        self.out_ar  = nn.Linear(hidden2, 1)
        self.out_val = nn.Linear(hidden2, 1)

    def forward(self, z):
        h = self.act1(self.fc1(z))
        h = self.drop(h)
        h = self.act2(self.fc2(h))
        return self.out_ar(h).squeeze(-1), self.out_val(h).squeeze(-1)


class StudentRegressionHead(nn.Module):
    """ECG特徴量(19次元) → Arousal/Valence"""
    def __init__(self, input_dim=19, hidden_dim=64):
        super().__init__()
        self.hidden  = nn.Linear(input_dim, hidden_dim)
        self.out_ar  = nn.Linear(hidden_dim, 1)
        self.out_val = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.elu(self.hidden(x))
        return self.out_ar(h).squeeze(-1), self.out_val(h).squeeze(-1)
