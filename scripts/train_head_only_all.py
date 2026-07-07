"""
④ MOMENT-HeadOnly 全被験者学習（検証なし・デプロイ用モデル保存）
事前学習済み教師チェックポイントをロードし、全データで生徒を学習。
"""
import os, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from ecg_kd.data.loader import load_waveform_samples
from ecg_kd.losses.torch_losses import student_kd_loss
from ecg_kd.models.torch.heads import TeacherRegressionHead, StudentRegressionHead
from ecg_kd.models.torch.datasets import AllSubjectsStudentDataset
from ecg_kd.models.torch.moment_utils import load_moment, get_embedding
from ecg_kd.training.cv_utils import compute_allsubjects_normalization

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--alpha',        type=float, required=True)
parser.add_argument('--teacher_ckpt', type=str,   required=True,
                    help='教師モデルチェックポイントパス (final_model.pt)')
args = parser.parse_args()

EPOCHS     = 100
BATCH_SIZE = 16
LR         = 1e-4
ALPHA      = args.alpha
BETA       = 1.0 - ALPHA
ECG_N      = 19
alpha_str  = f"alpha{int(ALPHA * 10):02d}"

FT_RAW_PATH      = '/mnt/learn/usr/hayashi/引継ぎ/実験データ/2025飲食実験/計測結果/raw_waveform'
FT_FEATURES_PATH = '/mnt/learn/usr/hayashi/引継ぎ/実験データ/2025飲食実験/計測結果/features'
FT_CSV           = '/mnt/learn/usr/hayashi/引継ぎ/result/all_data_filtered_step30s.csv'
RESULT_PATH      = f'/mnt/learn/usr/hayashi/引継ぎ/result/ECG_KD_student_MOMENT_HeadOnly_allsubjects_{alpha_str}/'
os.makedirs(RESULT_PATH, exist_ok=True)

print(f"ALPHA={ALPHA}, Device={DEVICE}")

print("データ読み込み中...")
all_samples = load_waveform_samples(FT_CSV, FT_FEATURES_PATH, FT_RAW_PATH)

mean_ecg, std_ecg = compute_allsubjects_normalization(all_samples)
np.save(os.path.join(RESULT_PATH, 'mean_ecg.npy'), mean_ecg)
np.save(os.path.join(RESULT_PATH, 'std_ecg.npy'),  std_ecg)

print("MOMENT + 教師HeadをロードI中...")
moment_teacher = load_moment(DEVICE, frozen=True)
teacher_head   = TeacherRegressionHead().to(DEVICE)
ckpt = torch.load(args.teacher_ckpt, map_location=DEVICE)
if hasattr(moment_teacher, 'model'):
    moment_teacher.model.load_state_dict(ckpt['moment_state_dict'])
else:
    moment_teacher.load_state_dict(ckpt['moment_state_dict'])
teacher_head.load_state_dict(ckpt['reg_head_state_dict'])
moment_teacher.eval(); teacher_head.eval()
for param in teacher_head.parameters():
    param.requires_grad = False
print('教師④（MOMENT HeadOnly）をロードしました')

student   = StudentRegressionHead(input_dim=ECG_N).to(DEVICE)
optimizer = torch.optim.Adam(student.parameters(), lr=LR)
dataloader = DataLoader(
    AllSubjectsStudentDataset(all_samples, mean_ecg, std_ecg),
    batch_size=BATCH_SIZE, shuffle=True
)

best_loss = float('inf')
print(f"生徒KD学習 {EPOCHS}epoch...")
for epoch in range(1, EPOCHS + 1):
    student.train()
    total_loss = ar_se = val_se = n = 0
    for x_stu, x_tea, y_ar, y_val in dataloader:
        x_stu = x_stu.to(DEVICE); x_tea = x_tea.to(DEVICE)
        y_ar  = y_ar.to(DEVICE);  y_val = y_val.to(DEVICE)
        with torch.no_grad():
            emb    = get_embedding(moment_teacher, x_tea)
            t_ar, t_val = teacher_head(emb)
        s_ar, s_val = student(x_stu)
        loss = student_kd_loss(s_ar, s_val, t_ar, t_val, y_ar, y_val, ALPHA, BETA)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        bs = x_stu.size(0); total_loss += loss.item() * bs
        ar_se  += ((s_ar.detach()  - y_ar)**2).sum().item()
        val_se += ((s_val.detach() - y_val)**2).sum().item(); n += bs

    t_loss     = total_loss / n
    t_rmse_ar  = (ar_se  / n)**0.5
    t_rmse_val = (val_se / n)**0.5
    print(f"Epoch {epoch:3d}/{EPOCHS} | Loss:{t_loss:.4f} | "
          f"RMSE_ar:{t_rmse_ar:.4f} | RMSE_val:{t_rmse_val:.4f}", flush=True)

    if t_loss < best_loss:
        best_loss = t_loss
        torch.save({'student_state_dict': student.state_dict()},
                   os.path.join(RESULT_PATH, 'best_model.pt'))

torch.save({'student_state_dict': student.state_dict()},
           os.path.join(RESULT_PATH, 'final_model.pt'))

result_str = (
    f"Final (Epoch {EPOCHS}):\n"
    f"  Loss={t_loss:.4f} | RMSE_ar={t_rmse_ar:.4f} | RMSE_val={t_rmse_val:.4f}\n"
    f"  Best Loss: {best_loss:.4f}\n"
    f"  ALPHA={ALPHA}, BETA={BETA}\n"
)
print(result_str)
with open(os.path.join(RESULT_PATH, 'summary.txt'), 'w') as f:
    f.write(result_str)
print('学習完了')
