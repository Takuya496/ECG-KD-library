"""
⑤ MOMENT-PartialUnfreeze Nested 5-fold GroupKFold 交差検証（評価用）
fold毎にMOMENT後ろ6ブロック+HeadをFT後、生徒KD学習。
"""
import os, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from ecg_kd.data.loader import load_waveform_samples
from ecg_kd.losses.torch_losses import teacher_loss, student_kd_loss
from ecg_kd.models.torch.heads import TeacherRegressionHead, StudentRegressionHead
from ecg_kd.models.torch.datasets import TeacherWaveformDataset, StudentDataset
from ecg_kd.models.torch.moment_utils import (
    load_moment, unfreeze_last_n_blocks,
    compute_teacher_preds_from_waveform
)
from ecg_kd.training.cv_utils import (
    get_fold_splits, compute_fold_normalization,
    save_fold_results, load_checkpoint, save_checkpoint
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--alpha',          type=float, default=0.5)
parser.add_argument('--teacher_epochs', type=int,   default=100)
parser.add_argument('--unfreeze_n',     type=int,   default=6)
args = parser.parse_args()

EPOCHS          = 100
TEACHER_EPOCHS  = args.teacher_epochs
BATCH_SIZE      = 16
TEACHER_BATCH   = 16
LR              = 1e-4
ALPHA           = args.alpha
BETA            = 1.0 - ALPHA
ECG_N           = 19
N_FOLDS         = 5
UNFREEZE_LAST_N = args.unfreeze_n
alpha_str       = f"alpha{int(ALPHA * 10):02d}"

FT_RAW_PATH      = '/mnt/learn/usr/hayashi/引継ぎ/実験データ/2025飲食実験/計測結果/raw_waveform'
FT_FEATURES_PATH = '/mnt/learn/usr/hayashi/引継ぎ/実験データ/2025飲食実験/計測結果/features'
FT_CSV           = '/mnt/learn/usr/hayashi/引継ぎ/result/all_data_filtered_step30s.csv'
RESULT_PATH      = f'/mnt/learn/usr/hayashi/引継ぎ/result/ECG_KD_student_MOMENT_PartialUnfreeze_nested5fold_{alpha_str}/'
os.makedirs(RESULT_PATH, exist_ok=True)

print(f"ALPHA={ALPHA}, TEACHER_EPOCHS={TEACHER_EPOCHS}, UNFREEZE_LAST_N={UNFREEZE_LAST_N}, Device={DEVICE}")

print("データ読み込み中...")
all_samples = load_waveform_samples(FT_CSV, FT_FEATURES_PATH, FT_RAW_PATH)

fold_splits             = get_fold_splits(all_samples, N_FOLDS)
fold_results, completed = load_checkpoint(RESULT_PATH)

for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
    if (fold_idx + 1) in completed:
        print(f"\n=== Fold {fold_idx + 1} スキップ（完了済み）===")
        continue

    train_subjs = sorted(set(all_samples[i]['subject'] for i in train_idx))
    test_subjs  = sorted(set(all_samples[i]['subject'] for i in test_idx))
    print(f"\n=== Fold {fold_idx + 1}/{N_FOLDS} | "
          f"train:{len(train_subjs)}人 {len(train_idx)}サンプル / "
          f"test:{len(test_subjs)}人 {len(test_idx)}サンプル ===")

    mean_ecg, std_ecg = compute_fold_normalization(all_samples, train_idx)

    # ---- Phase1: MOMENT後ろnブロック + HeadをFT ----
    print(f"  [Phase1] MOMENT pretrained をロード中...")
    moment_ft = load_moment(DEVICE, frozen=False)
    unfreeze_last_n_blocks(moment_ft, UNFREEZE_LAST_N)

    teacher_head      = TeacherRegressionHead().to(DEVICE)
    ft_params         = [p for p in moment_ft.parameters() if p.requires_grad] + \
                        list(teacher_head.parameters())
    teacher_optimizer = torch.optim.Adam(ft_params, lr=LR)
    teacher_loader    = DataLoader(
        TeacherWaveformDataset(list(train_idx), all_samples),
        batch_size=TEACHER_BATCH, shuffle=True
    )

    print(f"  [Phase1] MOMENT PartialUnfreeze + TeacherHeadを{TEACHER_EPOCHS}epoch学習中...")
    for epoch in range(1, TEACHER_EPOCHS + 1):
        moment_ft.train(); teacher_head.train()
        total = n = 0
        for x_tea_b, y_ar_b, y_val_b in teacher_loader:
            x_tea_b = x_tea_b.to(DEVICE); y_ar_b = y_ar_b.to(DEVICE); y_val_b = y_val_b.to(DEVICE)
            out = moment_ft(x_enc=x_tea_b)
            emb = out.embeddings
            if emb.dim() == 3: emb = emb.mean(dim=1)
            t_ar, t_val = teacher_head(emb)
            loss = teacher_loss(t_ar, t_val, y_ar_b, y_val_b)
            teacher_optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ft_params, 1.0)
            teacher_optimizer.step()
            total += loss.item() * x_tea_b.size(0); n += x_tea_b.size(0)
        if epoch % 20 == 0:
            print(f"    Teacher Ep{epoch:3d} | Loss:{total/n:.4f}")

    moment_ft.eval(); teacher_head.eval()
    all_indices = list(train_idx) + list(test_idx)
    t_ar_fold, t_val_fold = compute_teacher_preds_from_waveform(
        moment_ft, teacher_head, all_indices, all_samples, DEVICE
    )
    del moment_ft, teacher_head, teacher_optimizer
    torch.cuda.empty_cache()

    # ---- Phase2: 生徒KD学習 ----
    student   = StudentRegressionHead(input_dim=ECG_N).to(DEVICE)
    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    train_loader = DataLoader(
        StudentDataset(list(train_idx), all_samples, mean_ecg, std_ecg, t_ar_fold, t_val_fold),
        batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        StudentDataset(list(test_idx), all_samples, mean_ecg, std_ecg, t_ar_fold, t_val_fold),
        batch_size=BATCH_SIZE, shuffle=False
    )

    print(f"  [Phase2] 生徒KD学習 {EPOCHS}epoch...")
    for epoch in range(1, EPOCHS + 1):
        student.train()
        total_loss = ar_se = val_se = n = 0
        for x_stu, t_ar_b, t_val_b, y_ar_b, y_val_b in train_loader:
            x_stu   = x_stu.to(DEVICE);   t_ar_b  = t_ar_b.to(DEVICE)
            t_val_b = t_val_b.to(DEVICE); y_ar_b  = y_ar_b.to(DEVICE)
            y_val_b = y_val_b.to(DEVICE)
            s_ar, s_val = student(x_stu)
            loss = student_kd_loss(s_ar, s_val, t_ar_b, t_val_b, y_ar_b, y_val_b, ALPHA, BETA)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            bs = x_stu.size(0); total_loss += loss.item() * bs
            ar_se  += ((s_ar.detach()  - y_ar_b)**2).sum().item()
            val_se += ((s_val.detach() - y_val_b)**2).sum().item(); n += bs
        if epoch % 10 == 0:
            print(f"    Student Ep{epoch:3d} | Loss:{total_loss/n:.4f} | "
                  f"RMSE_ar:{(ar_se/n)**0.5:.4f} | RMSE_val:{(val_se/n)**0.5:.4f}")

    student.eval()
    ar_se = val_se = n = 0
    with torch.no_grad():
        for x_stu, _, _, y_ar_b, y_val_b in test_loader:
            x_stu, y_ar_b, y_val_b = x_stu.to(DEVICE), y_ar_b.to(DEVICE), y_val_b.to(DEVICE)
            s_ar, s_val = student(x_stu)
            ar_se  += ((s_ar  - y_ar_b)**2).sum().item()
            val_se += ((s_val - y_val_b)**2).sum().item(); n += x_stu.size(0)

    test_rmse_ar  = float((ar_se  / n)**0.5)
    test_rmse_val = float((val_se / n)**0.5)
    print(f"  Test RMSE_ar:{test_rmse_ar:.4f} | RMSE_val:{test_rmse_val:.4f}")
    fold_results.append({'fold': fold_idx + 1, 'test_subjects': str(test_subjs),
                         'rmse_ar': test_rmse_ar, 'rmse_val': test_rmse_val})
    save_checkpoint(fold_results, RESULT_PATH)

save_fold_results(fold_results, RESULT_PATH,
                  f"  ALPHA={ALPHA}, BETA={BETA}, TEACHER_EPOCHS={TEACHER_EPOCHS}, "
                  f"UNFREEZE_LAST_N={UNFREEZE_LAST_N}\n")
print('Nested 5-fold CV完了')
