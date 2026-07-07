"""
② KD-Ensemble Nested 5-fold GroupKFold 交差検証（評価用）
KD SingleModels×5は固定。fold毎にTeacherHeadを再学習後、生徒KD学習。
"""
import sys, os, argparse
sys.path.insert(0, '/mnt/learn/usr/hayashi/引継ぎ/プログラム/EmotionRecognition')
import tensorflow as tf
from Algorithms.Models.EnsembleFeaturesModel import SingleModel
from Algorithms.Models.Losses import PCCLoss, CCCLoss
from Conf.Settings import N_CLASS, ECG_N, PPG_N
import numpy as np
import pandas as pd

from ecg_kd.data.loader import load_feature_samples
from ecg_kd.models.tf.heads import RegressionHead
from ecg_kd.training.cv_utils import get_fold_splits, save_fold_results

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser()
parser.add_argument('--alpha',          type=float, default=0.5)
parser.add_argument('--teacher_epochs', type=int,   default=100)
args = parser.parse_args()

EPOCHS         = 100
TEACHER_EPOCHS = args.teacher_epochs
BATCH_SIZE     = 32
LR             = 1e-4
ALPHA          = args.alpha
BETA           = 1.0 - ALPHA
p, q, r        = 1/3, 1/3, 1/3
N_FOLDS        = 5
alpha_str      = f"alpha{int(ALPHA * 10):02d}"

FT_FEATURES_PATH = '/mnt/learn/usr/hayashi/引継ぎ/実験データ/2025飲食実験/計測結果/features'
FT_CSV           = os.path.join(FT_FEATURES_PATH, 'dataset', 'all_data_filtered_step30s.csv')
KD_RESULT_PATH   = '/mnt/learn/usr/hayashi/引継ぎ/result/KD_ECG_PPG/'
RESULT_PATH      = f'/mnt/learn/usr/hayashi/引継ぎ/result/ECG_KD_student_ensemble_nested5fold_{alpha_str}/'
os.makedirs(RESULT_PATH, exist_ok=True)

print(f"ALPHA={ALPHA}, TEACHER_EPOCHS={TEACHER_EPOCHS}")

print("データ読み込み中...")
all_samples = load_feature_samples(FT_CSV, FT_FEATURES_PATH)

kd_models = []
for fold in range(1, 6):
    ckpt_prefix = os.path.join(KD_RESULT_PATH, f'fold_{fold}', 'model_student_ECG_PPG_KD')
    m = SingleModel(num_output=N_CLASS).loadBaseModel(ckpt_prefix)
    m.trainable = False
    kd_models.append(m)
print('KD SingleModels（5個）をロードしました（固定）')

mse_loss_fn = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
pcc_loss_fn = PCCLoss(reduction=tf.keras.losses.Reduction.NONE)
ccc_loss_fn = CCCLoss(reduction=tf.keras.losses.Reduction.NONE)

teacher_reg_head  = RegressionHead(hidden_units=64, name='teacher_reg_head')
_                 = teacher_reg_head(tf.zeros([1, 32]))
teacher_optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

student   = RegressionHead(hidden_units=64, name='student')
_         = student(tf.zeros([1, ECG_N]))
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)


@tf.function
def teacher_train_step(x_tea, y_ar_b, y_val_b):
    y_r_ar  = tf.expand_dims(y_ar_b, -1)
    y_r_val = tf.expand_dims(y_val_b, -1)
    embeddings = [m(x_tea, training=False)[3] for m in kd_models]
    avg_z = tf.reduce_mean(tf.stack(embeddings, axis=0), axis=0)
    with tf.GradientTape() as tape:
        t_ar, t_val = teacher_reg_head(avg_z, training=True)
        mse_t = tf.reduce_mean(0.5 * (mse_loss_fn(y_r_ar, t_ar) + mse_loss_fn(y_r_val, t_val)))
        pcc_t = 1.0 - 0.5 * (pcc_loss_fn(y_r_ar, t_ar) + pcc_loss_fn(y_r_val, t_val))
        ccc_t = 1.0 - 0.5 * (ccc_loss_fn(y_r_ar, t_ar) + ccc_loss_fn(y_r_val, t_val))
        loss  = p * mse_t + q * pcc_t + r * ccc_t
    grads, _ = tf.clip_by_global_norm(tape.gradient(loss, teacher_reg_head.trainable_variables), 1.0)
    teacher_optimizer.apply_gradients(zip(grads, teacher_reg_head.trainable_variables))
    return loss


@tf.function
def student_train_step(x_stu, x_tea, y_ar_b, y_val_b):
    y_r_ar  = tf.expand_dims(y_ar_b, -1)
    y_r_val = tf.expand_dims(y_val_b, -1)
    embeddings = [m(x_tea, training=False)[3] for m in kd_models]
    avg_z = tf.reduce_mean(tf.stack(embeddings, axis=0), axis=0)
    t_ar, t_val = teacher_reg_head(avg_z, training=False)
    with tf.GradientTape() as tape:
        s_ar, s_val = student(x_stu, training=True)
        kd    = tf.reduce_mean(0.5 * (mse_loss_fn(t_ar, s_ar) + mse_loss_fn(t_val, s_val)))
        t_mse = tf.reduce_mean(0.5 * (mse_loss_fn(y_r_ar, s_ar) + mse_loss_fn(y_r_val, s_val)))
        t_pcc = 1.0 - 0.5 * (pcc_loss_fn(y_r_ar, s_ar) + pcc_loss_fn(y_r_val, s_val))
        t_ccc = 1.0 - 0.5 * (ccc_loss_fn(y_r_ar, s_ar) + ccc_loss_fn(y_r_val, s_val))
        final_loss = ALPHA * kd + BETA * (p * t_mse + q * t_pcc + r * t_ccc)
    grads, _ = tf.clip_by_global_norm(tape.gradient(final_loss, student.trainable_variables), 1.0)
    optimizer.apply_gradients(zip(grads, student.trainable_variables))
    return final_loss, s_ar, s_val, y_r_ar, y_r_val


glorot       = tf.keras.initializers.GlorotUniform()
fold_splits  = get_fold_splits(all_samples, N_FOLDS)
fold_results = []

for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
    train_samples = [all_samples[i] for i in train_idx]
    test_samples  = [all_samples[i] for i in test_idx]
    train_subjs   = sorted(set(s['subject'] for s in train_samples))
    test_subjs    = sorted(set(s['subject'] for s in test_samples))
    print(f"\n=== Fold {fold_idx + 1}/{N_FOLDS} | "
          f"train:{len(train_subjs)}人 {len(train_samples)}サンプル / "
          f"test:{len(test_subjs)}人 {len(test_samples)}サンプル ===")

    train_ecg_arr = np.array([s['ecg_raw'] for s in train_samples])
    train_ppg_arr = np.array([s['ppg_raw'] for s in train_samples])
    mean_ecg_f = train_ecg_arr.mean(axis=0); std_ecg_f  = train_ecg_arr.std(axis=0) + 1e-8
    mean_ppg_f = train_ppg_arr.mean(axis=0); std_ppg_f  = train_ppg_arr.std(axis=0) + 1e-8

    def to_ecg(s):
        return (s['ecg_raw'] - mean_ecg_f) / std_ecg_f

    def to_ecg_ppg(s):
        ecg = (s['ecg_raw'] - mean_ecg_f) / std_ecg_f
        ppg = (s['ppg_raw'] - mean_ppg_f) / std_ppg_f
        return np.concatenate([ecg, ppg])

    teacher_dataset = tf.data.Dataset.from_tensor_slices((
        np.array([to_ecg_ppg(s) for s in train_samples]),
        np.array([s['y_ar']  for s in train_samples]),
        np.array([s['y_val'] for s in train_samples]),
    )).shuffle(len(train_samples)).batch(BATCH_SIZE)

    train_dataset = tf.data.Dataset.from_tensor_slices((
        np.array([to_ecg(s)     for s in train_samples]),
        np.array([to_ecg_ppg(s) for s in train_samples]),
        np.array([s['y_ar']  for s in train_samples]),
        np.array([s['y_val'] for s in train_samples]),
    )).shuffle(len(train_samples)).batch(BATCH_SIZE)

    # ---- Phase1: TeacherHeadをリセット＆再学習 ----
    for w in teacher_reg_head.weights:
        w.assign(glorot(shape=w.shape, dtype=w.dtype))
    for var in teacher_optimizer.variables():
        var.assign(tf.zeros_like(var))
    teacher_loss_m = tf.keras.metrics.Mean()

    print(f"  [Phase1] 教師reg_headを{TEACHER_EPOCHS}epoch再学習中...")
    for epoch in range(TEACHER_EPOCHS):
        for x_tea, y_ar_b, y_val_b in teacher_dataset:
            teacher_loss_m(teacher_train_step(x_tea, y_ar_b, y_val_b))
        if (epoch + 1) % 20 == 0:
            print(f"    Teacher Ep{epoch + 1:3d} | Loss:{teacher_loss_m.result():.4f}")
        teacher_loss_m.reset_states()

    # ---- Phase2: 生徒KD学習 ----
    for w in student.weights:
        w.assign(glorot(shape=w.shape, dtype=w.dtype))
    for var in optimizer.variables():
        var.assign(tf.zeros_like(var))
    loss_m     = tf.keras.metrics.Mean()
    rmse_ar_m  = tf.keras.metrics.RootMeanSquaredError()
    rmse_val_m = tf.keras.metrics.RootMeanSquaredError()

    print(f"  [Phase2] 生徒KD学習 {EPOCHS}epoch...")
    for epoch in range(EPOCHS):
        for x_stu, x_tea, y_ar_b, y_val_b in train_dataset:
            final_loss, s_ar, s_val, y_r_ar, y_r_val = student_train_step(x_stu, x_tea, y_ar_b, y_val_b)
            loss_m(final_loss); rmse_ar_m(y_r_ar, s_ar); rmse_val_m(y_r_val, s_val)
        if (epoch + 1) % 10 == 0:
            print(f"    Student Ep{epoch + 1:3d} | Loss:{loss_m.result():.4f} | "
                  f"RMSE_ar:{rmse_ar_m.result():.4f} | RMSE_val:{rmse_val_m.result():.4f}")
        for met in [loss_m, rmse_ar_m, rmse_val_m]:
            met.reset_states()

    x_stu_test = tf.constant(np.array([to_ecg(s) for s in test_samples]))
    y_ar_test  = np.array([s['y_ar']  for s in test_samples])
    y_val_test = np.array([s['y_val'] for s in test_samples])
    s_ar_out, s_val_out = student(x_stu_test, training=False)
    test_rmse_ar  = float(np.sqrt(np.mean((s_ar_out.numpy().flatten()  - y_ar_test)  **2)))
    test_rmse_val = float(np.sqrt(np.mean((s_val_out.numpy().flatten() - y_val_test) **2)))
    print(f"  Test RMSE_ar:{test_rmse_ar:.4f} | RMSE_val:{test_rmse_val:.4f}")
    fold_results.append({'fold': fold_idx + 1, 'test_subjects': str(test_subjs),
                         'rmse_ar': test_rmse_ar, 'rmse_val': test_rmse_val})

save_fold_results(fold_results, RESULT_PATH,
                  f"  ALPHA={ALPHA}, BETA={BETA}, TEACHER_EPOCHS={TEACHER_EPOCHS}\n")
print('Nested 5-fold CV完了')
