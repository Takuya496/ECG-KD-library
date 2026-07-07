"""
③ KD-E2E 全被験者学習（検証なし・デプロイ用モデル保存）
事前学習済み教師チェックポイントをロードし、全データで生徒を学習。
"""
import sys, os
sys.path.insert(0, '/mnt/learn/usr/hayashi/引継ぎ/プログラム/EmotionRecognition')
import tensorflow as tf
from Algorithms.Models.EnsembleFeaturesModel import SingleModel
from Algorithms.Models.Losses import PCCLoss, CCCLoss
from Conf.Settings import N_CLASS, ECG_N, PPG_N
import numpy as np

from ecg_kd.data.loader import load_feature_samples
from ecg_kd.models.tf.heads import RegressionHead

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cross_tower_ops = tf.distribute.HierarchicalCopyAllReduce(num_packs=1)
strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_tower_ops)

EPOCHS         = 100
BATCH_SIZE     = 32
ALL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
LR             = 1e-4
ALPHA          = 0.5
BETA           = 0.5
p, q, r        = 1/3, 1/3, 1/3

FT_FEATURES_PATH   = '/mnt/learn/usr/hayashi/引継ぎ/実験データ/2025飲食実験/計測結果/features'
FT_CSV             = os.path.join(FT_FEATURES_PATH, 'dataset', 'all_data_filtered_step30s.csv')
TEACHER3_STAT_PATH = '/mnt/learn/usr/hayashi/引継ぎ/result/FT_KD_endtoend_allsubjects/'
TEACHER3_CKPT      = '/mnt/learn/usr/hayashi/引継ぎ/result/FT_KD_endtoend_allsubjects/model_FT_KD_endtoend'
RESULT_PATH        = '/mnt/learn/usr/hayashi/引継ぎ/result/ECG_KD_student_endtoend_allsubjects/'
os.makedirs(RESULT_PATH, exist_ok=True)

mean_ecg = np.load(os.path.join(TEACHER3_STAT_PATH, 'mean_ecg.npy'))
std_ecg  = np.load(os.path.join(TEACHER3_STAT_PATH, 'std_ecg.npy'))
mean_ppg = np.load(os.path.join(TEACHER3_STAT_PATH, 'mean_ppg.npy'))
std_ppg  = np.load(os.path.join(TEACHER3_STAT_PATH, 'std_ppg.npy'))
np.save(os.path.join(RESULT_PATH, 'mean_ecg.npy'), mean_ecg)
np.save(os.path.join(RESULT_PATH, 'std_ecg.npy'),  std_ecg)

print("データ読み込み中...")
all_samples = load_feature_samples(FT_CSV, FT_FEATURES_PATH)

all_data = []
for s in all_samples:
    ecg = (s['ecg_raw'] - mean_ecg) / std_ecg
    ppg = (s['ppg_raw'] - mean_ppg) / std_ppg
    all_data.append((ecg, np.concatenate([ecg, ppg]), s['y_ar'], s['y_val']))

train_dataset = tf.data.Dataset.from_generator(
    lambda: iter(all_data),
    output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
    output_shapes=(tf.TensorShape([ECG_N]), tf.TensorShape([ECG_N + PPG_N]), (), ())
).shuffle(len(all_data), reshuffle_each_iteration=True).batch(ALL_BATCH_SIZE)

with strategy.scope():
    m1 = SingleModel(num_output=N_CLASS); m2 = SingleModel(num_output=N_CLASS)
    m3 = SingleModel(num_output=N_CLASS); m4 = SingleModel(num_output=N_CLASS)
    m5 = SingleModel(num_output=N_CLASS)
    dummy = tf.zeros([1, ECG_N + PPG_N])
    for m in [m1, m2, m3, m4, m5]:
        _ = m(dummy)
    _tmp_reg_head = RegressionHead(hidden_units=64, name='_tmp_reg_head')
    _ = _tmp_reg_head(tf.zeros([1, 32]))
    _init_ckpt = tf.train.Checkpoint(
        step=tf.Variable(1), reg_head=_tmp_reg_head,
        m1=m1, m2=m2, m3=m3, m4=m4, m5=m5
    )
    _init_ckpt.restore(tf.train.latest_checkpoint(TEACHER3_CKPT)).expect_partial()
    for m in [m1, m2, m3, m4, m5]:
        m.trainable = False
    del _tmp_reg_head, _init_ckpt
    kd_models = [m1, m2, m3, m4, m5]

    teacher_reg_head = RegressionHead(hidden_units=64, name='teacher_reg_head')
    _ = teacher_reg_head(tf.zeros([1, 32]))
    teacher_ckpt = tf.train.Checkpoint(step=tf.Variable(1), reg_head=teacher_reg_head)
    teacher_ckpt.restore(tf.train.latest_checkpoint(TEACHER3_CKPT)).expect_partial()
    teacher_reg_head.trainable = False
    print('教師③（KD E2E）をロードしました')

    student    = RegressionHead(hidden_units=64, name='student')
    optimizer  = tf.keras.optimizers.Adam(learning_rate=LR)
    mse_loss_fn = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    pcc_loss_fn = PCCLoss(reduction=tf.keras.losses.Reduction.NONE)
    ccc_loss_fn = CCCLoss(reduction=tf.keras.losses.Reduction.NONE)
    rmse_ar_m  = tf.keras.metrics.RootMeanSquaredError()
    rmse_val_m = tf.keras.metrics.RootMeanSquaredError()
    loss_m     = tf.keras.metrics.Mean()
    _ = student(tf.zeros([1, ECG_N]))
    student_vars = student.trainable_variables

checkpoint_prefix = os.path.join(RESULT_PATH, 'model_ECG_KD_endtoend')
ckpt    = tf.train.Checkpoint(step=tf.Variable(1), student=student)
manager = tf.train.CheckpointManager(ckpt, checkpoint_prefix, max_to_keep=1)

with strategy.scope():
    def train_step(inputs, GLOBAL_BATCH_SIZE):
        x_stu   = inputs[0]; x_tea   = inputs[1]
        y_r_ar  = tf.expand_dims(inputs[2], -1)
        y_r_val = tf.expand_dims(inputs[3], -1)
        embeddings = [m(x_tea, training=False)[3] for m in kd_models]
        avg_z = tf.reduce_mean(tf.stack(embeddings, axis=0), axis=0)
        t_ar, t_val = teacher_reg_head(avg_z, training=False)
        with tf.GradientTape() as tape:
            s_ar, s_val = student(x_stu, training=True)
            kd = tf.nn.compute_average_loss(
                0.5 * (mse_loss_fn(t_ar, s_ar) + mse_loss_fn(t_val, s_val)),
                global_batch_size=GLOBAL_BATCH_SIZE)
            t_mse = tf.nn.compute_average_loss(
                0.5 * (mse_loss_fn(y_r_ar, s_ar) + mse_loss_fn(y_r_val, s_val)),
                global_batch_size=GLOBAL_BATCH_SIZE)
            t_pcc = tf.nn.compute_average_loss(
                1.0 - 0.5 * (pcc_loss_fn(y_r_ar, s_ar) + pcc_loss_fn(y_r_val, s_val)),
                global_batch_size=1)
            t_ccc = tf.nn.compute_average_loss(
                1.0 - 0.5 * (ccc_loss_fn(y_r_ar, s_ar) + ccc_loss_fn(y_r_val, s_val)),
                global_batch_size=1)
            final_loss = ALPHA * kd + BETA * (p * t_mse + q * t_pcc + r * t_ccc)
        grads = tape.gradient(final_loss, student_vars)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, student_vars))
        loss_m(final_loss); rmse_ar_m(y_r_ar, s_ar); rmse_val_m(y_r_val, s_val)
        return final_loss

    @tf.function
    def distributed_train_step(dataset_inputs, GLOBAL_BATCH_SIZE):
        per_replica_losses = strategy.run(train_step, args=(dataset_inputs, GLOBAL_BATCH_SIZE))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    for epoch in range(EPOCHS):
        for batch in train_dataset:
            distributed_train_step(batch, ALL_BATCH_SIZE)
        t_loss     = loss_m.result().numpy()
        t_rmse_ar  = rmse_ar_m.result().numpy()
        t_rmse_val = rmse_val_m.result().numpy()
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss:{t_loss:.4f} | "
              f"RMSE_ar:{t_rmse_ar:.4f} | RMSE_val:{t_rmse_val:.4f}")
        manager.save()
        for met in [loss_m, rmse_ar_m, rmse_val_m]:
            met.reset_states()

result_str = (
    f"Final (Epoch {EPOCHS}):\n"
    f"  Loss={t_loss:.4f} | RMSE_ar={t_rmse_ar:.4f} | RMSE_val={t_rmse_val:.4f}\n"
)
print(result_str)
with open(os.path.join(RESULT_PATH, 'summary.txt'), 'w') as f:
    f.write(result_str)
print('学習完了')
