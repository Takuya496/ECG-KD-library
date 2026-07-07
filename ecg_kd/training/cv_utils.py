"""
交差検証ユーティリティ（フレームワーク非依存）
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def get_fold_splits(all_samples, n_folds=5):
    """GroupKFoldで被験者単位のfold分割インデックスを返す"""
    subjects_arr = np.array([s['subject'] for s in all_samples])
    gkf = GroupKFold(n_splits=n_folds)
    return list(gkf.split(all_samples, groups=subjects_arr))


def compute_fold_normalization(all_samples, train_idx):
    """fold内の訓練データのみからECG特徴量の正規化統計量を計算"""
    train_feats   = np.array([all_samples[i]['ecg_feat_raw'] for i in train_idx])
    mean_ecg_fold = train_feats.mean(axis=0)
    std_ecg_fold  = train_feats.std(axis=0) + 1e-8
    return mean_ecg_fold, std_ecg_fold


def compute_allsubjects_normalization(all_samples):
    """全サンプルからECG特徴量の正規化統計量を計算（allsubjects用）"""
    all_feats = np.array([s['ecg_feat_raw'] for s in all_samples])
    mean_ecg  = all_feats.mean(axis=0)
    std_ecg   = all_feats.std(axis=0) + 1e-8
    return mean_ecg, std_ecg


def save_fold_results(fold_results, result_path, extra_info=''):
    """fold結果をCSV + summary.txtに保存"""
    mean_ar  = float(np.mean([r['rmse_ar']  for r in fold_results]))
    mean_val = float(np.mean([r['rmse_val'] for r in fold_results]))
    std_ar   = float(np.std( [r['rmse_ar']  for r in fold_results]))
    std_val  = float(np.std( [r['rmse_val'] for r in fold_results]))

    pd.DataFrame(fold_results).to_csv(os.path.join(result_path, 'fold_results.csv'), index=False)
    result_str = (
        f"Nested 5-fold CV結果:\n"
        f"  RMSE_ar:  {mean_ar:.4f} +/- {std_ar:.4f}\n"
        f"  RMSE_val: {mean_val:.4f} +/- {std_val:.4f}\n"
    )
    if extra_info:
        result_str += extra_info
    print(result_str)
    with open(os.path.join(result_path, 'summary.txt'), 'w') as f:
        f.write(result_str)
    return mean_ar, mean_val


def load_checkpoint(result_path):
    """チェックポイントCSVがあれば読み込んで完了済みfoldを返す"""
    checkpoint_csv = os.path.join(result_path, 'fold_results_checkpoint.csv')
    if os.path.exists(checkpoint_csv):
        checkpoint_df   = pd.read_csv(checkpoint_csv)
        fold_results    = checkpoint_df.to_dict('records')
        completed_folds = set(checkpoint_df['fold'].tolist())
        print(f"チェックポイント発見: fold {sorted(completed_folds)} は完了済み、スキップします")
        return fold_results, completed_folds
    return [], set()


def save_checkpoint(fold_results, result_path):
    """途中結果をチェックポイントCSVに保存"""
    checkpoint_csv = os.path.join(result_path, 'fold_results_checkpoint.csv')
    pd.DataFrame(fold_results).to_csv(checkpoint_csv, index=False)
    print(f"  チェックポイント保存: {checkpoint_csv}")
