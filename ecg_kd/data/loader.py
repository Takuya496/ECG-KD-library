"""
データ読み込みユーティリティ（フレームワーク非依存）
②③: load_feature_samples  — .npy特徴量のみ使用
④⑤⑥: load_waveform_samples — 生波形CSV + .npy特徴量を使用
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import resample as scipy_resample


def load_waveform(subject, modality, data_path):
    pattern = os.path.join(data_path, str(subject), modality,
                           f"filtered_{subject}_*_{modality}.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    df = pd.read_csv(files[0])
    ts = pd.to_datetime(df['Timestamp'])
    t0 = ts.iloc[0].replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed_sec = (ts - t0).dt.total_seconds().values.astype(np.float64)
    signal = df[modality].values.astype(np.float32)
    return elapsed_sec, signal


def extract_segment(elapsed_sec, signal, start_sec, end_sec):
    mask = (elapsed_sec >= start_sec) & (elapsed_sec < end_sec)
    seg = signal[mask]
    return seg if len(seg) >= 100 else None


def zscore(seg):
    mu, sigma = seg.mean(), seg.std()
    return (seg - mu) / (sigma + 1e-8)


def load_feature_samples(csv_path, feat_path):
    """
    ②③用: CSV + .npy特徴量からサンプルをロード
    Returns: list of dict {subject, ecg_raw, ppg_raw, y_ar, y_val}
    """
    all_df  = pd.read_csv(csv_path)
    samples = []
    for i in range(len(all_df)):
        row     = all_df.iloc[i]
        idx     = int(row['Idx'])
        subject = str(int(row['Subject']))
        ecg_path = os.path.join(feat_path, subject, 'ECG', '30s', f'ecg_{idx}.npy')
        ppg_path = os.path.join(feat_path, subject, 'PPG', '30s', f'ppg_{idx}.npy')
        if not os.path.exists(ecg_path) or not os.path.exists(ppg_path):
            continue
        ecg_raw = np.load(ecg_path).astype(np.float32)
        ppg_raw = np.load(ppg_path).astype(np.float32)
        y_ar    = np.float32(row['Arousal'])
        y_val   = np.float32(row['Valence'])
        if not (np.isfinite(ecg_raw).all() and np.isfinite(ppg_raw).all() and
                np.isfinite(y_ar) and np.isfinite(y_val)):
            continue
        samples.append({
            'subject': int(row['Subject']),
            'ecg_raw': ecg_raw,
            'ppg_raw': ppg_raw,
            'y_ar':    y_ar,
            'y_val':   y_val,
        })
    print(f"有効サンプル数: {len(samples)}")
    return samples


def load_waveform_samples(csv_path, feat_path, raw_path, moment_len=512):
    """
    ④⑤⑥用: CSV + .npy特徴量 + 生波形CSVからサンプルをロード
    Returns: list of dict {subject, ecg_feat_raw, x_tea, y_ar, y_val}
      ecg_feat_raw: 19次元ECG特徴量（正規化前）
      x_tea:        (2, 512) — ECG+PPG resample済み・zscore済み（MOMENT入力用）
    """
    all_df   = pd.read_csv(csv_path)
    wf_cache = {}
    samples  = []

    for i in range(len(all_df)):
        row     = all_df.iloc[i]
        subject = str(int(row['Subject']))
        idx     = int(row['Idx'])
        start   = float(row['Start'])
        end     = float(row['End'])
        y_ar    = np.float32(row['Arousal'])
        y_val   = np.float32(row['Valence'])

        ecg_path = os.path.join(feat_path, subject, 'ECG', '30s', f'ecg_{idx}.npy')
        if not os.path.exists(ecg_path):
            continue
        ecg_feat_raw = np.load(ecg_path).astype(np.float32)

        if subject not in wf_cache:
            wf_cache[subject] = (
                load_waveform(subject, 'ECG', raw_path),
                load_waveform(subject, 'PPG', raw_path),
            )
        ecg_wf, ppg_wf = wf_cache[subject]
        if ecg_wf is None or ppg_wf is None:
            continue

        ecg_seg = extract_segment(ecg_wf[0], ecg_wf[1], start, end)
        ppg_seg = extract_segment(ppg_wf[0], ppg_wf[1], start, end)
        if ecg_seg is None or ppg_seg is None:
            continue

        ecg_res = zscore(scipy_resample(ecg_seg, moment_len).astype(np.float32))
        ppg_res = zscore(scipy_resample(ppg_seg, moment_len).astype(np.float32))
        x_tea = np.stack([ecg_res, ppg_res], axis=0)

        if np.isnan(x_tea).any() or np.isnan(ecg_feat_raw).any() or np.isnan(y_ar) or np.isnan(y_val):
            continue

        samples.append({
            'subject':      int(row['Subject']),
            'ecg_feat_raw': ecg_feat_raw,
            'x_tea':        x_tea,
            'y_ar':         y_ar,
            'y_val':        y_val,
        })

    print(f"有効サンプル数: {len(samples)}")
    return samples
