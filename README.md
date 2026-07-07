# ECG-KD-library

ECG・PPG信号を用いた感情認識のための Knowledge Distillation（知識蒸留）ライブラリ。  
②③④⑤⑥の5モデルを1つのpip installableパッケージとして統合。

---

## モデル一覧

| 番号 | 名前 | 教師モデル | フレームワーク |
|------|------|-----------|---------------|
| ② | KD-Ensemble | KD SingleModels × 5（TF固定） | TensorFlow |
| ③ | KD-E2E | endtoend KD SingleModels × 5（TF固定） | TensorFlow |
| ④ | MOMENT-HeadOnly | MOMENT（全固定）+ Head再学習 | PyTorch |
| ⑤ | MOMENT-PartialUnfreeze | MOMENT後ろ6ブロック解放 + Head | PyTorch |
| ⑥ | MOMENT-FullFT | MOMENT全パラメータ解放 + Head | PyTorch |

**生徒モデル（共通）**：ECG特徴量 19次元 → Arousal / Valence 回帰

---

## インストール

```bash
cd /path/to/ECG-KD-library
pip install -e .
```

---

## ライブラリ構成

```
ecg_kd/
├── data/
│   └── loader.py           データ読み込み（全モデル共通）
├── losses/
│   └── torch_losses.py     損失関数 pcc / ccc / teacher / student_kd（④⑤⑥）
├── models/
│   ├── torch/
│   │   ├── heads.py        TeacherRegressionHead / StudentRegressionHead（④⑤⑥）
│   │   ├── datasets.py     Dataset定義（④⑤⑥）
│   │   └── moment_utils.py MOMENTロード・アンフリーズ・embedding計算（④⑤⑥）
│   └── tf/
│       └── heads.py        TF版 RegressionHead（②③）
└── training/
    └── cv_utils.py         fold分割・正規化・結果保存（全モデル共通）
```

---

## 実行ファイル（scripts/）

各モデルに **評価用（cv）** と **デプロイ用（all）** の2種類があります。

| スクリプト | 内容 |
|-----------|------|
| `train_*_cv.py` | Nested 5-fold GroupKFold 交差検証。RMSE を算出して評価する |
| `train_*_all.py` | 全100被験者のデータで学習。検証なし。実運用モデルの保存用 |

---

## 実行方法

### ④⑤⑥ PyTorchモデル

**評価（5-fold交差検証）：**
```bash
python scripts/train_head_only_cv.py --alpha 0.5 --teacher_epochs 100
python scripts/train_partial_cv.py   --alpha 0.5 --teacher_epochs 100
python scripts/train_full_ft_cv.py   --alpha 0.5 --teacher_epochs 100
```

**デプロイ用モデル保存（全被験者学習）：**
```bash
python scripts/train_head_only_all.py --alpha 0.5 --teacher_ckpt /path/to/final_model.pt
python scripts/train_partial_all.py   --alpha 0.5 --teacher_ckpt /path/to/final_model.pt
python scripts/train_full_ft_all.py   --alpha 0.5 --teacher_ckpt /path/to/final_model.pt
```

### ②③ TensorFlowモデル

**評価（5-fold交差検証）：**
```bash
python scripts/train_ensemble_cv.py  --alpha 0.5 --teacher_epochs 100
python scripts/train_endtoend_cv.py  --alpha 0.5 --teacher_epochs 100
```

**デプロイ用モデル保存（全被験者学習）：**
```bash
python scripts/train_ensemble_all.py
python scripts/train_endtoend_all.py
```

---

## 引数

| 引数 | 対象 | 説明 | デフォルト |
|------|------|------|-----------|
| `--alpha` | 全モデル | KD損失の重み（BETA = 1 - alpha） | 0.5 |
| `--teacher_epochs` | cvスクリプト | 教師モデルの学習エポック数 | 100 |
| `--teacher_ckpt` | allスクリプト | 事前学習済み教師チェックポイントパス | 必須 |
| `--unfreeze_n` | ⑤のみ | アンフリーズするMOMENTブロック数 | 6 |

---

## 学習の流れ

```
Phase 1: 教師モデルの学習（cvスクリプトのみ）
  ↓
  fold内の訓練データで教師モデルを再学習
  → テストデータへの情報漏洩を防ぐ

Phase 2: 生徒モデルのKD学習
  ↓
  教師の予測（ソフトラベル）+ 真のラベルを使って生徒を学習
  loss = alpha × KD損失 + beta × (MSE + PCC + CCC)
```

---

## 出力ファイル

| ファイル | 内容 |
|---------|------|
| `fold_results.csv` | 各foldのRMSE（cvスクリプト） |
| `fold_results_checkpoint.csv` | 途中再開用チェックポイント |
| `summary.txt` | 平均RMSE ± 標準偏差 |
| `best_model.pt` | 最小損失時のモデル（allスクリプト） |
| `final_model.pt` | 最終epochのモデル（allスクリプト） |
| `mean_ecg.npy` / `std_ecg.npy` | 正規化統計量（推論時に使用） |

---

## 評価結果（Nested 5-fold, alpha=0.5）

| モデル | RMSE Arousal | RMSE Valence |
|--------|-------------|--------------|
| ② KD-Ensemble | 0.8042 | 0.7670 |
| ③ KD-E2E | 0.8045 | 0.7668 |
| ④ MOMENT-HeadOnly | 0.7784 | 0.7638 |
| ⑤ MOMENT-PartialUnfreeze | 0.9641 | 0.9913 |
| ⑥ MOMENT-FullFT | **0.7799** | **0.7631** |

---

## 依存ライブラリ

```
numpy, pandas, scipy, scikit-learn  # 共通
torch, momentfm                     # ④⑤⑥
tensorflow                          # ②③
```
