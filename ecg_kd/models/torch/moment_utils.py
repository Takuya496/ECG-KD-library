"""
MOMENTモデルのロード・アンフリーズ・推論ユーティリティ（④⑤⑥共通）
"""
import numpy as np
import torch
from momentfm import MOMENTPipeline


def load_moment(device, frozen=True):
    """MOMENT-1-largeをロード。frozen=Trueで全パラメータ固定"""
    model = MOMENTPipeline.from_pretrained(
        'AutonLab/MOMENT-1-large',
        model_kwargs={'task_name': 'embedding'}
    )
    model.init()
    model = model.to(device)
    if frozen:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    return model


def unfreeze_last_n_blocks(moment_pipeline, n):
    """⑤用: MOMENT後ろnブロックのみアンフリーズ"""
    for param in moment_pipeline.parameters():
        param.requires_grad = False
    model   = moment_pipeline.model if hasattr(moment_pipeline, 'model') else moment_pipeline
    encoder = getattr(model, 'encoder', None)
    if encoder is None:
        print("  警告: encoderが見つかりません")
        return
    blocks = getattr(encoder, 'layers', None) or getattr(encoder, 'block', None)
    if blocks is None:
        print("  警告: encoder blocksが見つかりません。headのみ学習可能")
        return
    total = len(blocks)
    for i, blk in enumerate(blocks):
        if i >= total - n:
            for param in blk.parameters():
                param.requires_grad = True
    print(f"  MOMENT後ろ{n}/{total}ブロックをアンフリーズ")


def get_embedding(moment_model, x_enc):
    """(batch, 2, 512) → (batch, 1024) embedding"""
    out = moment_model(x_enc=x_enc)
    emb = out.embeddings
    if emb.dim() == 3:
        emb = emb.mean(dim=1)
    return emb


def compute_embeddings(moment_model, all_samples, device, batch_size=64):
    """④用: 全サンプルのembeddingを一括事前計算"""
    raw_embeddings = np.zeros((len(all_samples), 1024), dtype=np.float32)
    for i in range(0, len(all_samples), batch_size):
        batch       = all_samples[i:i + batch_size]
        x_tea_batch = torch.tensor(np.stack([s['x_tea'] for s in batch])).to(device)
        with torch.no_grad():
            emb = get_embedding(moment_model, x_tea_batch)
        raw_embeddings[i:i + len(batch)] = emb.cpu().numpy()
        if (i // batch_size + 1) % 10 == 0:
            print(f"  embedding計算: {i + len(batch)}/{len(all_samples)}")
    return raw_embeddings


def compute_teacher_preds_from_embeddings(teacher_head, indices, raw_embeddings, device,
                                          batch_size=64):
    """④用: 事前計算済みembeddingから教師予測キャッシュを生成"""
    n     = raw_embeddings.shape[0]
    t_ar  = np.zeros(n, dtype=np.float32)
    t_val = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            emb_b     = torch.tensor(raw_embeddings[batch_idx]).to(device)
            ta, tv    = teacher_head(emb_b)
            for k, orig_idx in enumerate(batch_idx):
                t_ar[orig_idx]  = ta[k].item()
                t_val[orig_idx] = tv[k].item()
    return t_ar, t_val


def compute_teacher_preds_from_waveform(moment_model, teacher_head, indices, all_samples,
                                        device, batch_size=32):
    """⑤⑥用: MOMENT推論→教師予測キャッシュを生成"""
    t_ar  = np.zeros(len(all_samples), dtype=np.float32)
    t_val = np.zeros(len(all_samples), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            x_tea_b   = torch.tensor(
                np.stack([all_samples[j]['x_tea'] for j in batch_idx])
            ).to(device)
            emb    = get_embedding(moment_model, x_tea_b)
            ta, tv = teacher_head(emb)
            for k, orig_idx in enumerate(batch_idx):
                t_ar[orig_idx]  = ta[k].item()
                t_val[orig_idx] = tv[k].item()
    return t_ar, t_val
