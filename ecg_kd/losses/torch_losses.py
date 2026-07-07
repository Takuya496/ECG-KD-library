"""
PyTorch損失関数（④⑤⑥共通）
"""
import torch
import torch.nn.functional as F


def pcc_loss(y_true, y_pred):
    vt = y_true - y_true.mean()
    vp = y_pred - y_pred.mean()
    return 1.0 - (vt * vp).sum() / (torch.sqrt((vt**2).sum() * (vp**2).sum() + 1e-8))


def ccc_loss(y_true, y_pred):
    mt, mp = y_true.mean(), y_pred.mean()
    vt  = ((y_true - mt)**2).mean()
    vp  = ((y_pred - mp)**2).mean()
    cov = ((y_true - mt) * (y_pred - mp)).mean()
    return 1.0 - 2.0 * cov / (vt + vp + (mt - mp)**2 + 1e-8)


def teacher_loss(t_ar, t_val, y_ar, y_val, p=1/3, q=1/3, r=1/3):
    """教師Phase1損失: MSE + PCC + CCC の平均"""
    mse = 0.5 * (F.mse_loss(t_ar, y_ar) + F.mse_loss(t_val, y_val))
    pcc = 0.5 * (pcc_loss(y_ar, t_ar)   + pcc_loss(y_val, t_val))
    ccc = 0.5 * (ccc_loss(y_ar, t_ar)   + ccc_loss(y_val, t_val))
    return p * mse + q * pcc + r * ccc


def student_kd_loss(s_ar, s_val, t_ar, t_val, y_ar, y_val,
                    alpha=0.5, beta=0.5, p=1/3, q=1/3, r=1/3):
    """生徒KD損失: alpha*KD + beta*(MSE+PCC+CCC)"""
    kd       = 0.5 * (F.mse_loss(s_ar, t_ar) + F.mse_loss(s_val, t_val))
    task_mse = 0.5 * (F.mse_loss(s_ar, y_ar) + F.mse_loss(s_val, y_val))
    task_pcc = 0.5 * (pcc_loss(y_ar, s_ar)   + pcc_loss(y_val, s_val))
    task_ccc = 0.5 * (ccc_loss(y_ar, s_ar)   + ccc_loss(y_val, s_val))
    return alpha * kd + beta * (p * task_mse + q * task_pcc + r * task_ccc)
