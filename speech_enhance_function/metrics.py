"""
评估指标相关
"""

import torch
import torch.nn as nn
from pesq import pesq_batch, PesqError
import numpy as np

def get_metrics(clean, estimate):
    clean = clean.squeeze(dim=1)
    estimate = estimate.squeeze(dim=1)
    lsd = get_lsd(clean, estimate).item()
    # pesq = compute_pesq_mean(clean, estimate, fs=8000, mode='nb').item()
    return lsd


class STFTMag(nn.Module):
    def __init__(self,
                 nfft=1024,
                 hop=256):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer('window', torch.hann_window(nfft), False)

    # x: [B,T] or [T]
    @torch.no_grad()
    def forward(self, x):
        T = x.shape[-1]
        stft = torch.stft(x, self.nfft, self.hop, window=self.window, return_complex=False)  #[B, F, TT,2]
        mag = torch.norm(stft, p=2, dim=-1)  # [B, F, TT]
        return mag

def get_lsd(ref_sig, out_sig):
    """
       Compute LSD (log spectral distance)
       Arguments:
           out_sig: vector (torch.Tensor), enhanced signal [B,T]
           ref_sig: vector (torch.Tensor), reference signal(ground truth) [B,T]
    """
    stft = STFTMag(2048, 512)
    sp = torch.log10(stft(ref_sig).square().clamp(1e-8))
    st = torch.log10(stft(out_sig).square().clamp(1e-8))
    return (sp - st).square().mean(dim=1).sqrt().mean()


def compute_pesq_mean(ref_tensor, deg_tensor, fs=16000, mode='wb'):
    """
    输入:
        ref_tensor: [B, T], 参考音频（干净音频）
        deg_tensor: [B, T], 待测音频（降质音频）
        fs: 采样率 (16000 for 'wb', 8000 for 'nb')
        mode: 'wb' (宽带) 或 'nb' (窄带)
    返回:
        pesq_mean: float, 整个 batch 的 PESQ 均值（跳过无效样本）
    """
    ref_np = ref_tensor.numpy()  # [B, T]
    deg_np = deg_tensor.numpy()  # [B, T]

    valid_scores = []
    for i in range(ref_np.shape[0]):
        try:
            score = pesq_batch(
                fs=fs,
                ref=ref_np[i],  # 第 i 个样本 [T]
                deg=deg_np[i],  # 第 i 个样本 [T]
                mode=mode,
                n_processor=1,
                on_error=PesqError.RAISE_EXCEPTION  # 严格模式
            )[0]  # 直接取分数
            valid_scores.append(score)
        except Exception as e:
            print(f"跳过无效样本 {i}: {str(e)}")

    if not valid_scores:
        return torch.nan  # 全部样本无效时返回 nan

    return torch.tensor(np.mean(valid_scores))  # 计算均值