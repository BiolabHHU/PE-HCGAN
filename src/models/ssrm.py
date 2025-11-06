
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import librosa
from src.models.utils import capture_init
from src.models.spec import spectro, ispectro
from src.models.modules import ScaledEmbedding, FTB
from src.models.harmonic_model import FusionHarmonicPredictor
from torchnssd.ex_bi_mamba2_ac import BiMamba2Ac2d
import logging
logger = logging.getLogger(__name__)


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale

def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)
# ===深度可分卷积（开始）================================================================================
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv2d, self).__init__()
        # Depthwise Convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        # Pointwise Convolution (1x1 Convolution)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
# ===深度可分卷积（结束）================================================================================

# ===sinc卷积（开始）================================================================================
def generate_sinc_kernel(M, fc, device='cpu'):
    """
    生成 2D sinc 卷积核，带可调截止频率，使用 PyTorch 操作
    参数:
        M: 窗口半径，例如 M=2 时，核大小为 5x5
        fc: 标准化截止频率，范围 [0, 0.5]，PyTorch 张量
        device: 计算设备（cpu 或 cuda）
    返回:
        归一化的 2D sinc 核（PyTorch 张量）
    """
    # 生成采样点
    x = torch.linspace(-M, M, 2 * M + 1, device=device)
    X, Y = torch.meshgrid(x, x, indexing='ij')  # 使用 indexing='ij' 确保正确网格
    # 计算 sinc 函数，torch.sinc 是标准化的 sinc(x) = sin(pi*x)/(pi*x)
    kernel = torch.sinc(2 * fc * X) * torch.sinc(2 * fc * Y)
    kernel = kernel / kernel.sum()  # 归一化
    return kernel


# 定义具有可学习截止频率的 sinc 卷积层，保持空间维度
class SincConv(nn.Module):
    def __init__(self, chin, kernel_size, fc_init=0.1):
        """
        参数:
            chin: 输入通道数
            kernel_size: 卷积核大小（整数，例如 3 或 5）
            fc_init: 截止频率初始值，默认为 0.25
        """
        super(SincConv, self).__init__()
        # 添加 padding 以保持空间维度
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(chin, chin, kernel_size, groups=chin, bias=False, padding=padding)
        self.kernel_size = kernel_size
        self.chin = chin

        # 可学习的截止频率参数，未限制的原始值
        self.fc_raw = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # 初始化时生成一个默认核
        M = (kernel_size - 1) // 2
        sinc_kernel = generate_sinc_kernel(M, torch.tensor(fc_init, device=self.conv.weight.device))
        with torch.no_grad():
            sinc_kernel_ = sinc_kernel.unsqueeze(0).unsqueeze(0).repeat(chin, 1, 1, 1)
            self.conv.weight.copy_(sinc_kernel_)

    def forward(self, x):
        # 将 fc_raw 映射到 [0.1, 0.5] 范围
        fc = torch.sigmoid(self.fc_raw) * 0.5  # 范围 [0, 0.5]
        # 动态生成 sinc 核，使用与输入相同的设备
        M = (self.kernel_size - 1) // 2
        sinc_kernel = generate_sinc_kernel(M, fc, device=x.device)
        # 更新卷积核权重
        with torch.no_grad():
            self.conv.weight.copy_(sinc_kernel.unsqueeze(0).unsqueeze(0).repeat(self.chin, 1, 1, 1))
        return self.conv(x)


# 定义新的 PreConv 模块，保持输出维度与输入一致
class PreConv(nn.Module):
    def __init__(self, chin, chout, kernel_size=5, fc_init=0.1):
        """
        参数:
            chin: 输入通道数
            chout: 输出通道数
            kernel_size: sinc 卷积核大小，默认为 5
            fc_init: 截止频率初始值，默认为 0.25
        """
        super(PreConv, self).__init__()
        self.sinc_conv = SincConv(chin, kernel_size, fc_init)  # sinc 卷积，带可学习截止频率
        self.conv1x1 = nn.Conv2d(chin, chout, 1)  # 1x1 卷积，调整通道数

    def forward(self, x):
        x = self.sinc_conv(x)  # 应用 sinc 空间滤波
        x = self.conv1x1(x)  # 调整通道数
        return x
# ===sinc卷积（结束）================================================================================

# ===编码器（开始）================================================================================
class HEncLayer(nn.Module):
    def __init__(self, chin, chout, kernel_size=8, stride=4, norm_groups=1, empty=False,
                 freq=True, is_first=False, freq_attn=False, freq_dim=None, norm=True, context=0,
                pad=True, rewrite=True):

        super().__init__()
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa
            # norm_fn = lambda d: nn.BatchNorm2d(d)  # noqa
        if stride == 1 and kernel_size % 2 == 0 and kernel_size > 1:
            kernel_size -= 1
        if pad:
            pad = (kernel_size - stride) // 2
        else:
            pad = 0
        # klass = DepthwiseSeparableConv2d
        klass = nn.Conv2d
        # klass = KANConv2d
        self.chin = chin
        self.chout = chout
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.empty = empty
        self.freq_attn = freq_attn
        self.freq_dim = freq_dim
        self.norm = norm
        self.pad = pad
        self.is_first = is_first

        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            if pad != 0:
                pad = [pad, 0]
            # klass = nn.Conv2d
        else:
            kernel_size = [1, kernel_size]
            stride = [1, stride]
            if pad != 0:
                pad = [0, pad]

        if is_first:
            self.pre_conv = nn.Conv2d(chin, chout, [1, 1])
            # self.pre_conv = PreConv(chin, chout, 1)
            chin = chout
        if self.freq_attn:
            self.freq_attn_block = FTB(input_dim=freq_dim, in_channel=chin)
        self.conv = klass(chin, chout, kernel_size, stride, pad)
        # self.conv = FastKANConv2DLayer(chin, chout, kernel_size=kernel_size, stride=stride, padding=pad)
        if self.empty:
            return
        self.norm1 = norm_fn(chout)
        self.rewrite = None
        if rewrite:
            self.rewrite = klass(chout, 2 * chout, 1 + 2 * context, 1, context)
            # self.rewrite = FastKANConv2DLayer(chout, 2 * chout, kernel_size=1 + 2 * context, stride=1, padding=context)
            self.norm2 = norm_fn(2 * chout)

    def forward(self, x, inject=None):
        """
        `inject` is used to inject the result from the time branch into the frequency branch,
        when both have the same stride.
        """
        if not self.freq:
            le = x.shape[-1]
            if not le % self.stride == 0:
                x = F.pad(x, (0, self.stride - (le % self.stride)))
        if self.is_first:
            x = self.pre_conv(x)
        if self.freq_attn:
            x = self.freq_attn_block(x)
        x = self.conv(x)
        x = F.gelu(self.norm1(x))
        if self.rewrite:
            x = self.norm2(self.rewrite(x))
            x = F.glu(x, dim=1)
        return x
# ===编码器（结束）================================================================================

# ===解码器（开始）================================================================================
class HDecLayer(nn.Module):
    def __init__(self, chin, chout, last=False, kernel_size=8, stride=4, norm_groups=1, empty=False,
                 freq=True, norm=True, context=1, pad=True,
                 context_freq=True, rewrite=True):
        """
        Same as HEncLayer but for decoder. See `HEncLayer` for documentation.
        """
        super().__init__()
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa
            # norm_fn = lambda d: nn.BatchNorm2d(d)  # noqa
        if stride == 1 and kernel_size % 2 == 0 and kernel_size > 1:
            kernel_size -= 1
        if pad:
            pad = (kernel_size - stride) // 2
        else:
            pad = 0
        self.pad = pad
        self.last = last
        self.freq = freq
        self.chin = chin
        self.empty = empty
        self.stride = stride
        self.kernel_size = kernel_size
        self.norm = norm
        self.context_freq = context_freq
        klass = DepthwiseSeparableConv2d
        # klass = nn.Conv2d
        # klass = KANConv2d
        klass_tr = nn.ConvTranspose2d
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
        else:
            kernel_size = [1, kernel_size]
            stride = [1, stride]
        self.conv_tr = klass_tr(chin, chout, kernel_size, stride)
        # self.conv_tr = FastKANConv2DLayer(chin, chout, kernel_size=kernel_size, stride=stride)
        self.norm2 = norm_fn(chout)
        # self.lastconv = klass(2, 2, (1, 1), (1, 1), 0)
        if self.empty:
            return
        self.rewrite = None
        if rewrite:
            if context_freq:
                self.rewrite = klass(chin, 2 * chin, 1 + 2 * context, 1, context)
                # self.rewrite = FastKANConv2DLayer(chin, 2 * chin, kernel_size=1 + 2 * context, stride=1, padding=context)
            else:
                self.rewrite = klass(chin, 2 * chin, [1, 1 + 2 * context], 1, [0, context])
                # self.rewrite = FastKANConv2DLayer(chin, 2 * chin, kernel_size=[1, 1 + 2 * context], stride=1, padding=[0, context])
            self.norm1 = norm_fn(2 * chin)

    def forward(self, x, skip, length):
        if self.freq and x.dim() == 3:
            B, C, T = x.shape
            x = x.view(B, self.chin, -1, T)
        if not self.empty:
            x = torch.cat([x, skip], dim=1)
            if self.rewrite:
                y = F.glu(self.norm1(self.rewrite(x)), dim=1)
            else:
                y = x
        else:
            y = x
            assert skip is None
        z = self.norm2(self.conv_tr(y))
        if self.freq:
            if self.pad:
                z = z[..., self.pad:-self.pad, :]
        else:
            z = z[..., self.pad:self.pad + length]
            assert z.shape[-1] == length, (z.shape[-1], length)
        if not self.last:
            z = F.gelu(z)
        # if self.last:
        #     z = self.lastconv(z)
        return z
# ===解码器（结束）================================================================================

# ===频谱网络（开始）================================================================================
class SSRM(nn.Module):
    """
    Deep model for Audio Super Resolution.
    """

    @capture_init
    def __init__(self,
                 # Channels
                 in_channels=1,
                 out_channels=1,
                 audio_channels=2,
                 channels=48,
                 growth=2,
                 # STFT
                 nfft=512,
                 hop_length=64,
                 end_iters=0,
                 cac=True,
                 # Main structure
                 rewrite=True,
                 hybrid=False,
                 hybrid_old=False,
                 # Frequency branch
                 freq_emb=0.2,
                 emb_scale=10,
                 emb_smooth=True,
                 # Convolutions
                 kernel_size=8,
                 strides=[4, 4, 2, 2],
                 context=1,
                 context_enc=0,
                 freq_ends=4,
                 enc_freq_attn=4,
                 # Normalization
                 norm_starts=2,
                 norm_groups=4,

                 # Weight init
                 rescale=0.1,
                 # Metadata
                 lr_sr=4000,
                 hr_sr=16000,
                 spec_upsample=True,
                 # act_func='snake',
                 pipr=False,
                 debug=False):

        super().__init__()
        self.cac = cac
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.audio_channels = audio_channels
        self.kernel_size = kernel_size
        self.context = context
        self.strides = strides
        self.depth = len(strides)
        self.channels = channels
        self.lr_sr = lr_sr
        self.hr_sr = hr_sr
        self.spec_upsample = spec_upsample

        self.scale = hr_sr / lr_sr if self.spec_upsample else 1

        self.nfft = nfft
        self.hop_length = int(hop_length // self.scale)  # this is for the input signal
        self.win_length = int(self.nfft // self.scale)  # this is for the input signal
        self.end_iters = end_iters
        self.freq_emb = None
        self.hybrid = hybrid
        self.hybrid_old = hybrid_old
        self.pipr = pipr
        self.debug = debug

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        # self.mamba = BiMamba2Ac2d(384,384,32)     # 当前谐波模型禁用mamba
        if self.pipr:
            self.hapr = FusionHarmonicPredictor()          # 还有其他谐波模型形式

        chin_z = self.in_channels
        if self.cac:
            chin_z *= 2
        chout_z = channels
        freqs = nfft // 2

        for index in range(self.depth):
            freq_attn = index >= enc_freq_attn
            norm = index >= norm_starts
            freq = index <= freq_ends
            stri = strides[index]
            ker = kernel_size

            pad = True
            if freq and freqs < kernel_size:
                ker = freqs

            kw = {
                'kernel_size': ker,
                'stride': stri,
                'freq': freq,
                'pad': pad,
                'norm': norm,
                'rewrite': rewrite,
                'norm_groups': norm_groups,
            }

            kw_dec = dict(kw)

            enc = HEncLayer(chin_z, chout_z,
                            context=context_enc,
                            is_first=index == 5,
                            freq_attn=freq_attn,
                            freq_dim=freqs,
                            **kw)

            self.encoder.append(enc)
            if index == 0:
                chin = self.out_channels
                chin_z = chin
                if self.cac:
                    chin_z *= 2
            dec = HDecLayer(2 * chout_z, chin_z,
                            last=index == 0,
                            context=context,
                            **kw_dec)

            self.decoder.insert(0, dec)

            chin_z = chout_z
            chout_z = int(growth * chout_z)

            if freq:
                freqs //= strides[index]

            if index == 0 and freq_emb:
                self.freq_emb = ScaledEmbedding(
                    freqs, chin_z, smooth=emb_smooth, scale=emb_scale)
                self.freq_emb_scale = freq_emb

        if rescale:
            rescale_module(self, reference=rescale)

    def _spec(self, x, scale=False):
        if np.mod(x.shape[-1], self.hop_length):
            x = F.pad(x, (0, self.hop_length - np.mod(x.shape[-1], self.hop_length)))
        hl = self.hop_length
        nfft = self.nfft
        win_length = self.win_length

        if scale:
            hl = int(hl * self.scale)
            win_length = int(win_length * self.scale)

        z = spectro(x, nfft, hl, win_length=win_length)[..., :-1, :]
        return z

    def _ispec(self, z):
        hl = int(self.hop_length * self.scale)
        win_length = int(self.win_length * self.scale)
        z = F.pad(z, (0, 0, 0, 1))
        x = ispectro(z, hl, win_length=win_length)
        return x

    def _move_complex_to_channels_dim(self, z):
        B, C, Fr, T = z.shape
        m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
        m = m.reshape(B, C * 2, Fr, T)
        return m

    def _convert_to_complex(self, x):
        """

        :param x: signal of shape [Batch, Channels, 2, Freq, TimeFrames]
        :return: complex signal of shape [Batch, Channels, Freq, TimeFrames]
        """
        out = x.permute(0, 1, 3, 4, 2)
        out = torch.view_as_complex(out.contiguous())
        return out


    def torch_pitch(self, signal, sr, nfft=1024, hop_length=256, fmin=40, fmax=300):
        """
        signal: (B, L)
        返回 pitches: (B, T)
        """
        B, L = signal.shape
        device = signal.device

        # 检查输入长度是否足够
        if L < nfft:
            # 如果信号太短，可以选择：
            # 1. 报错并提示用户
            # 2. 自动填充零（padding）
            # 3. 调整 nfft 使其不超过 L
            # 这里选择填充零
            pad_size = nfft - L
            signal = torch.nn.functional.pad(signal, (0, pad_size), mode='constant', value=0)
            L = nfft  # 更新长度

        # 分帧: unfold, (B, T, nfft)
        frames = signal.unfold(dimension=1, size=nfft, step=hop_length)
        B, T, _ = frames.shape

        # 加窗
        window = torch.hann_window(nfft, device=device)
        frames = frames * window

        # 自相关 via FFT：r = ifft(fft(x)*conj(fft(x)))
        # 为避免循环，先扩张 batch 和时间帧到 (B*T, nfft)
        frames_flat = frames.reshape(-1, nfft)  # (B*T, nfft)

        fft_size = 2 * nfft
        fft_frames = torch.fft.rfft(frames_flat, n=fft_size)
        power_spec = fft_frames * torch.conj(fft_frames)
        autocorr = torch.fft.irfft(power_spec, n=fft_size)  # (B*T, fft_size)

        # 取正的延迟部分，只保留长度 nfft
        autocorr = autocorr[:, :nfft]

        # 有效范围内找峰值：min_period ~ max_period
        min_period = int(sr / fmax)
        max_period = int(sr / fmin)
        search_region = autocorr[:, min_period:max_period]  # (B*T, P)

        # 找最大值索引
        peak_idx = search_region.argmax(dim=1) + min_period  # (B*T,)

        # 转回频率
        pitches = sr / peak_idx.float()  # (B*T,)

        pitches = pitches.view(B, T)  # (B, T)

        # 合理性过滤
        pitches[(pitches < fmin) | (pitches > fmax)] = float('nan')

        return pitches

    def _extract_harmonic_features(self, mix, magnitude, n_harmonics):
        """
        并行版本 + 封装基频提取
        mix: (B, C, L)
        magnitude: (B, C, Fq, T)
        返回:
            all_harmonic_mags: (B, T, n_harmonics)
            all_pitches: (B, T)
        """
        B, C, Fq, T = magnitude.shape
        device = magnitude.device
        with torch.no_grad():
            # Step 1: 提取基频
            # 对第一个通道
            mix_mono = mix[:, 0, :]  # (B, L)
            pitches = self.torch_pitch(mix_mono, sr=self.lr_sr, nfft=self.nfft, hop_length=self.hop_length, fmin=40,
                                  fmax=300)  # (B, T')
            # pad/crop 到 T 帧长度
            T_p = pitches.shape[1]
            if T_p < T:
                pitches = F.pad(pitches, (0, T - T_p))
            elif T_p > T:
                pitches = pitches[:, :T]

            # harmonic numbers
            harmonic_nums = torch.arange(1, n_harmonics + 1, device=device)  # (n_harmonics,)

            # harmonic freqs: (B, T, n_harmonics)
            harmonic_freqs = pitches.unsqueeze(-1) * harmonic_nums

            # nan → 0
            harmonic_freqs = torch.nan_to_num(harmonic_freqs, nan=0.0)

            # Step 2: 对应频率 bin
            frequencies = torch.fft.fftfreq(self.nfft, d=1. / self.lr_sr)[:Fq].to(device)  # (Fq,)
            harmonic_indices = torch.searchsorted(frequencies, harmonic_freqs)
            harmonic_indices = torch.clamp(harmonic_indices, 0, Fq - 1)  # (B, T, n_harmonics)

            # Step 3: 从 magnitude 中取出 harmonic mag
            # magnitude: (B, C, Fq, T)
            magnitude = magnitude.permute(0, 3, 2, 1)  # (B, T, Fq, C)

            harmonic_mags = []

            for h in range(n_harmonics):
                idx = harmonic_indices[:, :, h]  # (B, T)
                # gather：需要在 dim=2
                idx_exp = idx.unsqueeze(-1).expand(-1, -1, C)  # (B, T, C)
                # 对 freq dim gather
                gathered = torch.gather(magnitude, dim=2, index=idx_exp.unsqueeze(2))  # (B, T, 1, C)
                gathered = gathered.squeeze(2)  # (B, T, C)
                # 对通道取均值
                mag_h = gathered.mean(dim=2)  # (B, T)
                harmonic_mags.append(mag_h)

            # (n_harmonics, B, T) → (B, T, n_harmonics)
            all_harmonic_mags = torch.stack(harmonic_mags, dim=2)

        return all_harmonic_mags, pitches

    def _insert_harmonic_features(self, spec, harmonic, all_pitch):
        """
        输入：
            spec: (B, C, Fq, T)，复数
            harmonic: (B, C, n_harmonics, T)，实数
            all_pitch: (B, T)
        返回：
            spec: (B, C, Fq, T)，只在 all_pitch*harmonic_nums 对应频率的实部插入 harmonic，虚部保持不变
        """
        B, C, Fq, T = spec.shape
        n_harmonics = harmonic.shape[2]
        harmonic_nums = torch.arange(20, 25, device=spec.device)  # n_insert=5,35-40，具体插入位置需具体设置，或可设置训练参数

        # 计算目标频率
        harmonic_freqs = all_pitch.unsqueeze(-1) * harmonic_nums  # (B, T, 5)
        harmonic_freqs = torch.nan_to_num(harmonic_freqs, nan=0.0)

        frequencies_pr = librosa.fft_frequencies(sr=self.hr_sr, n_fft=self.nfft)
        frequencies_pr_torch = torch.tensor(frequencies_pr, device=spec.device)

        # 找到对应 bin
        harmonic_indices = torch.searchsorted(frequencies_pr_torch, harmonic_freqs)  # (B, T, 5)
        harmonic_indices = torch.clamp(harmonic_indices, 0, Fq - 1)

        n_insert = min(n_harmonics, harmonic_nums.shape[0])
        harmonic = harmonic[:, :, :n_insert, :]  # (B, C, 5, T)
        harmonic_indices = harmonic_indices[:, :, :n_insert]  # (B, T, 5)

        # permute harmonic to (B, C, T, 5)
        harmonic = harmonic.permute(0, 1, 3, 2)  # (B, C, T, 5)

        # 构造下标
        b_idx = torch.arange(B, device=spec.device).view(B, 1, 1).expand(-1, T, n_insert)  # (B, T, 5)
        t_idx = torch.arange(T, device=spec.device).view(1, T, 1).expand(B, -1, n_insert)  # (B, T, 5)
        f_idx = harmonic_indices  # (B, T, 5)

        # spec.real: (B, C, Fq, T) → (B, C, T, Fq)
        spec_real_T = spec.real.permute(0, 1, 3, 2)  # (B, C, T, Fq)

        # 当前通道索引： (1, C, 1, 1)
        c_idx = torch.arange(C, device=spec.device).view(1, C, 1, 1)

        # gather 原实部值： (B, C, T, 5)
        spec_vals = spec_real_T[b_idx.unsqueeze(1), c_idx, t_idx.unsqueeze(1), f_idx.unsqueeze(1)]  # (B, C, T, 5)

        # harmonic: (B, C, T, 5)
        # new_vals = 0.5 * spec_vals + 0.5 * harmonic
        new_vals = harmonic

        # 写回实部
        spec_real_T[b_idx.unsqueeze(1), c_idx, t_idx.unsqueeze(1), f_idx.unsqueeze(1)] = new_vals

        # 恢复 shape： (B, C, Fq, T)
        new_real = spec_real_T.permute(0, 1, 3, 2)

        # 拼回复数
        spec = torch.complex(new_real, spec.imag)

        return spec

    def check_tensor(self, name, tensor):
        if tensor.numel() == 0:
            logger.info(f"[DEBUG] {name} is empty!")
            return
        if torch.isnan(tensor).any():
            logger.info(f"[DEBUG] {name} contains NaN!")
        elif torch.isinf(tensor).any():
            logger.info(f"[DEBUG] {name} contains inf!")
        if tensor.max().item() > 10:
            logger.info(f"[DEBUG] {name} >10")
        # else:
        #     logger.info(
        #         f"{name}: mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}, "
        #         f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}"
        #     )

    def forward(self, mix, return_spec=False, return_lr_spec=False):
        x = mix
        length = x.shape[-1]
        # print(f"开始时间：{datetime.now()}")
        if self.debug:
            logger.info(f'hdemucs in shape: {x.shape}')
        z = self._spec(x)
        self.check_tensor("z real", torch.real(z))
        self.check_tensor("z imag", torch.imag(z))

        x = self._move_complex_to_channels_dim(z)

        if self.debug:
            logger.info(f'x spec shape: {x.shape}')

        B, C, Fq, T = x.shape

        # unlike previous Demucs, we always normalize because it is easier.
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        # okay, this is a giant mess I know...
        saved = []  # skip connections, freq.
        lengths = []  # saved lengths to properly remove padding, freq branch.
        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            x = encode(x, inject)
            if self.debug:
                logger.info(f'encoder {idx} out shape: {x.shape}')
            if idx == 0 and self.freq_emb is not None:
                # add frequency embedding to allow for non equivariant convolutions
                # over the frequency axis.
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb

            saved.append(x)
        # torch.Size([32, 384, 4, 501])
        # 使用mamba作为瓶颈层
        # x = self.mamba(x)     //当前谐波模型不要使用mamba
        # 无瓶颈层，直接传递0
        x = torch.zeros_like(x)

        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x = decode(x, skip, lengths.pop(-1))

            if self.debug:
                logger.info(f'decoder {idx} out shape: {x.shape}')

        # Let's make sure we used all stored skip connections.
        assert len(saved) == 0

        x = x.view(B, self.out_channels, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        if self.debug:
            logger.info(f'post view shape: {x.shape}')

        x_spec_complex = self._convert_to_complex(x)
        self.check_tensor("111x_spec_complex real", torch.real(x_spec_complex))
        self.check_tensor("111x_spec_complex imag", torch.imag(x_spec_complex))

        if self.debug:
            logger.info(f'x_spec_complex shape: {x_spec_complex.shape}')

        if self.pipr:
            # =================提取前20个谐波的幅度特征====================
            magnitude = torch.real(z)  # 获取频谱的实部
            n_harmonics = 5     # 谐波数，灵活可调
            # all_harmonic_mags --> [B, T, N]
            # t0 = datetime.now()
            # print(f"基频提取前时间：{t0}")
            all_harmonic_mags, all_pitch = self._extract_harmonic_features(mix, magnitude, n_harmonics)
            # print(f"基频提取后时间：{datetime.now() - t0}")
            # =================谐波特征进入神经网络========================
            # B, C, N, T = all_harmonic_mags.shape
            all_harmonic_mags = all_harmonic_mags.unsqueeze(1)
            # all_harmonic_mags = all_harmonic_mags.repeat(1, 2, 1, 1)
            all_harmonic_mags = all_harmonic_mags.permute(0, 1, 3, 2)
            # 归一化
            mean_harmonic = all_harmonic_mags.mean(dim=(1, 2, 3), keepdim=True)
            std_harmonic = all_harmonic_mags.std(dim=(1, 2, 3), keepdim=True)
            harmonic = (all_harmonic_mags - mean_harmonic) / (1e-5 + std_harmonic)

            # x_t = self._ispec(x_spec_complex)
            magnitude_t = torch.real(x_spec_complex)
            harmonic_from_spec, _ = self._extract_harmonic_features(mix, magnitude_t, n_harmonics)
            harmonic_from_spec = harmonic_from_spec.unsqueeze(1)
            harmonic_from_spec = harmonic_from_spec.permute(0, 1, 3, 2)
            # 归一化
            mean_harmonic_spec = harmonic_from_spec.mean(dim=(1, 2, 3), keepdim=True)
            std_harmonic_spec = harmonic_from_spec.std(dim=(1, 2, 3), keepdim=True)
            harmonic_from_spec = (harmonic_from_spec - mean_harmonic_spec) / (1e-5 + std_harmonic_spec)

            harmonic = self.hapr(harmonic, harmonic_from_spec)
            # harmonic = self.hapr(harmonic)

            # 逆归一化
            harmonic = harmonic * std_harmonic + mean_harmonic
            self.check_tensor("harmonic", harmonic)
            # logger.info(f'444 harmonic shape: {harmonic.shape}')


            # ============神经网络输出预测谐波harmonic===========
            # ============将预测的谐波特征插入预测的频谱===========
            # 频谱特征：x_spec_complex(B,C,F,T)      谐波特征：harmonic(B,C,N,T)
            # 基频：all_pitch(B,T)

            x_spec_complex = self._insert_harmonic_features(x_spec_complex, harmonic, all_pitch)
            self.check_tensor("222x_spec_complex real", torch.real(x_spec_complex))
            self.check_tensor("222x_spec_complex imag", torch.imag(x_spec_complex))
        # end========================================================================
        x = self._ispec(x_spec_complex)
        if self.debug:
            logger.info(f'hdemucs out shape: {x.shape}')

        x = x[..., :int(length * self.scale)]

        if self.debug:
            logger.info(f'hdemucs out - trimmed shape: {x.shape}')
        if return_spec:
            if return_lr_spec:
                return x, x_spec_complex, z
            else:
                return x, x_spec_complex
        return x
# ===频谱网络（结束）================================================================================