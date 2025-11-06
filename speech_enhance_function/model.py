"""
模型结构相关
"""
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from speech_enhance_function.spec import spectro, ispectro
from speech_enhance_function.HAM import HarmonicAttention


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    # format='[%(asctime)s-%(name)s-%(levelname)s]-%(message)s',
    # format='[%(asctime)s] %(message)s',
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)

class Encoder_layer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 ):
        super().__init__()
        padding = kernel_size // 2

        self.conv_down = self._block(in_channels, out_channels, kernel_size, stride=(stride, 1), padding=padding)
        self.conv = self._block(out_channels, out_channels, kernel_size, padding=padding)
        self.conv_f = self._block(out_channels, out_channels, kernel_size, padding=padding, mode='F')
        self.conv_t = self._block(out_channels, out_channels, kernel_size, padding=padding, mode='T')

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.ac = nn.GELU()

    def _block(self, in_channels, out_channels, kernel_size=3, stride=(1, 1), padding=1, mode=None):

        if mode == 'F':
            kernel_size = (kernel_size, 1)
            padding = (padding, 0)
        elif mode == 'T':
            kernel_size = (1, kernel_size)
            padding = (0, padding)
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),  # 保持尺寸
        )

    def forward(self, x):
        x = self.conv_down(x)
        # x = self.bn1(x)
        x_0 = self.ac(x)
        x_t = self.conv_t(x_0)
        x_f = self.conv_f(x_t + x_0)
        out = self.ac(x_0 + x_f)
        return out

class Decoder_layer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 last = False,
                 ):
        super().__init__()
        padding = kernel_size // 2
        self.last = last

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.ac = nn.GELU()

    def forward(self, x):
        x_0 = self.conv1x1(x)
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.ac(x)
        x = self.conv2(x)
        out = x + x_0
        if not self.last:
            out = self.ac(out)
        return out

class DualAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        # 通道压缩
        self.f_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),  # 沿时间轴压缩 => [B, C, F, 1]
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.t_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),  # 沿频率轴压缩 => [B, C, 1, T]
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, F, T]
        f_weight = self.f_attn(x)  # [B, C, F, 1]
        t_weight = self.t_attn(x)  # [B, C, 1, T]
        out = x * f_weight * t_weight  # 广播机制
        return out

class Audio_Net(nn.Module):
    def __init__(self,
                 in_channels=2,
                 out_channels=2,
                 channels=16,
                 kernel_size=3):
        super().__init__()
        # 假设输入频谱形状: [B, 2, F, T] (2通道: 实部和虚部)

        self.pre_conv = nn.Conv2d(in_channels, channels, kernel_size=1)
        self.en1 = Encoder_layer(channels, channels*2, kernel_size, 2)
        self.en2 = Encoder_layer(channels*2, channels*4, kernel_size, 2)
        self.en3 = Encoder_layer(channels*4, channels*8, kernel_size, 2)
        self.en4 = Encoder_layer(channels*8, channels*16, kernel_size, 2)

        self.middle = nn.Sequential(
            nn.Conv2d(channels*16, channels*16, kernel_size=3, padding=1),
            nn.GELU(),
            DualAttention(channels*16),
            nn.Conv2d(channels*16, channels*16, kernel_size=3, padding=1),
            nn.GELU()
        )

        self.de4 = Decoder_layer(channels*16, channels*8, kernel_size, False)
        self.de3 = Decoder_layer(channels*8, channels*4, kernel_size, False)
        self.de2 = Decoder_layer(channels*4, channels*2, kernel_size, False)
        self.de1 = Decoder_layer(channels*2, in_channels, kernel_size, True)

    def _upsample_freq(self, x, scale_factor=2, mode='bilinear'):
        # x shape: [B, C, F, T]
        return F.interpolate(x, scale_factor=(scale_factor, 1), mode=mode, align_corners=True)


    def forward(self, x):
        """输入x形状: [B, 2, F, T] (实部和虚部)"""
        x = self.pre_conv(x)
        e1 = self.en1(x)
        e2 = self.en2(e1)
        e3 = self.en3(e2)
        e4 = self.en4(e3)

        mid = self.middle(e4)

        up4 = self._upsample_freq(mid, 2)
        d4 = self.de4(up4)
        up3 = self._upsample_freq(d4, 2)
        d3 = self.de3(up3)
        up2 = self._upsample_freq(d3, 2)
        d2 = self.de2(up2)
        up1 = self._upsample_freq(d2, 2)
        d1 = self.de1(up1)

        out = d1
        return out

class AudioRestorationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.nfft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length


        self.model = Audio_Net()
        self.HAM = HarmonicAttention(in_channels=1, out_channels=1, freq_bins=self.nfft/2, max_freq=4000, max_harmonics=10)

    def _spec(self, x):    # 将输入的音频信号计算其频谱图，并根据需要进行长度调整和缩放
        if np.mod(x.shape[-1], self.hop_length):
            x = F.pad(x, (0, self.hop_length - np.mod(x.shape[-1], self.hop_length)))
        hl = self.hop_length
        nfft = self.nfft
        win_length = self.win_length

        z = spectro(x, nfft, hl, win_length=win_length)[..., :-1, :]
        return z

    def _ispec(self, z):    # 逆转 _spec 方法的操作，它将频谱图转换回原始的音频信号
        hl = self.hop_length
        win_length = self.win_length
        z = F.pad(z, (0, 0, 0, 1))
        x = ispectro(z, hl, win_length=win_length)
        return x

    def _move_complex_to_channels_dim(self, z):  # 将复数表示的频谱图中的实部和虚部移动到通道维度，将其重新组织为一个具有两倍通道数的张量
        B, C, F, T = z.shape
        m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
        m = m.reshape(B, C * 2, F, T)
        return m

    def _convert_to_complex(self, x):  # 将输入张量表示的实部和虚部分开的复数信号重新组合成一个复数信号张量

        out = x.permute(0, 1, 3, 4, 2)
        out = torch.view_as_complex(out.contiguous())
        return out

    def forward(self, x):
        length = x.shape[-1]

        # 时频转换，通道分离
        x = self._spec(x)
        x = self._move_complex_to_channels_dim(x)
        B, C, Fq, T = x.shape

        # 归一化
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        # logger.info(f"x1 :{x.shape}")

        # 拆分实部和虚部
        real, imag = torch.split(x, x.size(1) // 2, dim=1)  # 各 (B, C, Fr, T)
        # 模块
        f0 = torch.full((B, T), 200).to('cuda:0')  # 此处选用可调整的参数形式，以200为先验基频的情况下仍能取得增强效果
        real = self.HAM(real, f0)
        # 恢复为 x 的格式
        x = torch.cat([real, imag], dim=1)  # shape: (B, C*2, Fr, T)
        # logger.info(f"x2 :{x.shape}")

        x = self.model(x)

        # 逆归一化
        x = x.view(B, 1, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        # 通道合并，频时转化
        x = self._convert_to_complex(x)
        x = self._ispec(x)

        # 剪裁STFT填充的长度
        out = x[..., :int(length)]

        return out
