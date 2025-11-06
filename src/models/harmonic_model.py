
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


import logging
logger = logging.getLogger(__name__)


class SEBlock(nn.Module):
    """改进的SE模块，适配动态尺寸"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1)

class UNetHarmonic1(nn.Module):
    def __init__(self):
        super(UNetHarmonic1, self).__init__()

        # 编码器（双维度下采样）
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            SEBlock(16)
        )
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 双维度下采样

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GELU(),
            SEBlock(32)
        )
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # 中间层（无下采样）
        self.mid = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            SEBlock(64)
        )

        # 解码器（只包含卷积和处理，不包含上采样）
        self.up1 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1),  # 64 (上采样) + 32 (enc2) = 96
            nn.GELU(),
            SEBlock(32)
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(32 + 16, 16, 3, padding=1),  # 32 (上采样) + 16 (enc1) = 48
            nn.GELU(),
            SEBlock(16)
        )

        # 最终扩展层
        self.final_expand = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        # 保存原始尺寸
        orig_n, orig_t = x.shape[2], x.shape[3]

        # Encoder
        e1 = self.enc1(x)  # [B, 16, N, T]
        p1 = self.pool1(e1)  # [B, 16, ceil(N/2), ceil(T/2)]

        e2 = self.enc2(p1)  # [B, 32, ceil(N/2), ceil(T/2)]
        p2 = self.pool2(e2)  # [B, 32, ceil(N/4), ceil(T/4)]

        # Middle
        m = self.mid(p2)  # [B, 64, ceil(N/4), ceil(T/4)]

        # Decoder with skip connections
        d1_up = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 64, ceil(N/2), ceil(T/2)]
        d1_concat = self._crop_and_concat(d1_up, e2)  # [B, 64+32=96, ceil(N/2), ceil(T/2)]
        d1 = self.up1(d1_concat)  # [B, 32, ceil(N/2), ceil(T/2)]

        d2_up = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 32, N, T]
        d2_concat = self._crop_and_concat(d2_up, e1)  # [B, 32+16=48, N, T]
        d2 = self.up2(d2_concat)  # [B, 16, N, T]

        # Final output
        output = self.final_expand(d2)  # [B, 1, N, T]
        return output

    def _crop_and_concat(self, upsampled, bypass):
        """裁剪或填充 bypass 以匹配 upsampled 的尺寸并拼接"""
        up_h, up_w = upsampled.shape[2], upsampled.shape[3]
        by_h, by_w = bypass.shape[2], bypass.shape[3]

        # 如果 bypass 尺寸大于 upsampled，则裁剪
        if by_h > up_h or by_w > up_w:
            diff_h = by_h - up_h
            diff_w = by_w - up_w
            bypass = bypass[:, :, diff_h//2:-(diff_h//2) if diff_h > 0 else None,
                          diff_w//2:-(diff_w//2) if diff_w > 0 else None]
        # 如果 bypass 尺寸小于 upsampled，则填充
        elif by_h < up_h or by_w < up_w:
            pad_h = up_h - by_h
            pad_w = up_w - by_w
            bypass = F.pad(bypass, (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2))

        return torch.cat((upsampled, bypass), dim=1)

class UNetHarmonic2(nn.Module):
    def __init__(self):
        super(UNetHarmonic2, self).__init__()
        self.encoder1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.encoder3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.encoder4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2)

        # Decoder部分的卷积层输入通道数修改以适应拼接后的通道数
        self.decoder1 = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1)
        self.decoder2 = nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1)
        self.decoder3 = nn.Conv2d(32 + 16, 16, kernel_size=3, padding=1)
        self.decoder4 = nn.Conv2d(16, 1, kernel_size=3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=1)
        self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=1)
        self.upconv3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=1)
        self.upconv4 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=1)

        # self.norm0 = nn.BatchNorm2d(1)
        # self.norm1 = nn.BatchNorm2d(16)
        # self.norm2 = nn.BatchNorm2d(32)
        # self.norm3 = nn.BatchNorm2d(64)
        # self.norm4 = nn.BatchNorm2d(128)

    def forward(self, x):
        length = x.shape[-1]
        # Encoder path
        enc1 = F.gelu(self.encoder1(x))
        enc2 = F.gelu(self.encoder2(enc1))
        enc3 = F.gelu(self.encoder3(enc2))
        enc4 = F.gelu(self.encoder4(enc3))

        # Decoder path
        dec1 = F.gelu(self.upconv1(enc4))
        dec1 = self._crop_and_concat(dec1, enc3)

        dec2 = F.gelu(self.upconv2(F.gelu(self.decoder1(dec1))))
        dec2 = self._crop_and_concat(dec2, enc2)
        # dec2 = F.gelu(self.upconv2(enc3))
        # dec2 = self._crop_and_concat(dec2, enc2)

        dec3 = F.gelu(self.upconv3(F.gelu(self.decoder2(dec2))))
        dec3 = self._crop_and_concat(dec3, enc1)

        dec4 = self.decoder4(F.gelu(self.upconv4(F.gelu(self.decoder3(dec3)))))
        output = dec4[..., :length]
        return output

    def _crop_and_concat(self, upsampled, bypass):
        """裁剪bypass并与upsampled连接"""
        if upsampled.shape[-2] != bypass.shape[-2]:
            diff = upsampled.shape[-2] - bypass.shape[-2]
            bypass = F.pad(bypass, (0, 0, diff // 2, diff - diff // 2))

        if upsampled.shape[-1] != bypass.shape[-1]:
            diff = upsampled.shape[-1] - bypass.shape[-1]
            bypass = F.pad(bypass, (diff // 2, diff - diff // 2, 0, 0))

        return torch.cat((upsampled, bypass), dim=1)

class ResidualBlockTime(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        # self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0))
        self.activation = nn.GELU()
        self.se = SEBlock(out_channels)

    def forward(self, x):
        residual = x
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        x = self.activation(x + residual)
        x = self.se(x)
        return x

class ResidualBlockFreq(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        # self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0))
        self.activation = nn.GELU()
        self.se = SEBlock(out_channels)

    def forward(self, x):
        residual = x
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        x = self.activation(x + residual)
        x = self.se(x)
        return x

class FusionHarmonicPredictor(nn.Module):
    def __init__(self, in_channels=1, mid_channels=16, out_channels=1):
        super().__init__()

        # 初始升维
        self.input_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.activation = nn.GELU()

        # 交替的时频残差块
        self.res_blocks_1 = nn.Sequential(
            nn.Sequential(
                ResidualBlockTime(mid_channels, mid_channels),
                ResidualBlockFreq(mid_channels, mid_channels)
            ),
            nn.Sequential(
                ResidualBlockTime(mid_channels, mid_channels),
                ResidualBlockFreq(mid_channels, mid_channels)
            )
        )

        # 融合分支1特征（在中间维度拼接）
        self.fusion_conv = nn.Conv2d(mid_channels + in_channels, mid_channels, kernel_size=3, padding=1)

        # 再次交替残差块（可以不加也可加，灵活调整）
        self.res_blocks_2 = nn.Sequential(
            nn.Sequential(
                ResidualBlockTime(mid_channels, mid_channels),
                ResidualBlockFreq(mid_channels, mid_channels)
            ),
            nn.Sequential(
                ResidualBlockTime(mid_channels, mid_channels),
                ResidualBlockFreq(mid_channels, mid_channels)
            )
        )

        # 输出降维
        self.output_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, low_freq_harmonic, harmonic_from_branch1):
        """
        low_freq_harmonic: [B, 1, 5, T]
        harmonic_from_branch1: [B, 1, 5, T]
        """
        x = self.activation(self.input_conv(low_freq_harmonic))
        x = self.res_blocks_1(x)

        # 确保辅助特征大小匹配（如需要插值，理论不用）
        if harmonic_from_branch1.shape[2:] != x.shape[2:]:
            harmonic_from_branch1 = F.interpolate(harmonic_from_branch1, size=x.shape[2:], mode='nearest')

        # 拼接来自分支1的辅助谐波
        x = self.fusion_conv(torch.cat([x, harmonic_from_branch1], dim=1))

        x = self.res_blocks_2(x)

        out = self.output_conv(x)
        return out

