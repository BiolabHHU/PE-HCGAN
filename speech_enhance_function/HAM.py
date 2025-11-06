import torch
import torch.nn as nn
import torch.nn.functional as F


class HarmonicAttention(nn.Module):
    def __init__(self, in_channels, out_channels, freq_bins, max_freq=4000, max_harmonics=10, embed_dim=64, norm_groups=8):
        super(HarmonicAttention, self).__init__()
        self.freq_bins = freq_bins
        self.max_freq = max_freq
        self.max_harmonics = max_harmonics
        self.embed_dim = embed_dim

        # 嵌入层
        self.freq_embed = nn.Conv2d(1, embed_dim, kernel_size=3, padding=1)
        self.freq_norm = nn.GroupNorm(norm_groups, embed_dim)

        self.spec_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.spec_norm = nn.GroupNorm(norm_groups, embed_dim)

        # 注意力相关层
        self.attn_query = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.query_norm = nn.GroupNorm(norm_groups, embed_dim)

        self.attn_key = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.key_norm = nn.GroupNorm(norm_groups, embed_dim)

        self.attn_value = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.value_norm = nn.GroupNorm(norm_groups, embed_dim)

        # 输出层
        self.out_conv = nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)
        if out_channels == 1:
            self.out_norm = nn.GroupNorm(1, out_channels)  # 对1通道做GroupNorm等价于LayerNorm
        else:
            self.out_norm = nn.GroupNorm(norm_groups, out_channels)

        # 激活函数
        self.activation = nn.GELU()

    def forward(self, spec, f0):
        # 输入：spec (B, 1, F, T), f0 (B, T)
        batch_size, _, freq_bins, time_frames = spec.shape

        harmonics = torch.arange(1, self.max_harmonics + 1, device=f0.device).float()
        harmonic_freqs = f0.unsqueeze(-1) * harmonics  # (B, T, H)

        max_freq = self.max_freq
        freq_indices = (harmonic_freqs / max_freq) * self.freq_bins  # (B, T, H)
        freq_indices = freq_indices.clamp(0, self.freq_bins - 1)

        freq_indices = freq_indices.unsqueeze(1)  # (B, 1, T, H)
        freq_embed = self.freq_embed(freq_indices)  # (B, E, T, H)
        freq_embed = self.activation(self.freq_norm(freq_embed))

        spec_embed = self.spec_embed(spec)  # (B, E, F, T)
        spec_embed = self.activation(self.spec_norm(spec_embed))

        query = self.attn_query(freq_embed)
        query = self.activation(self.query_norm(query))
        query = query.permute(0, 2, 3, 1)  # (B, T, H, E)

        key = self.attn_key(spec_embed)
        key = self.activation(self.key_norm(key))
        key = key.permute(0, 2, 3, 1)  # (B, F, T, E)

        value = self.attn_value(spec_embed)
        value = self.activation(self.value_norm(value))
        value = value.permute(0, 2, 3, 1)  # (B, F, T, E)

        attn_scores = torch.matmul(query, key.permute(0, 2, 3, 1))  # (B, T, H, F)
        attn_weights = F.softmax(attn_scores / (self.embed_dim ** 0.5), dim=-1)  # (B, T, H, F)

        attn_weights = torch.mean(attn_weights, dim=2, keepdim=True)  # (B, T, 1, F)
        attn_weights = attn_weights.permute(0, 2, 3, 1)  # (B, 1, F, T)

        harmonic_enhanced = spec_embed * attn_weights  # (B, E, F, T)
        enhanced_spec = spec_embed + harmonic_enhanced  # (B, E, F, T)

        output = self.out_conv(enhanced_spec)  # (B, 1, F, T)
        output = self.out_norm(output)
        output = output + spec  # 残差连接

        return output


# 示例用法
if __name__ == "__main__":
    batch_size, freq_bins, time_frames = 2, 256, 100
    spec = torch.randn(batch_size, 16, freq_bins, time_frames)
    f0 = torch.abs(torch.randn(batch_size, time_frames)) * 500 + 100

    model = HarmonicAttention(in_channels=16, out_channels=16, freq_bins=freq_bins, max_freq=4000, max_harmonics=10)
    enhanced_spec = model(spec, f0)

    print(f"输入形状: {spec.shape}")
    print(f"输出形状: {enhanced_spec.shape}")
