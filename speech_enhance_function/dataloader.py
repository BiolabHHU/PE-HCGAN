"""
数据加载相关
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import librosa

# --------------------
# 数据集类
# --------------------
class AudioDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.de_files = self._collect_files(config.de_root)
        self.ta_files = self._collect_files(config.ta_root)

        # 验证数据一致性
        self._validate_file_structure()
        # 新增文件名对存储
        self.file_pairs = list(zip(self.de_files, self.ta_files))

    def _collect_files(self, root_dir):
        """递归收集所有音频文件路径"""
        file_list = []
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.endswith(self.config.audio_ext):
                    full_path = os.path.join(dirpath, fname)
                    file_list.append(full_path)
        return sorted(file_list)

    def _validate_file_structure(self):
        """验证文件结构一致性"""
        assert len(self.de_files) == len(self.ta_files), "文件数量不匹配"

        for de_path, ta_path in zip(self.de_files, self.ta_files):
            # 获取相对路径
            de_rel = os.path.relpath(de_path, self.config.de_root)
            ta_rel = os.path.relpath(ta_path, self.config.ta_root)

            # 统一路径分隔符（防止Windows/Linux差异）
            de_rel = de_rel.replace("\\", "/")
            ta_rel = ta_rel.replace("\\", "/")

            assert de_rel == ta_rel, f"文件结构不匹配:\nA: {de_rel}\nC: {ta_rel}"

    def _load_wav(self, file_path):
        """加载音频并转为幅度谱"""
        # 加载音频
        wav, _ = librosa.load(file_path, sr=self.config.sr)

        return torch.FloatTensor(wav).unsqueeze(0)

    def __len__(self):
        return len(self.de_files)

    def __getitem__(self, idx):
        de_path, ta_path = self.file_pairs[idx]  # 获取原始路径
        de_wav = self._load_wav(self.de_files[idx])
        ta_wav = self._load_wav(self.ta_files[idx])
        base_name = os.path.splitext(os.path.basename(ta_path))[0]
        wav_len = ta_wav.shape[1]
        return de_wav, ta_wav, base_name, wav_len


# --------------------
# 创建数据加载器
# --------------------
def create_dataloaders(config):
    # 创建完整数据集
    full_dataset = AudioDataset(config)

    # 划分训练集和验证集
    train_size = int(config.train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    # 创建DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=dynamic_pad_collate,  # 使用自定义collate
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=dynamic_pad_collate,  # 使用自定义collate
        pin_memory=True
    )

    return train_loader, val_loader


def dynamic_pad_collate(batch):
    de_wavs, ta_wavs, filenames, wav_lens = zip(*batch)

    # 计算最大时间维度
    max_time = max(
        max(wav.shape[-1] for wav in de_wavs),
        max(wav.shape[-1] for wav in ta_wavs)
    )

    # 定义填充函数
    def pad_wav(wav):
        pad_size = max_time - wav.shape[-1]
        return torch.nn.functional.pad(wav, (0, pad_size))

    # 填充所有样本
    de_padded = torch.stack([pad_wav(wav) for wav in de_wavs], dim=0)
    ta_padded = torch.stack([pad_wav(wav) for wav in ta_wavs], dim=0)

    return de_padded, ta_padded, filenames, wav_lens