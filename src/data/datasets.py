"""
This code is based on Facebook's HDemucs code: https://github.com/facebookresearch/demucs
"""

import json
import logging
import os

import torch
from tqdm import tqdm
import torchaudio

from torch.nn import functional as F
from torchaudio.functional import resample
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram

from src.data.audio import Audioset
from src.utils import match_signal

logger = logging.getLogger(__name__)


def match_files(lr, hr):    # 排序文件使lr，hr匹配
    """match_files.
    Sort files to match lr and hr filenames.
    :param lr: list of the low-resolution filenames
    :param hr: list of the high-resolution filenames
    """
    lr.sort()
    hr.sort()


def assert_sets(lr_set, hr_set):    # 验证lr_set, hr_set中元素形状是否相同
    n_samples = len(lr_set)
    for i in tqdm(range(n_samples)):
        assert lr_set[i].shape == hr_set[i].shape


def match_source_to_target_length(source_sig, target_sig):  # 调整source_sig的长度，使其与target_sig的长度相匹配
    target_len = target_sig.shape[-1]
    source_len = source_sig.shape[-1]
    if target_len < source_len:
        source_sig = source_sig[..., :target_len]
    elif target_len > source_len:
        source_sig = F.pad(source_sig, (0, target_len - source_len))
    return source_sig


class PrHrSet(Dataset):     # 加载和预处理包含低分辨率（lr）、高分辨率（hr）和预测（pr）音频文件的数据集
    def __init__(self, samples_dir, filenames=None):
        self.samples_dir = samples_dir
        if filenames is not None:
            files = [i for i in os.listdir(samples_dir) if any(i for j in filenames if j in i)]
        else:
            files = os.listdir(samples_dir)

        self.hr_filenames = list(sorted(filter(lambda x: x.endswith('_hr.wav'), files)))
        self.lr_filenames = list(sorted(filter(lambda x: x.endswith('_lr.wav'), files)))
        self.pr_filenames = list(sorted(filter(lambda x: x.endswith('_pr.wav'), files)))

    def __len__(self):
        return len(self.hr_filenames)

    def __getitem__(self, i):
        lr_i, lr_sr = torchaudio.load(os.path.join(self.samples_dir, self.lr_filenames[i]))
        hr_i, hr_sr = torchaudio.load(os.path.join(self.samples_dir, self.hr_filenames[i]))
        pr_i, pr_sr = torchaudio.load(os.path.join(self.samples_dir, self.pr_filenames[i]))
        pr_i = match_signal(pr_i, hr_i.shape[-1])
        assert hr_i.shape == pr_i.shape
        lr_filename = self.lr_filenames[i]

        lr_filename = lr_filename[:lr_filename.index('_lr.wav')]
        hr_filename = self.hr_filenames[i]
        hr_filename = hr_filename[:hr_filename.index('_hr.wav')]
        pr_filename = self.pr_filenames[i]
        pr_filename = pr_filename[:pr_filename.index('_pr.wav')]
        assert lr_filename == hr_filename == pr_filename

        return lr_i, hr_i, pr_i, lr_filename


class LrHrSet(Dataset):     # 加载包含低分辨率（LR）和高分辨率（HR）音频数据的数据集，并进行预处理
    def __init__(self, json_dir, lr_sr, hr_sr, stride=None, segment=None,
                 pad=True, with_path=False, stft=False, win_len=64, hop_len=16, n_fft=4096, complex_as_channels=True,
                 upsample=True):
        """__init__.
        :param json_dir: directory containing both hr.json and lr.json
        :param stride: the stride used for splitting audio sequences in seconds
        :param segment: the segment length used for splitting audio sequences in seconds
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        :param with_path: whether to return tensors with filepath
        :param stft: convert to spectrogram
        :param win_len: stft window length in seconds
        :param hop_len: stft hop length in seconds
        :param n_fft: stft number of frequency bins
        :param complex_as_channels: True - move complex dimension to channel dimension. output is [2, Fr, T]
                                    False - last dimension is complex channels, output is [1, Fr, T, 2]
        """

        self.lr_sr = lr_sr
        self.hr_sr = hr_sr
        self.stft = stft
        self.with_path = with_path
        self.upsample = upsample

        if self.stft:
            self.window_length = int(self.hr_sr / 1000 * win_len)  # 64 ms
            self.hop_length = int(self.hr_sr / 1000 * hop_len)  # 16 ms
            self.window = torch.hann_window(self.window_length)
            self.n_fft = n_fft
            self.complex_as_channels = complex_as_channels
            # 生成频谱图（stft）
            self.spectrogram = Spectrogram(n_fft=n_fft, win_length=self.window_length, hop_length=self.hop_length,
                                           power=None)
        # 路径拼接
        lr_json = os.path.join(json_dir, 'lr.json')
        hr_json = os.path.join(json_dir, 'hr.json')
        # 加载JSON文件
        with open(lr_json, 'r') as f:
            lr = json.load(f)
        with open(hr_json, 'r') as f:
            hr = json.load(f)
        # 计算步长和长度
        lr_stride = stride * lr_sr if stride else None
        hr_stride = stride * hr_sr if stride else None
        lr_length = segment * lr_sr if segment else None
        hr_length = segment * hr_sr if segment else None
        # 确保lr, hr匹配，处理lr, hr
        match_files(lr, hr)
        self.lr_set = Audioset(lr, sample_rate=lr_sr, length=lr_length, stride=lr_stride, pad=pad, channels=1,
                               with_path=with_path)
        self.hr_set = Audioset(hr, sample_rate=hr_sr, length=hr_length, stride=hr_stride, pad=pad, channels=1,
                               with_path=with_path)
        assert len(self.hr_set) == len(self.lr_set)

    def __getitem__(self, index):
        if self.with_path:  # 是否返回音频路径
            hr_sig, hr_path = self.hr_set[index]
            lr_sig, lr_path = self.lr_set[index]
        else:
            hr_sig = self.hr_set[index]
            lr_sig = self.lr_set[index]
        if self.upsample:   # 上采样低分辨率信号，此处的上采样是使得低分辨率与高分辨率相匹配，并非预测
            lr_sig = resample(lr_sig, self.lr_sr, self.hr_sr)
            lr_sig = match_signal(lr_sig, hr_sig.shape[-1])     # 确保长度匹配

        if self.stft:   # 计算短时傅里叶变换
            hr_sig = torch.view_as_real(self.spectrogram(hr_sig))
            lr_sig = torch.view_as_real(self.spectrogram(lr_sig))
            if self.complex_as_channels:
                Ch, Fr, T, _ = hr_sig.shape
                hr_sig = hr_sig.reshape(2 * Ch, Fr, T)
                lr_sig = lr_sig.reshape(2 * Ch, Fr, T)

        if self.with_path:
            return (lr_sig, lr_path), (hr_sig, hr_path)
        else:
            return lr_sig, hr_sig

    def __len__(self):
        return len(self.lr_set)


if __name__ == "__main__":
    json_dir = '../egs/vctk/16-24/val'
    lr_sr = 16000
    hr_sr = 24000
    pad = True
    stride_sec = 2
    segment_sec = 2

    data_set = LrHrSet(json_dir, lr_sr, hr_sr, stride_sec, segment_sec)
    assert_sets(data_set.lr_set, data_set.hr_set)
    print(f'done asserting dataset from {json_dir}')
