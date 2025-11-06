import torch

# --------------------
# 全局训练配置
# --------------------
class TrainingConfig:
    # 路径参数

    de_root = ""  # 编码音频根目录
    ta_root = ""  # 原始音频根目录

    save_root = ""  # 保存路径
    overwrite_save = True  # 是否覆盖已有文件

    # 音频处理参数
    sr = 8000  # 采样率
    n_fft = 512  # FFT点数
    hop_length = 64  # 步长
    win_length = 256  # 窗长
    audio_ext = ".wav"  # 音频文件扩展名

    # 训练参数
    batch_size = 8
    epochs = 90
    learning_rate = 2e-4
    train_ratio = 0.95  # 训练集比例
    val_interval = 5  # 验证间隔参数

    # 系统参数
    device = torch.device('cuda:0')
    num_workers = 0  # 数据加载线程数

    # 优化器配置
    optimizer = "adam"  # 可选 ["adam", "sgd"]

    # 判别器参数
    adversarial = 0
    disc_lr = 1e-5
    num_D = 2
    ndf = 8
    n_layers = 3
    downsampling_factor = 3

    # 模型保存配置
    save_model = True  # 是否保存模型
    save_best = True  # 是否保存最佳模型
    model_save_dir = "saved_models"  # 模型保存目录
    best_model_metric = "LSD"  # 用于判断最佳模型的指标("LSD"或"val_loss")