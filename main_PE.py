import logging
import os
import torch
import torch.optim as optim
from speech_enhance_function.loss import get_losses, get_losses_with_discriminator
from speech_enhance_function.metrics import get_metrics
from speech_enhance_function.dataloader import create_dataloaders
from speech_enhance_function.model import AudioRestorationModel
from src.models.discriminators import Discriminator
from tqdm import tqdm
import torchaudio
from speech_enhance_function.other import count_parameters, save_model, load_model
from speech_enhance_function.config import TrainingConfig

"""
25.07.15 版本
上一版本：25.07.14 版本
基础框架：dataloader，wav_save，LSD，STFT_Loss，speech_enhance_function, logger, generator
新增：模型保存相关，添加other.py文件，将参数配置移动至config.py文件
"""

# --------------------
# 日志记录器
# --------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    # format='[%(asctime)s-%(name)s-%(levelname)s]-%(message)s',
    # format='[%(asctime)s] %(message)s',
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)

# --------------------
# 训练流程
# --------------------
def train_model(config):
    # 初始化组件
    generator = AudioRestorationModel(config).to(config.device)

    # 变量用于跟踪最佳模型
    best_metric = float('inf')  # 初始化为最大值(假设指标越小越好)

    # 创建模型保存目录
    os.makedirs(config.model_save_dir, exist_ok=True)

    # 打印生成器参数量
    num_params, num_params_m = count_parameters(generator)
    logger.info(f"Generator Parameters: {num_params:,} ({num_params_m:.2f}M)")

    if config.adversarial:
        discriminator = Discriminator(num_D=config.num_D,
                                      ndf=config.ndf,
                                      n_layers=config.n_layers,
                                      downsampling_factor=config.downsampling_factor).to(config.device)
    train_loader, val_loader = create_dataloaders(config)

    # 初始化优化器
    if config.optimizer.lower() == "adam":
        g_optimizer = optim.Adam(generator.parameters(), lr=config.learning_rate)
        if config.adversarial:
            d_optimizer = optim.Adam(discriminator.parameters(), lr=config.disc_lr)
    elif config.optimizer.lower() == "sgd":
        g_optimizer = optim.SGD(generator.parameters(), lr=config.learning_rate)
        if config.adversarial:
            d_optimizer = optim.SGD(discriminator.parameters(), lr=config.disc_lr)

    # 创建保存目录
    os.makedirs(config.save_root, exist_ok=True)

    # 训练循环
    for epoch in range(config.epochs):
        # 训练阶段
        generator.train()
        if config.adversarial:
            discriminator.train()
        train_loss = 0.0

        # 创建训练进度条
        train_loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.epochs} [Train]",
            bar_format="{l_bar}{bar:20}{r_bar}",
            leave=False
        )

        for batch_idx, (de_wav, ta_wav, filenames, _) in enumerate(train_loop):
            de_wav = de_wav.to(config.device, non_blocking=True)
            ta_wav = ta_wav.to(config.device, non_blocking=True)
            pr_wav = generator(de_wav)

            if config.adversarial:
                g_loss, d_loss = get_losses_with_discriminator(config, discriminator, ta_wav, pr_wav)
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
            else:
                g_loss = get_losses(ta_wav, pr_wav)
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

            # 更新统计信息
            train_loss += g_loss.item()
            avg_loss = train_loss / (batch_idx + 1)
            # 实时更新进度条显示
            train_loop.set_postfix({
                "batch_loss": f"{g_loss.item():.4f}",
                "avg_loss": f"{avg_loss:.4f}"
            })

        avg_train_loss = train_loss / len(train_loader)
        train_loop.close()  # 关闭当前进度条

        # 验证阶段
        avg_val_loss = None
        if (epoch + 1) % config.val_interval == 0:
            generator.eval()
            val_loss = 0.0
            LSD = 0.0
            # 清除旧文件（如果需要）
            if config.overwrite_save and epoch > 0:
                for f in os.listdir(config.save_root):
                    if f.startswith("pred_"):
                        os.remove(os.path.join(config.save_root, f))
            # 验证进度条
            val_loop = tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{config.epochs} [Val]  ",
                bar_format="{l_bar}{bar:20}{r_bar}",
                leave=False
            )
            with torch.no_grad():
                for val_batch_idx, (de_wav, ta_wav, filenames, wav_lens) in enumerate(val_loop):
                    de_wav = de_wav.to(config.device, non_blocking=True)
                    ta_wav = ta_wav.to(config.device, non_blocking=True)
                    pr_wav = generator(de_wav)
                    val_loss += get_losses(ta_wav, pr_wav)
                    val_loop.set_postfix({"val_progress": f"{(val_batch_idx + 1) / len(val_loader):.0%}"})

                    ta_wav = ta_wav.cpu()
                    pr_wav = pr_wav.cpu()
                    LSD += get_metrics(ta_wav, pr_wav)

                    # 保存所有样本
                    for i in range(pr_wav.size(0)):
                        # 获取原始长度
                        original_len = wav_lens[i]
                        # 截取原始长度部分
                        audio_tensor = pr_wav[i]
                        audio_tensor = audio_tensor[:, :original_len]

                        # 构建保存路径
                        save_rel_name = f"pred_{os.path.basename(filenames[i])}.wav"
                        save_path = os.path.join(config.save_root, save_rel_name)
                        # 保存音频
                        torchaudio.save(save_path, audio_tensor, config.sr)

            avg_val_loss = val_loss / len(val_loader)
            avg_LSD = LSD / len(val_loader)
            val_loop.close()

            # 保存模型逻辑
            if config.save_model:
                # 始终保存最新模型（覆盖前一个）
                save_model(generator, config, epoch, "latest", avg_LSD)

                # 条件保存最佳模型
                current_metric = avg_LSD if config.best_model_metric == "LSD" else avg_val_loss
                if config.save_best and current_metric < best_metric:
                    best_metric = current_metric
                    save_model(generator, config, epoch, "best", avg_LSD)
                    logger.info(
                        f"New best model at epoch {epoch + 1} with {config.best_model_metric}={best_metric:.4f}")

        # 打印输出结果
        log_msg = f"Epoch [{epoch + 1:02d}/{config.epochs}] | Train Loss: {avg_train_loss:.4f}"
        if avg_val_loss is not None:
            log_msg += f" | Val Loss: {avg_val_loss:.4f} | Val LSD: {avg_LSD:.4f}"
        logger.info(log_msg)


# --------------------
# 主程序
# --------------------
if __name__ == "__main__":
    # 相关参数
    config = TrainingConfig()

    # 模型加载路径(如果需要继续训练或测试)
    load_model_path = None  # 例如: "saved_models/best_model_epoch10_20230715.pth"
    # load_model_path = "saved_models/best_model_epoch45_LSD_0.5014.pth"

    # 快速检查路径是否存在
    assert os.path.exists(config.de_root), f"路径不存在: {config.de_root}"
    assert os.path.exists(config.ta_root), f"路径不存在: {config.ta_root}"

    # 开始训练
    if load_model_path:
        # 加载已有模型
        generator, loaded_config = load_model(load_model_path, config.device)
        # 可以选择使用加载的配置或保持当前配置
        # config = loaded_config  # 取消注释以使用保存的配置
        train_model(config)  # 继续训练
    else:
        # 从头开始训练
        train_model(config)