# other.py
import os
import glob
import torch
import time
import logging
# from speech_enhance_function.config import TrainingConfig

logger = logging.getLogger(__name__)

def count_parameters(model):
    """统计模型的可训练参数量和内存占用(MB)"""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params_m = num_params / 1e6
    return num_params, num_params_m

def save_model(model, config, epoch, model_type, metric_value=None):
    """保存模型到文件，文件名包含LSD指标

    参数:
        model_type: "latest" 或 "best"
        metric_value: 当前的LSD指标值
    """
    # 删除同类型的旧模型
    for old_file in glob.glob(os.path.join(config.model_save_dir, f"{model_type}_model_*.pth")):
        try:
            os.remove(old_file)
            logger.debug(f"Removed old {model_type} model: {os.path.basename(old_file)}")
        except OSError as e:
            logger.warning(f"Failed to remove {old_file}: {e}")

    # 构建包含指标的文件名
    if metric_value is not None:
        filename = f"{model_type}_model_epoch{epoch + 1}_LSD_{metric_value:.4f}.pth"
    else:
        filename = f"{model_type}_model_epoch{epoch + 1}.pth"

    save_path = os.path.join(config.model_save_dir, filename)

    # 确保目录存在
    os.makedirs(config.model_save_dir, exist_ok=True)

    # 保存模型
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'config': vars(config),
        'save_time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'LSD': metric_value
    }, save_path)

    logger.info(
        f"Saved {model_type} model (epoch {epoch + 1}, LSD {metric_value if metric_value else 'N/A'}) to {save_path}")

def load_model(model_path, device, config=None):
    """从文件加载模型"""
    checkpoint = torch.load(model_path, map_location=device)

    if config is None:
        config_dict = checkpoint['config']
        from speech_enhance_function.config import TrainingConfig  # 避免循环导入
        config = TrainingConfig()
        for key, value in config_dict.items():
            setattr(config, key, value)

    from speech_enhance_function.model import AudioRestorationModel  # 延迟导入
    model = AudioRestorationModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(
        f"Loaded model from {model_path} (epoch {checkpoint['epoch']}, saved at {checkpoint.get('save_time', 'unknown')})")
    return model, config