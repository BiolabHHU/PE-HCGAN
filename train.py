"""
This code is based on Facebook's HDemucs code: https://github.com/facebookresearch/demucs
"""
import itertools
import logging
import os
import shutil

import hydra
import wandb

from src.ddp.executor import start_ddp_workers
from src.models import modelFactory
from src.utils import print_network
from src.wandb_logger import _init_wandb_run

logger = logging.getLogger(__name__)



def run(args):
    import torch

    from src.ddp import distrib
    from src.data.datasets import LrHrSet
    from src.solver import Solver
    logger.info(f'calling distrib.init')
    distrib.init(args)

    # _init_wandb_run(args)
    # 如果distrib.rank为0，则检查args.samples_dir是否存在。如果存在，则删除该目录，并重新创建它
    if distrib.rank == 0:
        if os.path.exists(args.samples_dir):
            shutil.rmtree(args.samples_dir)
        os.makedirs(args.samples_dir)

    # 设置随机数种子（只影响CPU）
    torch.manual_seed(args.seed)

    # 获取并打印模型
    models = modelFactory.get_model(args)
    for model_name, model in models.items():
        print_network(model_name, model, logger)
    # wandb.watch(tuple(models.values()), log=args.wandb.log, log_freq=args.wandb.log_freq)

    # 显示模型信息和大小，打印后直接返回
    if args.show:
        logger.info(models)
        mb = sum(p.numel() for p in models.parameters()) * 4 / 2 ** 20
        logger.info('Size: %.1f MB', mb)
        return

    # 验证batch_size大小
    assert args.experiment.batch_size % distrib.world_size == 0
    args.experiment.batch_size //= distrib.world_size

    # 构建数据集和加载器
    tr_dataset = LrHrSet(args.dset.train, args.experiment.lr_sr, args.experiment.hr_sr,
                         args.experiment.stride, args.experiment.segment, upsample=args.experiment.upsample)
    tr_loader = distrib.loader(tr_dataset, batch_size=args.experiment.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    # 是否提供验证集
    if args.dset.valid:
        args.valid_equals_test = args.dset.valid == args.dset.test
    if args.dset.valid:
        cv_dataset = LrHrSet(args.dset.valid, args.experiment.lr_sr, args.experiment.hr_sr,
                            stride=None, segment=None, upsample=args.experiment.upsample)
        cv_loader = distrib.loader(cv_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    else:
        cv_loader = None

    # 是否提供测试集
    if args.dset.test:
        tt_dataset = LrHrSet(args.dset.test, args.experiment.lr_sr, args.experiment.hr_sr,
                             stride=None, segment=None, with_path=True, upsample=args.experiment.upsample)
        tt_loader = distrib.loader(tt_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    else:
        tt_loader = None
    data = {"tr_loader": tr_loader, "cv_loader": cv_loader, "tt_loader": tt_loader}

    # 将模型移动到GPU（如果可用）
    if torch.cuda.is_available() and args.device=='cuda':
        for model in models.values():
            model.cuda()

    # 设置优化器
    if args.optim == "adam":
        optimizer = torch.optim.Adam(models['generator'].parameters(), lr=args.lr, betas=(0.9, args.beta2))
    else:
        logger.fatal('Invalid optimizer %s', args.optim)
        os._exit(1)
    # 学习率调度器：余弦退火
    # if args.scheduler == "cosine":
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer,
    #         T_max=args.scheduler_tmax,  # 每个周期的最大迭代步数
    #         eta_min=args.scheduler_min_lr  # 最小学习率
    #     )
    # else:
    #     scheduler = None  # 未启用调度器

    optimizers = {'optimizer': optimizer}

    # if scheduler is not None:
    #     optimizers.update({'scheduler': scheduler})

    # 是否需要对抗训练
    if 'adversarial' in args.experiment and args.experiment.adversarial:
        disc_optimizer = torch.optim.Adam(
            itertools.chain(*[models[disc_name].parameters() for disc_name in
                              args.experiment.discriminator_models]),
            args.lr, betas=(0.9, args.beta2))
        optimizers.update({'disc_optimizer': disc_optimizer})


    # 启动训练
    solver = Solver(data, models, optimizers, args)
    solver.train()

    distrib.close()



def _main(args):
    global __file__
    print(args)
    # Updating paths in config
    # 更新路径为绝对路径
    for key, value in args.dset.items():
        if isinstance(value, str):
            args.dset[key] = hydra.utils.to_absolute_path(value)
    __file__ = hydra.utils.to_absolute_path(__file__)
    # 设置是否输出调试信息
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("src").setLevel(logging.DEBUG)
    # 输出工作目录的路径，打印args对象
    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)
    # 如果 args.ddp 为真且 args.rank 为空，则启动ddp模式，否则，调用run来运行程序
    if args.ddp and args.rank is None:
        start_ddp_workers(args)
    else:
        run(args)

    # wandb.finish()


@hydra.main(config_path="conf", config_name="main_config")  # for latest version of hydra=1.0
def main(args): # 接收main_config的参数作为args
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)


if __name__ == "__main__":
    main()

# python /home/hhdx/PycharmProjects/HCGAN/train.py dset=4-16 experiment=ssrm_4-16_512_64
# python /home/hhdx/PycharmProjects/HCGAN/train.py dset=8-16 experiment=ssrm_8-16_512_64
