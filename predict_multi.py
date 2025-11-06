import math
import os
import time
import hydra
import torch
import logging
from pathlib import Path
import torchaudio
from torchaudio.functional import resample
from src.enhance import write
from src.models import modelFactory
from src.model_serializer import SERIALIZE_KEY_MODELS, SERIALIZE_KEY_BEST_STATES, SERIALIZE_KEY_STATE
from src.utils import bold
from memory_profiler import profile  # 引入内存监测库

logger = logging.getLogger(__name__)

SEGMENT_DURATION_SEC = 10


def _load_model(args):
    model_name = args.experiment.model
    checkpoint_file = Path(args.checkpoint_file)
    model = modelFactory.get_model(args)['generator']
    package = torch.load(checkpoint_file, 'cpu')
    load_best = args.continue_best
    if load_best:
        logger.info(bold(f'Loading model {model_name} from best state.'))
        model.load_state_dict(
            package[SERIALIZE_KEY_BEST_STATES][SERIALIZE_KEY_MODELS]['generator'][SERIALIZE_KEY_STATE])
    else:
        logger.info(bold(f'Loading model {model_name} from last state.'))
        model.load_state_dict(package[SERIALIZE_KEY_MODELS]['generator'][SERIALIZE_KEY_STATE])

    return model


def process_single_file(model, device, filename, output_dir, args):
    file_basename = Path(filename).stem
    lr_sig, sr = torchaudio.load(str(filename))

    if args.experiment.upsample:
        lr_sig = resample(lr_sig, sr, args.experiment.hr_sr)
        sr = args.experiment.hr_sr

    logger.info(f'lr wav shape: {lr_sig.shape}')

    segment_duration_samples = sr * SEGMENT_DURATION_SEC
    n_chunks = math.ceil(lr_sig.shape[-1] / segment_duration_samples)
    logger.info(f'number of chunks: {n_chunks}')

    lr_chunks = []
    for i in range(n_chunks):
        start = i * segment_duration_samples
        end = min((i + 1) * segment_duration_samples, lr_sig.shape[-1])
        lr_chunks.append(lr_sig[:, start:end])

    pr_chunks = []

    model.eval()
    pred_start = time.time()
    with torch.no_grad():
        for i, lr_chunk in enumerate(lr_chunks):
            pr_chunk = model(lr_chunk.unsqueeze(0).to(device)).squeeze(0)
            logger.info(f'lr chunk {i} shape: {lr_chunk.shape}')
            logger.info(f'pr chunk {i} shape: {pr_chunk.shape}')
            pr_chunks.append(pr_chunk.cpu())

    pred_duration = time.time() - pred_start
    logger.info(f'prediction duration: {pred_duration}')

    pr = torch.concat(pr_chunks, dim=-1)

    logger.info(f'pr wav shape: {pr.shape}')

    out_filename = os.path.join(output_dir, file_basename + '_pr.wav')
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f'saving to: {out_filename}, with sample_rate: {args.experiment.hr_sr}')
    write(pr, out_filename, args.experiment.hr_sr)


# @profile  # 使用装饰器监测内存
@hydra.main(config_path="conf", config_name="main_config")
def main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)

    print(args)
    model = _load_model(args)
    device = torch.device('cuda')
    model.cuda()

    input_path = Path(args.filename)
    output_dir = args.output

    if input_path.is_file() and input_path.suffix.lower() == '.wav':
        # 处理单个文件
        process_single_file(model, device, input_path, output_dir, args)
    elif input_path.is_dir():
        # 处理文件夹中的所有.wav文件
        wav_files = list(input_path.glob('*.wav'))
        logger.info(f'Found {len(wav_files)} WAV files in directory {input_path}')

        for wav_file in wav_files:
            logger.info(f'Processing file: {wav_file}')
            try:
                process_single_file(model, device, wav_file, output_dir, args)
            except Exception as e:
                logger.error(f'Error processing file {wav_file}: {str(e)}')
                continue
    else:
        logger.error(f'Input path {input_path} is neither a WAV file nor a directory')


"""
Need to add filename and output to args.
Usage for single file: 
python predict.py dset=4-16 experiment=ssrm_4-16_512_64 +filename=/path/to/file.wav +output=/path/to/output
python predict.py dset=8-16 experiment=ssrm_8-16_512_64 +filename=/home/hhdx/PycharmProjects/speech_2.4k_8k/output/pred_p351_046.wav +output=/home/hhdx/PycharmProjects/speech_2.4k_8k/output_pr/
Usage for directory: 
python predict.py dset=4-16 experiment=ssrm_4-16_512_64 +filename=/path/to/folder +output=/path/to/output
python predict_multi.py dset=8-16 experiment=ssrm_8-16_512_64 +filename=/home/hhdx/PycharmProjects/speech_2.4k_8k/output +output=/home/hhdx/PycharmProjects/speech_2.4k_8k/output_pr
python predict_multi.py dset=8-16 experiment=ssrm_8-16_512_64 +filename=/home/hhdx/PycharmProjects/speech_2.4k_8k/de +output=/home/hhdx/PycharmProjects/speech_2.4k_8k/de_pr
"""

if __name__ == "__main__":
    main()