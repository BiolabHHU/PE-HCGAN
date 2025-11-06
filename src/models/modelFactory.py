from src.models.ssrm import SSRM
# from src.models.ssrm_first import SSRM

from src.models.seanet import Seanet
from src.models.discriminators import Discriminator, MultiPeriodDiscriminator, MultiScaleDiscriminator


def get_model(args):
    if args.experiment.model == 'ssrm':
        generator = SSRM(**args.experiment.ssrm)
    elif args.experiment.model == 'seanet':
        generator = Seanet(**args.experiment.seanet)
    models = {'generator': generator}

    # if 'pitch' in args.experiment and args.experiment.pitch:
    #     if 'pipr' in args.experiment.model_picth:
    #         generator_pitch = Pipr(**args.experiment.pipr)
    #         models.update({'generator_pitch': generator_pitch})

    if 'adversarial' in args.experiment and args.experiment.adversarial:
        if 'msd_melgan' in args.experiment.discriminator_models:
            discriminator = Discriminator(**args.experiment.melgan_discriminator)
            models.update({'msd_melgan': discriminator})
        if 'msd_hifi' in args.experiment.discriminator_models:
            msd = MultiScaleDiscriminator(**args.experiment.msd)
            models.update({'msd': msd})
        if 'mpd' in args.experiment.discriminator_models:
            mpd = MultiPeriodDiscriminator(**args.experiment.mpd)
            models.update({'mpd': mpd})
        if 'hifi' in args.experiment.discriminator_models:
            mpd = MultiPeriodDiscriminator(**args.experiment.mpd)
            msd = MultiScaleDiscriminator(**args.experiment.msd)
            models.update({'mpd': mpd, 'msd': msd})

    return models