# PE-HCGAN
Speech super-resolution

# File Overview
- `main_config.yaml` and `ssrm_4-16_512_64.yaml` - Parameter settings

- `ssrm.py` - Main model

- `harmonic_model.py` - Model for predicting harmonics

- `create_meta_files.py` - Used to create corresponding path files for training and validation.
  
  Usage:
  
  `python data_prep/create_meta_files.py <path for 4 kHz data> egs/vctk/4-16 lr`
  
  `python data_prep/create_meta_files.py <path for 16 kHz data> egs/vctk/4-16 hr`
  
- `train.py` - Used to train the model.

  Usage:

  `python train.py dset=4-16 experiment=ssrm_4-16_512_64`
  
  
- `predict_multi.py` - Used to predict high-resolution speech.

  Usage:
  
  `python predict_multi.py dset=4-16 experiment=ssrm_4-16_512_64 +filename=<folder path for input speech> +output=<folder path for output speech>`



# How to use
1. Install all requirements
2. ​Configure paths
3. Run create_meta_files.py
4. Run train.py
