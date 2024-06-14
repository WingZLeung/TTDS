# TTDS
Text-to-dysarthric speech (TTDS) synthesis. An implementation to train the Grad-TTS model with the TORGO database (https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html) as proposed in the paper "Training Data Augmentation for Dysarthric Automatic Speech Recognition by Text-to-Dysarthric-Speech Synthesis". The generated speech files are available on request.

## Installation

Python 3.9.18

You will need to install the dependencies for Grad-TTS (https://github.com/huawei-noah/Speech-Backbones):

```
pip install cython
```

```
cd TTDS/speech-diff
pip install -r requirements.txt
```

```
cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..
```

You will also need to install Hydra, which was used for configuring model optimisation during experiments:
```
pip install hydra-core --upgrade
```


## Dataset:
- download and pre-process TORGO. Including filtering corrupt or empty audio files, inaccurate transcripts, and audio with no transcript. Also, correction of transcripts (e.g. from provided instruction to actual words read aloud). Criteria for pre-processing can be changed, but requires changes to the prepare_TORGO.py file (see comments in the file). 
- create csv manifest
- create random data split, and filelists and config files to train Grad-TTS 
- train Grad-TTS and synthesise dysarthric data
- evaluate test samples

Use:
```
cd TTDS/dataset
TTDS.py TORGO_DIR --speechdiff_dir SPEECHDIFF_DIR --output_dir OUTPUT_DIR
```
TORGO_DIR is where TORGO is saved, and if it doesn't exist TORGO will be downloaded and extracted here. \
SPEECHDIFF_DIR should be the path to TTDS/speech-diff, by default it is '../speech-diff' which should run correctly if your working directory is TTDS/dataset. \
OUTPUT_DIR is where all output .csv and .txt files, and the Grad-TTS model and synthesised samples will be saved.

If you would like to only download and pre-process the TORGO for other uses:
```
prepare_TORGO.py TORGO_DIR --output_dir OUTPUT_DIR
```


## Speech-diff. 
Note: forked from https://github.com/huawei-noah/Speech-Backbones. See link for full details on the repo and installation. You will also need to install Hydra, which was used for configuring model optimisation during experiments. 

Changes to speech-diff:
- hydra added
- initialisation of speaker embedding for multi and single speaker data
- configs
- evaluation script. Modified from https://github.com/espnet/espnet


## Whisper-finetune. 
Note: forked from https://github.com/vasistalodagala/whisper-finetune. The code to create filelists and configs for LOSO training will be added to the repo in a future update. Changes to whisper-finetune:
- spec augment training .py script added

## Citing this code

Please cite the following paper if you use this code in your work:

Leung, W.-Z., Cross, M., Ragni, A. and Goetze, S., 2024. Training data augmentation for dysarthric automatic speech recognition by text-to-dysarthric-speech synthesis. *arXiv*. Available at: https://arxiv.org/abs/2406.08568.

```
@misc{leung2024training,
      title={Training Data Augmentation for Dysarthric Automatic Speech Recognition by Text-to-Dysarthric-Speech Synthesis}, 
      author={Wing-Zin Leung and Mattias Cross and Anton Ragni and Stefan Goetze},
      year={2024},
      eprint={2406.08568},
      archivePrefix={arXiv},
      primaryClass={id='cs.SD' full_name='Sound' is_active=True alt_name=None in_archive='cs' is_general=False description='Covers all aspects of computing with sound, and sound as an information channel. Includes models of sound, analysis and synthesis, audio user interfaces, sonification of data, computer music, and sound signal processing. Includes ACM Subject Class H.5.5, and intersects with H.1.2, H.5.1, H.5.2, I.2.7, I.5.4, I.6.3, J.5, K.4.2.'}
}
```