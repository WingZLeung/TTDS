# TTDS
Text-to-dysarthric speech (TTDS) synthesis. An implementation to train the Grad-TTS model with the TORGO database (https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html) as proposed in the paper "Training Data Augmentation for Dysarthric Automatic Speech Recognition by Text-to-Dysarthric-Speech Synthesis". Generated speech files are available on request.

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
Note: forked from https://github.com/vasistalodagala/whisper-finetune. The code to create filelists for and configs for LOSO training will be added to the repo in a future update. Changes to whisper-finetune:
- spec augment training .py script added

## Citing this code

Please cite the following paper if you use this code in your work:

Wing-Zin Leung, Mattias Cross, Anton Ragni, Stefan Goetze. 'Training Data Augmentation for Dysarthric Automatic Speech Recognition by Text-to-Dysarthric-Speech Synthesis'. ArXiv LINK

```
BIBTEX REFERENCE 
```


