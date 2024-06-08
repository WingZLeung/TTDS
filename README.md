# TTDS
Text-to-dysarthric speech (TTDS) synthesis. An implementation to train the Grad-TTS model with the TORGO database.

## Installation

Python 3.9.18

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

## Dataset:
- download and pre-process TORGO. Including filtering corrupt or empty audio files, inaccurate transcripts, and audio with no transcript. Also correction of transcripts (e.g. from provided instruction to actual words read aloud). Criteria for pre-processing filtering can be changed, but requires changes to the prepare_TORGO.py file (see comments in file). 
- create csv manifest
- create random data split
- create filelists and config files to train Grad-TTS 
- train Grad-TTS and synthesise dysarthric data
- evaluate test samples

Use:
```
cd TTDS/dataset
```

```
run.sh
```

## Speech-diff. 
Note: forked from https://github.com/huawei-noah/Speech-Backbones. See link for full details on installation. You will also need to install Hydra, which was used for configuring model optimisation during experiments. 

Changes to speech-diff:
- hydra added
- initialisation of speaker embedding for multi and single speaker data
- configs
- evaluation script. Modified from https://github.com/espnet/espnet


## Whisper-finetune. 
Note: forked from https://github.com/vasistalodagala/whisper-finetune. Code for filelists and training to be updated to repo. Changes to whisper-finetune:
- spec augment training .py script added

## Reference
```
Wing-Zin Leung, Mattias Cross, Anton Ragni, Stefan Goetze. 'Training Data Augmentation for Dysarthric Automatic Speech Recognition by Text-to-Dysarthric-Speech Synthesis'. ArXiv LINK
```


