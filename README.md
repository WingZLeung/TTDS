# TTDS
Text-to-dysarthric speech (TTDS) synthesis. An implementation to train the Grad-TTS model with the TORGO database.

Dataset contains:
- download and pre-process TORGO. Including filtering: corrupt or empty audio files, audio with no transcript. Also correction of transcripts (e.g. from provided instruction to words read aloud)
- create csv manifest
- create random data split
- create filelists and config files to train Grad-TTS 

Speech-diff 
Note: forked from https://github.com/huawei-noah/Speech-Backbones. See link for installation.  
Changes to speech-diff:
- hydra added
- initialisation of speaker embedding for multi and single speaker data
- configs
- evaluation scripts