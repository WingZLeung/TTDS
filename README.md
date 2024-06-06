# TTDS
Text-to-dysarthric speech (TTDS) synthesis. An implementation using the Grad-TTS model with the TORGO database.

# Dataset contains:
- download and pre-process TORGO. Including filter: corrupt audio, audio with no transcript
- create csv manifest
- create filelists and config files to train Grad-TTS 

# SpeechDiff 
Note: forked from https://github.com/huawei-noah/Speech-Backbones. See link for installation.  
Changes to speech-diff:
- hydra added
- initialisation of speaker embedding for multi and single speaker data
- evaluation scripts