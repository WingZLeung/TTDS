# Speechdiff, a framework for diffusion applied to speech

The goal of this repository is to make it easy to experiment with audiodatasets and diffusion. This is powered by [hydra](https://hydra.cc/docs/intro/https://hydra.cc/), scorebased generative models (Song et al.) and Grad-TTS (Popov et al.).

At the moment it's currently a revamp of [Grad-TTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS)

## Installation

Python 3.9.17

```
pip install cython
```

```
pip install -r requirements.txt
```

```
cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..
```

## Train a Grad-TTS model

Create a filelist in the form `f'{audio_path}|{transcription}|{speaker_id}'` where `speaker_id` is an integer. Edit `config/data/data.yaml` to suit your dataset.

Run
```
python train_multi_speaker.py --config-name=config +data=data
```

Edit config as desired, or make use of hydra's [multirun](https://hydra.cc/docs/1.0/tutorials/basic/running_your_app/multi-run/#internaldocs-banner) utility.

## Evaluate a TTS model

First generate predictions for your dataset
```
python generate_tts_preds.py --config-name=config +data=delete_this +eval=eval
```
Then calculate `log-f0 rmse`
```
python evaluate_tts.py --config-name=config +data=data +eval=eval
```

## Compute log-likelihoods

Coming soon

## Citation

```
@Misc{Cross2023SpeechDiff,
  author =       {Mattias Cross},
  title =        {Speech diff, a framework for diffusion applied to speech},
  howpublished = {Github},
  year =         {2023},
  url =          {https://github.com/Mattias421/speech-diff}
}
```
