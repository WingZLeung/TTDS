
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from omegaconf import DictConfig, OmegaConf
import hydra

from model import GradTTS
from utils import save_plot
from text.symbols import symbols

# from nemo.collections.tts.models import HifiGanModel
from speechbrain.pretrained import HIFIGAN
from scipy.io.wavfile import write

from utils import intersperse, save_plot
from text import text_to_sequence, cmudict

@hydra.main(version_base=None, config_path='./config')
def main(cfg: DictConfig):
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    device = torch.device(f'cuda:{cfg.training.gpu}')


    print('Initializing model...')
    model = GradTTS(cfg)
    model.load_state_dict(torch.load(cfg.eval.checkpoint, map_location=lambda loc, storage: loc))
    model.to(device).eval()
    print('Number of encoder parameters = %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters = %.2fm' % (model.decoder.nparams/1e6))

    print('Initializing vocoder...')
    # vocoder = HifiGanModel.from_pretrained(model_name='nvidia/tts_hifigan')
    vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz")

    print(f'Synthesizing text...', end=' ')

    text = 'Now I am become speech, the destroyer of text'

    cmu = cmudict.CMUDict(cfg.data.cmudict_path)

    x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).to(device)[None]
    x_lengths = torch.LongTensor([x.shape[-1]]).to(device)
    spk = cfg.eval.spk
    spk = torch.tensor(spk).to(device)

    y_enc, y_dec, attn = model.forward(x, x_lengths, n_timesteps=cfg.eval.timesteps, spk=spk)

    # audio = vocoder.convert_spectrogram_to_audio(spec=y_dec)
    audio = vocoder.decode_batch(y_dec)
    audio = audio.squeeze().to('cpu').detach().numpy()

    out_path = f'{cfg.eval.out_dir}/output.wav'

    write(out_path, 22050, audio)
    save_plot(y_dec[0].cpu(), f'{cfg.eval.out_dir}/output_spec')


if __name__ == '__main__':
    main()