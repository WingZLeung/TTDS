# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from omegaconf import DictConfig, OmegaConf
import hydra
import os
import logging

log = logging.getLogger()

from model import GradTTS
from data import TextMelSpeakerDataset, TextMelSpeakerBatchCollate
from utils import plot_tensor, save_plot
from text.symbols import symbols
from model.utils import fix_len_compatibility

@hydra.main(version_base=None, config_path='./config')
def main(cfg: DictConfig):
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    device = torch.device(f'cuda:{cfg.training.gpu}')

    log.info('Initializing logger...')
    log_dir = cfg.training.tensorboard_dir
    logger = SummaryWriter(log_dir=log_dir)

    log.info('Initializing data loaders...')
    train_dataset = TextMelSpeakerDataset('train', cfg)
    batch_collate = TextMelSpeakerBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=cfg.training.batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=cfg.training.num_workers, shuffle=True)
    test_dataset = TextMelSpeakerDataset('dev', cfg)

    log.info('Initializing model...')
    model = GradTTS(cfg)
    model.to(device)
    log.info('Number of encoder parameters = %.2fm' % (model.encoder.nparams/1e6))
    log.info('Number of decoder parameters = %.2fm' % (model.decoder.nparams/1e6))

    log.info('Initializing optimizer...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.training.learning_rate)

    log.info('Logging test batch...')
    test_batch = test_dataset.sample_test_batch(size=cfg.training.test_size)
    for item in test_batch:
        mel, spk = item['y'], item['spk']
        i = int(spk.cpu())
        logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
                         global_step=0, dataformats='HWC')
        save_plot(mel.squeeze(), f'{log_dir}/original_{i}.png')

    out_size = fix_len_compatibility(2*cfg.data.sample_rate//256)

    log.info('Start training...')
    iteration = 0
    for epoch in range(1, cfg.training.n_epochs + 1):
        model.eval()
        log.info('Synthesis...')
        with torch.no_grad():
            for item in test_batch:
                x = item['x'].to(torch.long).unsqueeze(0).to(device)
                x_lengths = torch.LongTensor([x.shape[-1]]).to(device)
                spk = item['spk'].to(torch.long).to(device)
                i = int(spk.cpu())
                
                y_enc, y_dec, attn = model(x, x_lengths, n_timesteps=50, spk=spk)
                logger.add_image(f'image_{i}/generated_enc',
                                 plot_tensor(y_enc.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/generated_dec',
                                 plot_tensor(y_dec.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/alignment',
                                 plot_tensor(attn.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                save_plot(y_enc.squeeze().cpu(), 
                          f'{log_dir}/generated_enc_{i}.png')
                save_plot(y_dec.squeeze().cpu(), 
                          f'{log_dir}/generated_dec_{i}.png')
                save_plot(attn.squeeze().cpu(), 
                          f'{log_dir}/alignment_{i}.png')
        
        model.train()
        dur_losses = []
        prior_losses = []
        diff_losses = []
        with tqdm(loader, total=len(train_dataset)//cfg.training.batch_size) as progress_bar:
            for batch in progress_bar:
                model.zero_grad()
                x, x_lengths = batch['x'].to(device), batch['x_lengths'].to(device)
                y, y_lengths = batch['y'].to(device), batch['y_lengths'].to(device)
                spk = batch['spk'].to(device)
                dur_loss, prior_loss, diff_loss = model.compute_loss(x, x_lengths,
                                                                     y, y_lengths,
                                                                     spk=spk, out_size=out_size)
                loss = sum([dur_loss, prior_loss, diff_loss])
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 
                                                            max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 
                                                            max_norm=1)
                optimizer.step()

                logger.add_scalar('training/duration_loss', dur_loss,
                                global_step=iteration)
                logger.add_scalar('training/prior_loss', prior_loss,
                                global_step=iteration)
                logger.add_scalar('training/diffusion_loss', diff_loss,
                                global_step=iteration)
                logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                global_step=iteration)
                logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                global_step=iteration)
                
                msg = f'Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}'
                progress_bar.set_description(msg)
                
                dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())
                iteration += 1

        msg = 'Epoch %d: duration loss = %.3f ' % (epoch, np.mean(dur_losses))
        msg += '| prior loss = %.3f ' % np.mean(prior_losses)
        msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)

        log.info(msg)
        
        if epoch % cfg.training.save_every > 0:
            continue
        
        ckpt = model.state_dict()
        os.makedirs(f"{cfg.training.checkpoint_dir}/", exist_ok=True)
        torch.save(ckpt, f=f"{cfg.training.checkpoint_dir}/grad_{epoch}.pt")

if __name__ == '__main__':
    main()
