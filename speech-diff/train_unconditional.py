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

from model import SpeechSynth
from data import TextMelDataset, TextMelBatchCollate
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

    directory_name = "my_directory"

    # Check if the directory exists
    if not os.path.exists(cfg.training.checkpoint_dir):
        # Create the directory
        os.makedirs(cfg.training.checkpoint_dir)


    log.info('Initializing data loaders...')
    train_dataset = TextMelDataset('train', cfg)
    batch_collate = TextMelBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=cfg.training.batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=cfg.training.num_workers, shuffle=True)
    test_dataset = TextMelDataset('dev', cfg)

    log.info('Initializing model...')
    model = SpeechSynth(cfg)
    model.to(device)
    log.info('Number of decoder parameters = %.2fm' % (model.decoder.nparams/1e6))

    log.info('Initializing optimizer...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.training.learning_rate)

    log.info('Logging test batch...')
    test_batch = test_dataset.sample_test_batch(size=cfg.training.test_size)
    for item in test_batch:
        mel = item['y']
        logger.add_image(f'image/ground_truth', plot_tensor(mel.squeeze()),
                         global_step=0, dataformats='HWC')
        save_plot(mel.squeeze(), f'{log_dir}/original.png')

    out_size = fix_len_compatibility(2*cfg.data.sample_rate//256)

    log.info('Start training...')
    iteration = 0
    for epoch in range(1, cfg.training.n_epochs + 1):
        model.eval()
        log.info('Synthesis...')
        with torch.no_grad():
            for item in test_batch:

                x = item['y'][None].to(device)
                lengths = torch.tensor(x.size(-1))[None]

                y_dec= model(x, lengths, n_timesteps=50)

                logger.add_image(f'image/generated_dec',
                                 plot_tensor(y_dec.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                save_plot(y_dec.squeeze().cpu(), 
                          f'{log_dir}/generated_dec.png')
        
        model.train()
        prior_losses = []
        diff_losses = []
        with tqdm(loader, total=len(train_dataset)//cfg.training.batch_size) as progress_bar:
            for batch in progress_bar:
                model.zero_grad()
                y, y_lengths = batch['y'].to(device), batch['y_lengths'].to(device)
                prior_loss, diff_loss = model.compute_loss(y, y_lengths)
                use_prior_loss = True
                if use_prior_loss == True:
                    loss = sum([prior_loss, diff_loss])
                else:
                    loss = diff_loss
                loss.backward()

                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 
                                                            max_norm=1)
                optimizer.step()

                logger.add_scalar('training/prior_loss', prior_loss,
                                global_step=iteration)
                logger.add_scalar('training/diffusion_loss', diff_loss,
                                global_step=iteration)
                logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                global_step=iteration)
                
                msg = f'Epoch: {epoch}, iteration: {iteration} prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}'
                progress_bar.set_description(msg)
                
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())
                iteration += 1

        msg += '| prior loss = %.3f ' % np.mean(prior_losses)
        msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)

        log.info(msg)
        
        if epoch % cfg.training.save_every > 0:
            continue
        
        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{cfg.training.checkpoint_dir}/{epoch}.pt")

if __name__ == '__main__':
    main()
