
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math

import torch

from model.base import BaseModule
from model.unconditional_diffusion import Diffusion
from model.utils import fix_len_compatibility, sequence_mask


class SpeechSynth(BaseModule):
    def __init__(self, cfg):
        super(SpeechSynth, self).__init__()

        self.n_feats = cfg.data.n_feats

        self.decoder = Diffusion(cfg)

    @torch.no_grad()
    def forward(self, x, lengths, n_timesteps, temperature=1.0, stoc=False, length_scale=1.0):
        """
        Generates mel-spectrogram from text. Returns:
            1. decoder outputs
        
        Args:
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """

        z = torch.randn_like(x, device=x.device) / temperature
        # Generate sample by performing reverse dynamics
        max_length = int(lengths.max())
        max_length_ = fix_len_compatibility(max_length)

        # Using obtained durations `w` construct alignment map `attn`
        mask = sequence_mask(lengths, max_length_).unsqueeze(1).to(x.device)
        
        decoder_outputs = self.decoder(z, mask, n_timesteps, stoc)

        return decoder_outputs 

    def compute_loss(self, x, x_lengths):
        """
        Computes 3 losses:
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.
            
        Args:
            x (torch.Tensor): batch of mel-spectrograms.
            x_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """

        max_length = x.shape[-1]
        x_mask = sequence_mask(x_lengths, max_length).unsqueeze(1).to(x)


        # Compute loss of score-based decoder
        diff_loss, xt = self.decoder.compute_loss(x, x_mask)
        
        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((x) ** 2 + math.log(2 * math.pi)) * x_mask)
        prior_loss = prior_loss / (torch.sum(x_mask) * self.n_feats)
        
        return prior_loss, diff_loss

