# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import random
import numpy as np

import torch
import torchaudio as ta

from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import parse_filelist, intersperse
from model.utils import fix_len_compatibility

import sys
sys.path.insert(0, 'hifi-gan')
from librosa.filters import mel as librosa_mel_fn
import os
# from meldataset import mel_spectrogram

from lhotse import CutSet
from typing import Dict
import re
from time import time

# create spectrogram
mel_basis = {}
hann_window = {}

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    global mel_basis
    if str(fmax) not in mel_basis:
        mel_basis[str(fmax)] = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)

    # Padding
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode='reflect')
    y = y.squeeze(1)

    # Short-time Fourier transform
    hann_window = torch.hann_window(win_size)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)

    # Magnitude spectrogram
    spec_mag = torch.abs(spec)

    # Mel spectrogram
    mel = torch.matmul(torch.tensor(mel_basis[str(fmax)]).to(y.device), spec_mag)

    # Spectral normalization
    mel = spectral_normalize_torch(mel)

    return mel


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(self, split, cfg):
        super().__init__()

        random.seed(cfg.training.seed)
        cfg = cfg.data
        
        if split == 'train':
            filelist_path = cfg.train_filelist_path
        elif split == 'dev':
            filelist_path = cfg.dev_filelist_path
        elif split == 'test':
            filelist_path = cfg.test_filelist_path

        self.filelist = parse_filelist(filelist_path, split_char='|')
        self.cmudict = cmudict.CMUDict(cfg.cmudict_path)
        self.n_fft = cfg.n_fft
        self.n_mels = cfg.n_feats
        self.sample_rate = cfg.sample_rate
        self.hop_length = cfg.hop_length
        self.win_length = cfg.win_length
        self.f_min = cfg.f_min
        self.f_max = cfg.f_max
        self.add_blank = cfg.add_blank

    def get_pair(self, line):
        filepath, text = line[0], line[1]
        text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)
        return (text, mel)

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False).squeeze()
        return mel

    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        text, mel= self.get_pair(self.filelist[index])
        item = {'y': mel, 'x': text}
        return item

    def __len__(self):
        return len(self.filelist)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []

        for i, item in enumerate(batch):
            y_, x_ = item['y'], item['x']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths}


class TextMelSpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, split, cfg):
        super().__init__()

        random.seed(cfg.training.seed)
        cfg = cfg.data
        
        if split == 'train':
            filelist_path = cfg.train_filelist_path
        elif split == 'dev':
            filelist_path = cfg.dev_filelist_path
        elif split == 'test':
            filelist_path = cfg.test_filelist_path

        self.filelist = parse_filelist(filelist_path, split_char='|')
        self.cmudict = cmudict.CMUDict(cfg.cmudict_path)
        self.n_fft = cfg.n_fft
        self.n_mels = cfg.n_feats
        self.sample_rate = cfg.sample_rate
        self.hop_length = cfg.hop_length
        self.win_length = cfg.win_length
        self.f_min = cfg.f_min
        self.f_max = cfg.f_max
        self.add_blank = cfg.add_blank

    def get_triplet(self, line):
        filepath, text, speaker = line[0], line[1], line[2]
        text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)
        speaker = self.get_speaker(speaker)
        return (text, mel, speaker)

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False).squeeze()
        return mel

    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_speaker(self, speaker):
        speaker = torch.LongTensor([int(speaker)])
        return speaker

    def __getitem__(self, index):
        text, mel, speaker = self.get_triplet(self.filelist[index])
        item = {'y': mel, 'x': text, 'spk': speaker}
        return item

    def __len__(self):
        return len(self.filelist)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelSpeakerBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []
        spk = []

        for i, item in enumerate(batch):
            y_, x_, spk_ = item['y'], item['x'], item['spk']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
            spk.append(spk_)

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        spk = torch.cat(spk, dim=0)
        return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths, 'spk': spk}