"""
Thank you Alfredo!
"""
import numpy as np

from text.symbols import symbols
from utils import parse_filelist

import pyworld as pw
import pysptk
import soundfile as sf
from fastdtw import fastdtw
from scipy import spatial
import pandas as pd
import csv

#import nemo.collections.asr.models.EncDecCTCModelBPE as asr
import jiwer
#from speechbrain.pretrained import EncoderASR
from transformers import pipeline



import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
import logging

log = logging.getLogger()

def fastdtw_distance_on_mels(ref_mel, pred_mel):
    ref_mel = np.asarray(ref_mel)
    pred_mel = np.asarray(pred_mel)
    _, path = fastdtw(ref_mel, pred_mel, dist=spatial.distance.euclidean)
    twf = np.array(path).T
    h_dtw = ref_mel[twf[0]]
    r_dtw = pred_mel[twf[1]]
    dtw_distance = np.sum((h_dtw-r_dtw)**2,1)
    mel_dtw = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * dtw_distance), 0)
    return mel_dtw

###@_same_t_in_true_and_est
def gross_pitch_error(true_f, est_f):
    """The relative frequency in percent of pitch estimates that are outside a threshold around the true pitch. Only frames that are considered pitched by both the ground truth and the estimator (if
    applicable) are considered.    """

    correct_frames = _true_voiced_frames(true_f, est_f)
    gross_pitch_error_frames = _gross_pitch_error_frames(true_f, est_f)
    return np.sum(gross_pitch_error_frames) / np.sum(correct_frames)

def _gross_pitch_error_frames(true_f, est_f, eps=1e-8):
    voiced_frames = _true_voiced_frames(true_f, est_f)
    true_f_p_eps = [x + eps for x in true_f]
    pitch_error_frames = np.abs(est_f / true_f_p_eps - 1) > 0.2
    return voiced_frames & pitch_error_frames

def _true_voiced_frames(true_f, est_f):
    return (est_f != 0) & (true_f != 0)

def _voicing_decision_error_frames(true_f, est_f):
    return (est_f != 0) != (true_f != 0)

##@_same_t_in_true_and_est
def f0_frame_error(true_f, est_f):
    gross_pitch_error_frames = _gross_pitch_error_frames(true_f, est_f)
    voicing_decision_error_frames = _voicing_decision_error_frames(true_f, est_f)
    return (np.sum(gross_pitch_error_frames) + np.sum(voicing_decision_error_frames)) / (len(true_f))

###@_same_t_in_true_and_est
def voicing_decision_error(true_f, est_f):
    voicing_decision_error_frames = _voicing_decision_error_frames(true_f, est_f)
    return np.sum(voicing_decision_error_frames) / (len(true_f))

def sptk_extract(x: np.ndarray, fs: int, n_fft: int = 512, n_shift: int = 256,  mcep_dim: int = 25,  mcep_alpha: float = 0.41, is_padding: bool = False,) -> np.ndarray:
    """Extract SPTK-based mel-cepstrum.
    Returns:
        ndarray: Mel-cepstrum with the size (N, n_fft).

    """
    # perform padding
    if is_padding:
        n_pad = n_fft - (len(x) - n_fft) % n_shift
        x = np.pad(x, (0, n_pad), "reflect")
    # get number of frames
    n_frame = (len(x) - n_fft) // n_shift + 1
    # get window function
    win = pysptk.sptk.hamming(n_fft)
    # check mcep and alpha
    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)
    # calculate spectrogram
    mcep = [ pysptk.mcep(x[n_shift * i : n_shift * i + n_fft] * win,  mcep_dim,  mcep_alpha,   eps=1e-6,etype=1,  )  for i in range(n_frame) ]

    return np.stack(mcep)


def world_extract(x: np.ndarray, fs: int, f0min: int = 40, f0max: int = 800, n_fft: int = 512, n_shift: int = 256, mcep_dim: int = 25, mcep_alpha: float = 0.41,) -> np.ndarray:
    """Extract World-based acoustic features.
    Returns:
        ndarray: Mel-cepstrum with the size (N, n_fft).
        ndarray: F0 sequence (N,).

    """
    # extract features
    x = x.astype(np.float64)
    f0, time_axis = pw.harvest(x,fs,f0_floor=f0min, f0_ceil=f0max, frame_period=n_shift / fs * 1000, )
    sp = pw.cheaptrick(x, f0, time_axis, fs, fft_size=n_fft)
    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)
    mcep = pysptk.sp2mc(sp, mcep_dim, mcep_alpha)

    return mcep, f0


def obtainMetrics(predFile, refFile):
    pred_x, pred_fs = sf.read(predFile,dtype="float32")  ##int16")
    ref_x, ref_fs = sf.read(refFile,dtype="float32")    ##int16")
    fs = pred_fs
    pred_mcep, pred_f0 = world_extract(x=pred_x, fs=fs, f0min=70, f0max=400, n_fft=512, n_shift=256, mcep_dim=34, mcep_alpha=0.45) ##, mcep_dim=34, mcep_alpha=0.45)
    ref_mcep, ref_f0 = world_extract(x=ref_x, fs=fs, f0min=70, f0max=400, n_fft=512, n_shift=256, mcep_dim=34, mcep_alpha=0.45) ##, mcep_dim=34, mcep_alpha=0.45)

    # DTW
    _, path = fastdtw(pred_mcep, ref_mcep, dist=spatial.distance.euclidean)
    twf = np.array(path).T
    pred_f0_dtw = pred_f0[twf[0]]
    ref_f0_dtw = ref_f0[twf[1]]

    ##Get voiced part
    nonzero_idxs = np.where((pred_f0_dtw!=0)&(ref_f0_dtw!=0))[0]
    pred_f0_dtw_voiced = np.log(pred_f0_dtw[nonzero_idxs])
    ref_f0_dtw_voiced = np.log(ref_f0_dtw[nonzero_idxs])

    ##log F0 RMSE
    log_f0_rmse = np.sqrt(np.mean((pred_f0_dtw_voiced-ref_f0_dtw_voiced)**2))
    gen_mcep = sptk_extract(x=pred_x, fs=fs, n_fft=512,n_shift=256,mcep_dim=34,mcep_alpha=0.45)
    gt_mcep = sptk_extract(x=ref_x, fs=fs, n_fft=512,n_shift=256, mcep_dim=34,mcep_alpha=0.45)

    # DTW
    _, path = fastdtw(gen_mcep, gt_mcep, dist=spatial.distance.euclidean)
    twf = np.array(path).T
    gen_mcep_dtw = gen_mcep[twf[0]]
    gt_mcep_dtw = gt_mcep[twf[1]]

    ##MCD
    diff2sum =  np.sum((gen_mcep_dtw - gt_mcep_dtw) ** 2, 1)
    mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)

    ###Compute DTW distance between mel-spectrograms
    gpe = gross_pitch_error(ref_f0_dtw, pred_f0_dtw)
    vde = voicing_decision_error(ref_f0_dtw, pred_f0_dtw)
    ffe = f0_frame_error(ref_f0_dtw, pred_f0_dtw)

    log.info(f'log_f0_rmse is  {log_f0_rmse} MCD is  {mcd} gpe is {gpe} vde is {vde} ffe is {ffe}')

    return  log_f0_rmse, mcd, gpe, vde, ffe

def calc_logf0(pred_x, ref_x, fs):
    pred_mcep, pred_f0 = world_extract(x=pred_x, fs=fs, f0min=70, f0max=400, n_fft=512, n_shift=256, mcep_dim=34, mcep_alpha=0.45) ##, mcep_dim=34, mcep_alpha=0.45)
    ref_mcep, ref_f0 = world_extract(x=ref_x, fs=fs, f0min=70, f0max=400, n_fft=512, n_shift=256, mcep_dim=34, mcep_alpha=0.45) ##, mcep_dim=34, mcep_alpha=0.45)
    
      
    _, path = fastdtw(pred_mcep, ref_mcep, dist=spatial.distance.euclidean)
    twf = np.array(path).T
    pred_f0_dtw = pred_f0[twf[0]]
    ref_f0_dtw = ref_f0[twf[1]]
    
    #print('pred_f0_dtw: ',pred_f0_dtw)
    #print('ref_f0_dtw: ',ref_f0_dtw)
    
    nonzero_idxs = np.where((pred_f0_dtw!=0)&(ref_f0_dtw!=0))[0]
    #print('nonzero_idxs: ', nonzero_idxs)
    pred_f0_dtw_voiced = np.log(pred_f0_dtw[nonzero_idxs])
    ref_f0_dtw_voiced = np.log(ref_f0_dtw[nonzero_idxs])
    
    #print('pred_f0_dtw_voiced: ', pred_f0_dtw_voiced)

    log_f0_rmse = np.sqrt(np.mean((pred_f0_dtw_voiced-ref_f0_dtw_voiced)**2))

    return log_f0_rmse

def calc_mcd(pred_x, ref_x, fs):
    gen_mcep = sptk_extract(x=pred_x, fs=fs, n_fft=512,n_shift=256,mcep_dim=34,mcep_alpha=0.45)
    gt_mcep = sptk_extract(x=ref_x, fs=fs, n_fft=512,n_shift=256, mcep_dim=34,mcep_alpha=0.45)

    _, path = fastdtw(gen_mcep, gt_mcep, dist=spatial.distance.euclidean)
    twf = np.array(path).T
    gen_mcep_dtw = gen_mcep[twf[0]]
    gt_mcep_dtw = gt_mcep[twf[1]]

    diff2sum =  np.sum((gen_mcep_dtw - gt_mcep_dtw) ** 2, 1)
    mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)

    return mcd

def get_transcriptions(pred_file, ref_file, ref, model):
    #pred_transcription = model.transcribe_file(pred_file)
    #ref_transcription = model.transcribe_file(ref_file)
    pred_transcription = model(pred_file)
    ref_transcription = model(ref_file)
    return pred_transcription['text'], ref_transcription['text']

def wer(ref_transcriptions, pred_texts, transformation):
    measures = jiwer.compute_measures(
    ref_transcriptions,
    pred_texts,
    truth_transform=transformation,
    hypothesis_transform=transformation
    )
    wer = measures['wer']
    mer = measures['mer']
    wil = measures['wil']
    return wer


@hydra.main(version_base=None, config_path='./config')
def main(cfg):

    split = cfg.eval.split

    if split == 'train':
        ref_filelist_path = cfg.data.train_filelist_path
    elif split == 'dev':
        ref_filelist_path= cfg.data.dev_filelist_path
    elif split == 'test':
        ref_filelist_path = cfg.data.test_filelist_path
    
    print('reference filelist path: ', ref_filelist_path)
    pred_filelist_path= cfg.eval.pred_filelist_path
    ref_f = parse_filelist(ref_filelist_path, split_char='|')
    ref_files = [ref[0] for ref in ref_f]
    
    
    ref_transcriptions = [ref[1] for ref in ref_f]

 
    with open(pred_filelist_path, 'r') as file:
        # Read the lines of the file into a list of strings
        pred_files = [line.strip() for line in file.readlines()]   #strip \n from lines
        
    n_evaluations = cfg.eval.n_evaluations
    #n_evaluations = len(pred_files)
    files = zip(pred_files[:n_evaluations], ref_files[:n_evaluations])

    results = np.zeros((n_evaluations, 2))
    
    #asr_model = asr.from_pretrained("nvidia/stt_en_conformer_ctc_large")
    #asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-14-en", savedir="/fastdata/speech-diff-dys/pretrained_models/asr-wav2vec2-commonvoice-14-en")
    #asr_model = pipeline("automatic-speech-recognition", model="openai/whisper-medium")

    pred_texts = []
    ref_texts = []
    counter = 0
    output_lines = []
    for i, (pred, ref) in enumerate(files):
        #print(pred)
        pred_x, pred_fs = sf.read(pred,dtype="float32") 
        ref_x, ref_fs = sf.read(ref,dtype="float32")    
        fs = pred_fs
        log_f0 = calc_logf0(pred_x, ref_x, fs)
        #print('log_f0', log_f0)
        if np.isnan(log_f0):
            log_f0 = 0
            counter +=1
         
        mcd = calc_mcd(pred_x, ref_x, fs)

        results[i] = [log_f0, mcd]
        #print('RESULTS I: ', results[i])
        #p_text, r_text = get_transcriptions(pred, ref, ref_transcriptions[i], asr_model)
        #pred_texts.append(p_text)
        #ref_texts.append(r_text)
        #output_line = f"PRED: {p_text},  R_AUDIO: {r_text},  LABEL: {ref_transcriptions[i]}"
        #output_lines.append(output_line)
        #print('PRED: ', p_text, ', R_AUDIO: ', r_text, ', LABEL: ', ref_transcriptions[i])
    log.info(np.mean(results))
    np.save('tts_metrics.npy', results)
    print('Number of log_f0 = 0: ', counter)
    
    '''
    with open ('transcriptions.txt', 'w') as file:
        for line in output_lines:
            file.write(line + '\n')
    '''
    transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.RemovePunctuation(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
    ])
    '''
    wer_pred = wer(ref_transcriptions[:n_evaluations], pred_texts, transformation)
    wer_ref = wer(ref_transcriptions[:n_evaluations], ref_texts, transformation)
    wer_change = wer_pred - wer_ref
    '''
    #wer_change = wer(ref_transcriptions[:n_evaluations], pred_texts) - wer(ref_transcriptions[:n_evaluations], ref_texts)
    
    
    #wer_change = 0
    logf0 = results[:, 0]
    mcd = results[:, 1]

    results = {
        'max_logf0': max(logf0),
        'min_logf0': min(logf0),
        'mean_logf0': np.mean(logf0),
        'max_mcd': max(mcd),
        'min_mcd': min(mcd),
        'mean_mcd': np.mean(mcd),
        #'wer_pred': wer_pred,
        #'wer_ref': wer_ref,
        #'wer_change': wer_change,
    }
    
    print(results)
    
    with open('results.yaml', 'w') as f:
        yaml.dump(results, f)
    
    with open('results.txt', 'w') as file:
        file.write(str(results))
    '''    
    data_dict = {'Pred': pred_texts, 'R_audio': ref_texts, 'Label': ref_transcriptions[:n_evaluations]}
    df = pd.DataFrame(data_dict)
    df.to_csv('transcriptions.csv')
    '''
    
if __name__ == '__main__':
    main()