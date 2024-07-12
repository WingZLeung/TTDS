import os
import sys
import argparse
import pandas as pd
import json
import subprocess

import utils

def main(TORGO_dir, speechdiff_dir, output_dir):
    '''
    main execution of TORGO pre-processing and filelists creation
    '''
    # check if TORGO directory exists, if not download and extract
    utils.check_TORGO(TORGO_dir)

    # define output directories to save filelists
    filelist_out = os.path.join(output_dir, 'filelists')
    TORGO_out = os.path.join(filelist_out, 'TORGO')
    speechd_out = os.path.join(filelist_out, 'speechdiff')
    os.makedirs(speechd_out, exist_ok=True)
    os.makedirs(TORGO_out, exist_ok=True)

    # create text file for .txt summaries
    text_file = []

    # get lists of .wav and .txt transcript files to create csv
    wav_lst, txt_lst = utils.wav_txt_lst(TORGO_dir)

    # extract labels from .txt transcript files
    text_labels = utils.TOR_labels(txt_lst)

    # create csv manifest and pre-process, including check audio files 
    csv, text_file = utils.make_csv(TORGO_out, wav_lst, text_labels, text_file)
    filtered_csv, text_file = utils.preproces_csv(csv, text_file)
    checked_csv, text_file = utils.check_audio(filtered_csv, text_file)
    # save checked csv
    checked_csv.to_csv(os.path.join(TORGO_out, 'TORGO.csv'), index=False)
    
    # save summary of unique labels and removed files
    utils.analyze_csv(checked_csv, text_file, TORGO_out)
    unique_labels = checked_csv.label.unique()
    with open(os.path.join(TORGO_out, 'TORGO_unique_labels.txt'), 'w') as file:
        for item in unique_labels:
            file.write(f"{item}\n")

    with open(os.path.join(TORGO_out, 'TORGO_removed.txt'), 'w') as file:
        for item in text_file:
            file.write(f"{item}\n")

    # check if TORGO csv exists, if so create random splits and save to TORGO_split.csv
    TORGO_csv = os.path.join(TORGO_out, 'TORGO.csv')
    if TORGO_csv:
        print('Creating random splits')
        utils.random_split(TORGO_csv, os.path.join(TORGO_out), 42)
    else:
        print(f'TORGO csv {TORGO_csv} does not exist')
    
    # check if TORGO splits csv exists, if so create speechdiff filelists
    TORGO_split = os.path.join(TORGO_out, 'TORGO_split.csv')
    if TORGO_split:
        print('Creating speechdiff filelists')
        data = pd.read_csv(TORGO_split) #read csv file
        utils.all_splits(data, speechd_out)
    else:
        print(f'TORGO random splits csv {TORGO_split} does not exist')
    
    # # define path to speech-diff submodule
    # speechdiff_dir = "../speech-diff"

    #check if the speechdiff filelists dir exists
    filelist_dir = os.path.join(output_dir, 'filelists', 'speechdiff', 'TORGO')
    if filelist_dir:
        # load speaker id dict
        with open(os.path.join(filelist_dir, 'speaker_id_dict.json'), 'r') as file:
            spk_dict = json.load(file)
        epochs = 1000 # define which epoch checkpoint to peform inferenc with
        # define output directories
        sh_out = os.path.join(output_dir, 'shell_scripts')
        model_out = os.path.join(output_dir, 'Grad-TTS_TORGO')
        os.makedirs(sh_out, exist_ok=True)
        os.makedirs(model_out, exist_ok=True)
        # create data config files
        print('Creating training config files')
        utils.data_conf(speechdiff_dir, filelist_dir, model_out, sh_out)
        #create synthesis and eval config files
        print('Creating synthesis and eval config files')
        utils.eval_conf(speechdiff_dir, filelist_dir, model_out, sh_out, spk_dict, epochs)
    else:
        print(f"filelist dir {filelist_dir} does not exist")

    # check if train script exists and run
    train_script = os.path.join(sh_out, 'train.sh')
    if train_script:
        print('Training script found. Running...')
        subprocess.run(["bash", train_script])
    else:
        print(f'train script {train_script} does not exist')
    
    eval_scripts = utils.get_sh_files_in_directory(os.path.join(sh_out, f'inf_{epochs}'))
    if eval_scripts:
        print('Executing eval scripts...')
        for script in eval_scripts:
            script_path = os.path.join(sh_out, f'inf_{epochs}', script)
            print(f'Running synthesis and eval for {script}')
            subprocess.run(["bash", script_path])
    else:
        print('No eval .sh files found')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the TORGO.")
    parser.add_argument("TORGO_dir", type=str, help="Path to the TORGO directory, if it doesn't exist then the TORGO will be downloaded to this dir") #root directory, i.e. where all the speaker dirs are saved
    parser.add_argument("--speechdiff_dir", type=str, default="../speech-diff", help="Path to the speech-diff directory") #the default is ../speech-diff which should run if your wd is TTDS/dataset
    parser.add_argument("--output_dir", "-o", type=str, default="output", help="Output directory, csv manifests and .txt summaries will be saved here") # output dir, this will save the csv manifests here

    args = parser.parse_args()

    main(args.TORGO_dir, args.speechdiff_dir, args.output_dir)