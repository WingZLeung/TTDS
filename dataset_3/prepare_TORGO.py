import os
import sys
import argparse
import pandas as pd
from utils import check_TORGO, wav_txt_lst, TOR_labels, make_csv_only, preproces_csv, check_audio, analyze_csv

def main(TORGO_dir, output_dir):
    '''
    main execution of TORGO pre-processing and filelists creation
    '''
    # check if TORGO directory exists, if not download and extract
    check_TORGO(TORGO_dir)

    # define output directories to save filelists
    filelist_out = os.path.join(output_dir, 'filelists')
    TORGO_out = os.path.join(filelist_out, 'TORGO')
    speechd_out = os.path.join(filelist_out, 'speechdiff')
    os.makedirs(speechd_out, exist_ok=True)
    os.makedirs(TORGO_out, exist_ok=True)

    # create text file for .txt summaries
    text_file = []

    # get lists of .wav and .txt transcript files to create csv
    wav_lst, txt_lst = wav_txt_lst(TORGO_dir)

    # extract labels from .txt transcript files
    text_labels = TOR_labels(txt_lst)

    # create csv manifest and pre-process, including check audio files 
    csv, text_file = make_csv_only(TORGO_out, wav_lst, text_labels, text_file)
    filtered_csv, text_file = preproces_csv(csv, text_file)
    checked_csv, text_file = check_audio(filtered_csv, text_file)
    # save checked csv
    checked_csv.to_csv(os.path.join(TORGO_out, 'TORGO.csv'), index=False)
    
    # save summary of unique labels and removed files
    analyze_csv(checked_csv, text_file, TORGO_out)
    unique_labels = checked_csv.label.unique()
    with open(os.path.join(TORGO_out, 'TORGO_unique_labels.txt'), 'w') as file:
        for item in unique_labels:
            file.write(f"{item}\n")

    with open(os.path.join(TORGO_out, 'TORGO_removed.txt'), 'w') as file:
        for item in text_file:
            file.write(f"{item}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the TORGO.")
    parser.add_argument("TORGO_dir", type=str, help="Path to the TORGO directory, if it doesn't exist then the TORGO will be downloaded to this dir") #root directory, i.e. where all the speaker dirs are saved
    parser.add_argument("--output_dir", "-o", type=str, default="output", help="Output directory, csv manifests and .txt summaries will be saved here") # output dir, this will save the csv manifests here

    args = parser.parse_args()

    main(args.TORGO_dir, args.output_dir)