import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import json
import librosa
import subprocess
import tarfile
import random
import yaml
from tqdm import tqdm

def check_TORGO(root_dir):
        '''
        check if the TORGO directory exists, if False download and extract files from the TORGO database website.
        '''
        # Define the URLs for the files to download. These are the links from the TORGO database website.
        urls = [
            "https://www.cs.toronto.edu/~complingweb/data/TORGO/F.tar.bz2",
            "https://www.cs.toronto.edu/~complingweb/data/TORGO/FC.tar.bz2",
            "https://www.cs.toronto.edu/~complingweb/data/TORGO/M.tar.bz2",
            "https://www.cs.toronto.edu/~complingweb/data/TORGO/MC.tar.bz2"
        ]
        # Get the current working directory to return to later
        if os.path.isdir(root_dir):
            print(f"Directory 'TORGO' already exists in {root_dir}. Skipping download and extraction.")
        else:
            print(f"Directory 'TORGO' does not exist. Creating it now...")
            os.makedirs(root_dir)

            # Download and extract files
            for url in urls:
                filename = os.path.basename(url)
                file_path = os.path.join(root_dir, filename)

                print(f"Downloading {url}...")
                subprocess.run(["wget", "-q", "--show-progress", "-O", file_path, url], check=True)

                print(f"Extracting {filename}...")
                if tarfile.is_tarfile(file_path):
                    with tarfile.open(file_path, 'r:bz2') as tar:
                        tar.extractall(path=root_dir)
                else:
                    print(f"Error: {filename} is not a valid tar.bz2 file.")

                # Clean up the downloaded tar.bz2 file
                print(f"Removing {filename}...")
                os.remove(file_path)

            print("Download and extraction completed.")


def wav_txt_lst(root):
        '''
        traverse dirs and make a list of all .wav files, and a list of all .txt files
        '''
        wav_lst = glob.glob(os.path.join(root, "**/*.wav"), recursive=True)
        text_lst = glob.glob(os.path.join(root, "**/*.txt"), recursive=True)
        print("total .wav audio files {}".format(len(wav_lst)))
        print("total .txt transcript files {}".format(len(text_lst)))
        return wav_lst, text_lst


def TOR_labels(text_files):
    '''
    create a dictionary of tag: label for processing
    '''
    text_labels = {}
    punc = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'    #punctuation to filter, edit as required
    for txtf in sorted(text_files):
        ttag = txtf.split('.')
        temp = " ".join(ttag[0:-1])
        ttemp = temp.split('/')
        tag = "".join(ttemp[0:-2])  # rootdirSpkSessionN
        tag2 = tag + ttemp[-1] # rootdirSpkSessionNWAV_ID
        with open(txtf, "r") as f:
            lines = f.readlines() 
        for i, line in enumerate(lines):  #pre-process transcripts
            line = line.strip("\n")
            line = line.translate(str.maketrans('', '', punc)) # filter out punctuation 
            line = line.upper() # change to upper case
            text_labels[tag2] = line
    return text_labels


def make_csv(output_dir, wav_lst, text_labels, text_file):
    '''
    create a csv manifest
    '''
    TORGO = ['F01', 'F03', 'F04', 'M01', 'M02', 'M03', 'M04', 'M05']  #specify dysarthric speakers here
    TORGO_control = ['FC01', 'FC02', 'FC03', 'MC01', 'MC02', 'MC03', 'MC04'] # control speakers
    with open('subs.json', 'r') as json_file:  #read the json file with original transcript: substituted transcript. Edit json as required
        subs_dict = json.load(json_file)
    csv_data = [['wav', 'speaker', 'corpus', 'block', 'label', 'ID', 'mic', 'length']] #headers
    for audio in sorted(wav_lst):
        ttag = audio.split('.')
        temp = " ".join(ttag[0:-1])
        ttemp = temp.split('/')
        tag = "".join(ttemp[0:-2]) #rootdirSpkSessionN
        tag2 = tag + ttemp[-1]
        spk = ttemp[-4]
        speaker = f"TORGO_{spk}"
        mic = ttemp[-2].split('_')[-1]
        if spk in TORGO:
            corpus = 'TORGO'
        else:
            corpus = 'TORGO_control'
        if tag2 in text_labels:
            if text_labels[tag2] in subs_dict:
                lab = text_labels[tag2] 
                label = subs_dict[lab]
            else:
                label = text_labels[tag2]
            csv_data.append([audio, speaker, corpus, 'block', label, tag2, mic, 'length'])
        else:
            text_file.append(f"{audio}|None|No label")
    df = pd.DataFrame(csv_data[1:], columns=csv_data[0])  # Skip the header row
    return df, text_file

def make_csv_only(output_dir, wav_lst, text_labels, text_file):
    '''
    create a csv manifest
    '''
    TORGO = ['F01', 'F03', 'F04', 'M01', 'M02', 'M03', 'M04', 'M05']  #specify dysarthric speakers here
    TORGO_control = ['FC01', 'FC02', 'FC03', 'MC01', 'MC02', 'MC03', 'MC04'] # control speakers
    with open('subs.json', 'r') as json_file:  #read the json file with original transcript: substituted transcript. Edit json as required
        subs_dict = json.load(json_file)
    csv_data = [['wav', 'speaker', 'corpus', 'label', 'ID', 'mic', 'length']] #headers
    for audio in sorted(wav_lst):
        ttag = audio.split('.')
        temp = " ".join(ttag[0:-1])
        ttemp = temp.split('/')
        tag = "".join(ttemp[0:-2]) #rootdirSpkSessionN
        tag2 = tag + ttemp[-1]
        spk = ttemp[-4]
        speaker = f"TORGO_{spk}"
        mic = ttemp[-2].split('_')[-1]
        if spk in TORGO:
            corpus = 'TORGO'
        else:
            corpus = 'TORGO_control'
        if tag2 in text_labels:
            if text_labels[tag2] in subs_dict:
                lab = text_labels[tag2] 
                label = subs_dict[lab]
            else:
                label = text_labels[tag2]
            csv_data.append([audio, speaker, corpus, label, tag2, mic, 'length'])
        else:
            text_file.append(f"{audio}|None|No label")
    df = pd.DataFrame(csv_data[1:], columns=csv_data[0])  # Skip the header row
    return df, text_file

def preproces_csv(df, text_file):
    '''
    pre-process transcripts
    '''
    print('Pre-processing transcripts...')
    # list of filter words, edit as required
    filter_words = ['JPG', 'AHPEEE', 'PAHTAHKAH', 'EEEPAH', 'FOR 5 SECONDS', 
                    'HE WILL ALLOW A RARE', "SAY 'OA' AS IN COAT", 'EEE', 'SAY OA', 
                    'RELAX YOUR MOUTH', 'XXX']
    filtered_df = df[~df['label'].str.contains('|'.join(filter_words), case=False, na=False)]
    for index, row in df.iterrows():
        if any(word in row['label'] for word in filter_words):
            text_file.append(f"{row['wav']}|{row['label']}|filter word")
    return filtered_df, text_file


def check_audio(csv, text_file):
    '''
    check audio for corrupt or empty files
    '''
    print('Checking audio for corrupt or empty files...')
    corrupt_files = []
    total_files = len(csv)
    print('Total files to check: ', total_files)
    progress_bar = tqdm(total=total_files)
    for index, row in csv.iterrows():
        audio = row['wav']
        progress_bar.set_description(f"Checking {audio}")
        if not os.path.exists(audio):
            text_file.append(f"{audio}|{row['label']}|file not found")
            corrupt_files.append(index)
            continue
        if os.path.getsize(audio) == 0:
            text_file.append(f"{audio}|{row['label']}|empty file")
            corrupt_files.append(index)
            continue
        try:
            y, sr = librosa.load(audio, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            row['length'] = duration
            if duration < 0.4:    # Edit value to specify length of audio to filter out. Currently = 0.4 seconds
                text_file.append(f"{audio}|{row['label']}|short duration")
                corrupt_files.append(index)
                continue
        except Exception as e:
            corrupt_files.append(index)
            text_file.append(f"{audio}|{row['label']}|{str(e)}")
        progress_bar.update(1)
    progress_bar.close()
    checked_csv = csv.drop(corrupt_files)
    return checked_csv, text_file


def analyze_csv(df, text_file, output_dir):
    '''
    Create summary files including number of wavs, unique labels, and audio/transcripts pairs omitted from preprocessed dataset
    '''
    print('creating .txt summaries...')
    report_file = []
    report_file.append(f"Total number of wav files: {len(df)}")
    report_file.append(f"Unique labels: {len(df.label.unique())}\n")
    for corp in df.corpus.unique():
        sub_data = df[df['corpus']==corp]
        report_file.append(f"Number of wav files in {corp}: {len(sub_data)}")
    report_file.append('\n')
    for speaker in df.speaker.unique():
        sub_data = df[df['speaker']==speaker]
        report_file.append(f"Number of wav files for {speaker}: {len(sub_data)}, unique labels: {len(sub_data.label.unique())}")
    total_files = 0
    reason_counts = {}
    for item in text_file:
        total_files+=1
        parts = item.strip().split('|')
        reason = parts[-1]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    report_file.append(f"\nTotal number of wav files removed: {total_files}")
    for reason, count in reason_counts.items():
        report_file.append(f"{reason}: {count}")
    with open(os.path.join(output_dir, 'TORGO_report.txt'), 'w') as file:
        for item in report_file:
            file.write(f"{item}\n")    


def random_split(csv_path, output_dir, random_seed=None):
    '''
    Create blocks for the TORGO from paired audio
    '''
    os.makedirs(output_dir, exist_ok=True)
    
    # Code for randomised split
    '''
    #set seed
    if random_seed is not None:
        random.seed(random_seed)
    # shuffle per speaker and randomly allocate data splits
    df = pd.read_csv(csv_path)
    train_ratio = 0.8
    val_ratio = test_ratio = 0.1 # 80:10:10 ratio, adjust as required. Will need to define val/test end if val and test ratios are different
    for speaker, speaker_data in df.groupby('speaker'):
        speaker_data = speaker_data.sample(frac=1)  # Shuffle the data for each speaker
        n = len(speaker_data)
        train_end = int(train_ratio * n)
        val_end = train_end + int(val_ratio * n)
        print(speaker, n, train_end, val_end)
        # Assign block numbers to splits
        df.loc[speaker_data.iloc[:train_end].index, 'block'] = 'B1' # training block
        df.loc[speaker_data.iloc[train_end:val_end].index, 'block'] = 'B2' # test block
        df.loc[speaker_data.iloc[val_end:].index, 'block'] = 'B3' # eval block
    df.to_csv(os.path.join(output_dir, 'TORGO_split.csv'))   
    '''
    # code for paired audio split 
    # set seed
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed) 
    #read processed dataframe
    data = pd.read_csv(csv_path)
    speakers = data.speaker.unique().tolist()
    # for each speaker find paired audio (i.e same recordings from wab and array mics)
    for speaker in speakers:
        sub_data = data[data['speaker']==speaker]
        u_tags = sub_data.ID.unique().tolist()
        np.random.shuffle(u_tags) # shuffle order of paired audio to randomise blocks
        num_combinations = len(u_tags)
        print(f"{speaker}. Paired audio: {num_combinations}")
        combinations_per_block = num_combinations // 3
        Block1 = u_tags[:combinations_per_block]
        Block2 = u_tags[combinations_per_block: 2 * combinations_per_block]
        Block3 = u_tags[2 * combinations_per_block:]
        for tag in Block1:
            data.loc[(data['speaker'] == speaker) & (data['ID'] == tag), 'block'] = 'B1'
        for tag in Block2:
            data.loc[(data['speaker'] == speaker) & (data['ID'] == tag), 'block'] = 'B2'
        for tag in Block3:
            data.loc[(data['speaker'] == speaker) & (data['ID'] == tag), 'block'] = 'B3'
        os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "TORGO_paired.csv"), index=False)


def spk_id(speaker_input):
        '''
        create a dict of speaker label: speaker_id to correspond to speaker embedding IDs in the Grad-TTS model
        '''
        speaker_id_dict = {}
        current_id = 0
        if isinstance(speaker_input, str):
            # Handle the case where the input is a single string
            speaker_list = [speaker_input]
        elif isinstance(speaker_input, list):
            # Handle the case where the input is a list of speakers
            speaker_list = speaker_input
        else:
            raise ValueError("Invalid input type. Input must be a string or a list of speakers.")
        for speaker in speaker_list:
            if speaker not in speaker_id_dict:
                # Assign a unique ID to the speaker
                speaker_id_dict[speaker] = current_id
                current_id += 1
        return speaker_id_dict


def prepare_filelists(data, speaker_id_dict, save_root, extension):
    '''
    create audio and transcription .txt files for speechdiff filelists
    '''
    aud = []
    wavs = data['wav'].tolist()
    transcripts = data['label'].tolist()
    spks = data['speaker'].tolist()
    for i in range(len(wavs)):
        aud.append(f"{wavs[i]}|{transcripts[i]}|{speaker_id_dict[spks[i]]}")
    file_path = os.path.join(save_root, f"{extension}.txt")
    with open(file_path, 'w') as file:
        file_content = '\n'.join(aud)
        file.write(file_content)
    file_path = os.path.join(save_root, f"{extension}_labels.txt")
    with open(file_path, 'w') as file:
        file_content = '\n'.join(transcripts)
        file.write(file_content)


def all_splits(data, output_dir):
    '''
    all model configs, i.e. multispeaker Grad-TTS model using all speakers. ASp model in the paper.
    '''
    corps = data.corpus.unique()
    blocks = data.block.unique()
    for corp in corps:
        save_c = os.path.join(output_dir, corp)
        os.makedirs(save_c, exist_ok=True)
        corp_data = data[data['corpus']==corp]
        speakers = corp_data.speaker.unique()
        speaker_id_dict = spk_id(list(speakers))
        with open(os.path.join(save_c, 'speaker_id_dict.json'), 'w') as file:
            json.dump(speaker_id_dict, file)
        print(speaker_id_dict)
        for block in blocks:
            sub_data = corp_data[corp_data['block']==block]
            prepare_filelists(sub_data, speaker_id_dict, save_c, block)
        for speaker in speakers:
            save_s = os.path.join(save_c, speaker)
            os.makedirs(save_s, exist_ok=True)
            for block in blocks:
                sp_data = corp_data[(corp_data['speaker'] == speaker) & (corp_data['block']==block)]
                prepare_filelists(sp_data, speaker_id_dict, save_s, block)

def get_subdirectories(root):
    """
    Get a list of subdirectories under a root directory.
    
    Args:
        root (str): The path to the root directory.
    
    Returns:
        list: A list of subdirectory names.
    """
    subdirectories = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            subdirectories.append(name)
    return subdirectories

def n_evals(spk_root):
    with open(os.path.join(os.path.join(spk_root, 'B1.txt')), 'r') as file:
        train_evals = sum(1 for line in file)
    with open(os.path.join(os.path.join(spk_root, 'B3.txt')), 'r') as file:
        val_evals = sum(1 for line in file)
    with open(os.path.join(os.path.join(spk_root, 'B2.txt')), 'r') as file:
        test_evals = sum(1 for line in file)
    return train_evals, val_evals, test_evals

def data_conf(speechdiff_dir, filelist_dir, model_out, sh_out):
    '''
    create speech-diff data config files for TORGO_all model
    '''
    # create data.yaml config file
    with open("./data_template.yaml", 'r') as file:
        template = yaml.safe_load(file)
    template['cmudict_path'] = os.path.join(speechdiff_dir, 'resources', 'cmu_dictionary')
    template['train_filelist_path'] = os.path.join(filelist_dir, 'B1.txt')
    template['dev_filelist_path'] = os.path.join(filelist_dir, 'B3.txt')
    template['test_filelist_path'] = os.path.join(filelist_dir, 'B2.txt')
    with open(os.path.join(filelist_dir, 'speaker_id_dict.json'), 'r') as file:
        speaker_id = json.load(file)
    template['n_spks'] = len(speaker_id)
    data_path = os.path.join(speechdiff_dir, 'config', 'data', f'TORGO_all.yaml')
    with open(data_path, 'w') as file:
        yaml.dump(template, file)
    sh_template = f'''
python "{speechdiff_dir}/train_multi_speaker.py" \\
        --config-name="config_template" \\
        hydra.run.dir="{model_out}" \\
        +data="TORGO_all"
'''
    with open(os.path.join(sh_out, f'train.sh'), 'w') as file:
            file.write(sh_template)

def eval_conf(speechdiff_dir, filelist_dir, model_out, sh_out, spk_dict, epochs=1000):
    '''
    create eval config and sh files
    '''
    config_name = 'TORGO_all'
    os.makedirs(os.path.join(sh_out, f"inf_{epochs}"), exist_ok=True)
    spk_dirs = get_subdirectories(filelist_dir)
    eval_path = os.path.join(speechdiff_dir, 'config', 'eval', config_name)
    os.makedirs(eval_path, exist_ok=True)
    #create eval.yaml config files
    with open("./eval_template.yaml", 'r') as file:
        eval_template = yaml.safe_load(file)
    eval_template['split'] = 'test'
    eval_template['checkpoint'] = os.path.join(model_out, 'checkpoints', f"grad_{epochs}.pt")
    for speaker in spk_dirs:
        spk_root = os.path.join(filelist_dir, speaker)
        train_evals, val_evals, test_evals = n_evals(spk_root)  
        eval_template['n_evaluations'] = test_evals     
        eval_template['spk'] = spk_dict[speaker]
        with open(os.path.join(eval_path, f"{speaker}.yaml"), 'w') as file:
            yaml.dump(eval_template, file)
        run_dir = os.path.join(model_out, 'HFGN_inference', 'TORGO_all', speaker)
        inf_template =f'''
python "{speechdiff_dir}/generate_tts_preds.py" --config-name="config_template" hydra.run.dir="{run_dir}/B1" +data="{config_name}" ++data.train_filelist_path="{os.path.join(spk_root, 'B1.txt')}" +eval="{config_name}/{speaker}" ++eval.split="train" ++eval.out_dir="{run_dir}/B1" ++eval.n_evaluations={train_evals}
python "{speechdiff_dir}/generate_tts_preds.py" --config-name="config_template" hydra.run.dir="{run_dir}/B3" +data="{config_name}" ++data.dev_filelist_path="{os.path.join(spk_root, 'B3.txt')}" +eval="{config_name}/{speaker}" ++eval.split="dev" ++eval.out_dir="{run_dir}/B3" ++eval.n_evaluations={val_evals}
python "{speechdiff_dir}/generate_tts_preds.py" --config-name="config_template" hydra.run.dir="{run_dir}/B2" +data="{config_name}" ++data.test_filelist_path="{os.path.join(spk_root, 'B2.txt')}" +eval="{config_name}/{speaker}" ++eval.split="test" ++eval.out_dir="{run_dir}/B2" ++eval.n_evaluations={test_evals}
python "{speechdiff_dir}/evaluate_tts.py" --config-name="config_template" hydra.run.dir="{run_dir}/test/evaluation" +data="{config_name}"  ++data.test_filelist_path="{os.path.join(spk_root, 'B2.txt')}" +eval="{config_name}/{speaker}" +eval.pred_filelist_path="{run_dir}/B2/test_preds.txt"
            '''
        with open(os.path.join(sh_out, f"inf_{epochs}", f'{speaker}.sh'), 'w') as file:
                file.write(inf_template)

def get_sh_files_in_directory(directory):
    sh_files = [f for f in os.listdir(directory) if f.endswith('.sh') and os.path.isfile(os.path.join(directory, f))]
    return sh_files