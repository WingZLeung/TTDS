import os
import sys
import argparse
import glob
import pandas as pd
import json
import librosa


def main(root_dir, output_dir):
    def wav_txt_lst(root):
        '''
        traverse dirs and make a list of all .wav files, and a list of all .txt files
        '''
        wav_lst = glob.glob(os.path.join(root, "**/*.wav"), recursive=True)
        text_lst = glob.glob(os.path.join(root, "**/*.txt"), recursive=True)
        print("tot wav audio files {}".format(len(wav_lst)))
        print("tot txt prompt files {}".format(len(text_lst)))
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
                line = line.translate(str.maketrans('', '', punc))
                line = line.upper()
                text_labels[tag2] = line
        return text_labels
    def make_csv(output_dir, wav_lst, text_labels, text_file):
        '''
        create a csv manifest
        '''
        with open('./subs.json', 'r') as json_file:  #read the json file with original transcript: substituted transcript. Edit json as required
            subs_dict = json.load(json_file)
        csv_data = [['wav', 'speaker', 'corpus', 'block', 'label', 'mic', 'length']] #headers
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
                csv_data.append([audio, speaker, corpus, 'block', label, mic, 'length'])
            else:
                text_file.append(f"{audio}|None|No label")
        df = pd.DataFrame(csv_data[1:], columns=csv_data[0])  # Skip the header row
        return df, text_file
    
    def preproces_csv(df, text_file):
        '''
        pre-process transcripts
        '''
        print('Pre-processing transcripts')
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
        print('Checking audio for corrupt or exmpty files')
        corrupt_files = []
        for index, row in csv.iterrows():
            audio = row['wav']
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
                if duration < 0.1:    # Edit value to specify length of audio to filter out. Currently = 0.1 seconds
                    text_file.append(f"{audio}|{row['label']}|short duration")
                    corrupt_files.append(index)
                    continue
            except Exception as e:
                corrupt_files.append(index)
                text_file.append(f"{audio}|{row['label']}|{str(e)}")
        checked_csv = csv.drop(corrupt_files)
        return checked_csv, text_file
    def analyze_csv(df, text_files):
        '''
        Create summary files including number of wavs, unique labels, and audio/transcripts pairs omitted from preprocessed dataset
        '''
        print('creating .txt summaries')
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

    text_file = []

    TORGO = ['F01', 'F03', 'F04', 'M01', 'M02', 'M03', 'M04', 'M05']  #specify dysarthric speakers here
    TORGO_control = ['FC01', 'FC02', 'FC03', 'MC01', 'MC02', 'MC03', 'MC04'] # control speakers

    #speakers = TORGO + TORGO_control  #create list of all speakers to process
    
    wav_lst, txt_lst = wav_txt_lst(root_dir)
    text_labels = TOR_labels(txt_lst)
    
    csv, text_file = make_csv(output_dir, wav_lst, text_labels, text_file)
    filtered_csv, text_file = preproces_csv(csv, text_file)

    checked_csv, text_file = check_audio(filtered_csv, text_file)
    checked_csv.to_csv(os.path.join(output_dir, 'TORGO.csv'), index=False)

    analyze_csv(checked_csv, text_file)
    unique_labels = checked_csv.label.unique()

    with open(os.path.join(output_dir, 'TORGO_unique_labels.txt'), 'w') as file:
        for item in unique_labels:
            file.write(f"{item}\n")

    with open(os.path.join(output_dir, 'TORGO_removed.txt'), 'w') as file:
        for item in text_file:
            file.write(f"{item}\n")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the TORGO.")
    parser.add_argument("root_dir", type=str, help="Path to the TORGO directory, i.e. where the speaker dirs are") #root directory, i.e. where all the speaker dirs are saved
    parser.add_argument("--output_dir", "-o", type=str, default="output", help="Output directory, csv manifests and .txt summaries will be saved here") # output dir, this will save the csv manifests here

    args = parser.parse_args()

    main(args.root_dir, args.output_dir)