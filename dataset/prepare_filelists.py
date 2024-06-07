import argparse
import pandas as pd
import os
import json

def main(speechdiff_dir, csv_path, output_dir):
    

    def spk_id(speaker_input):
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

    # #THIS IS A VERSION USING BOTH DYSARTHIC AND CONTROL DATA, I.E. COMBINED SPEAKER ID DICT
    # def all_splits(data, output_dir):
    #     save_r = os.path.join(output_dir)
    #     os.makedirs(save_r,  exist_ok=True)
    #     blocks = data.block.unique()
    #     for block in blocks:
    #         sub_data = data[data['block']==block]
    #         prepare_filelists(sub_data, speaker_id_dict, save_r, block)
    #     speakers = data.speaker.unique()
    #     for speaker in speakers:
    #         save_s = os.path.join(save_r, speaker)
    #         os.makedirs(save_s, exist_ok=True)
    #         for block in blocks:
    #             sp_data = data[(data['speaker'] == speaker) & (data['block']==block)]
    #             prepare_filelists(sp_data, speaker_id_dict, save_s, block)
    #     corps = data.corpus.unique()
    #     for corp in corps:
    #         save_c = os.path.join(save_r, corp)
    #         os.makedirs(save_c, exist_ok=True)
    #         for block in blocks:
    #             corp_data = data[(data['corpus'] == corp) & (data['block']==block)]
    #             prepare_filelists(corp_data, speaker_id_dict, save_c, block)

    def all_splits(data, output_dir):
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



    data = pd.read_csv(csv_path)
    speakers = data.speaker.unique()

    # UNCOMMENT TO USE COMBINED VERSION
    # speaker_id_dict = spk_id(list(speakers))
    # with open(os.path.join(output_dir, 'speaker_id_dict.json'), 'w') as file:
    #     json.dump(speaker_id_dict, file)
    # print(speaker_id_dict)
    
    
    all_splits(data, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create filelists for speechdiff.")
    parser.add_argument("speechdiff_dir", type=str, help="Path to the speechdiff directory") 
    parser.add_argument("--csv_path", type=str, help="Path to the dataset csv file") 
    parser.add_argument("--output_dir", "-o", type=str, default="output", help="Output directory, csv manifests will be saved here") 

    args = parser.parse_args()

    main(args.speechdiff_dir, args.csv_path, args.output_dir)