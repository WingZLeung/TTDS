import pandas as pd
import os
import argparse
import random
import yaml
import json


def main(speechdiff_dir, sh_out, model_out, config_name, filelist_dir):
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

    def data_conf():
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
        data_path = os.path.join(speechdiff_dir, 'config', 'data', f'{config_name}.yaml')
        with open(data_path, 'w') as file:
            yaml.dump(template, file)
        
#         #create shell file for train script
#         sh_template = f'''{sh_header}
        sh_template = f'''
cd "{model_root}"
python "{speechdiff_dir}/train_multi_speaker.py" --config-name="config_template"  hydra.run.dir="{model_root}" +data="{config_name}"
        '''
        with open(os.path.join(sh_out, f'{config_name}.sh'), 'w') as file:
            file.write(sh_template.format(TEXT=config_name))

    def eval_conf():
        '''
        create eval config and sh files
        '''
        os.makedirs(os.path.join(sh_out, f"inf_{epochs}"), exist_ok=True)
        spk_dirs = get_subdirectories(filelist_dir)
        eval_path = os.path.join(speechdiff_dir, 'config', 'eval', config_name)
        os.makedirs(eval_path, exist_ok=True)
        #create eval.yaml config files
        with open("./eval_template.yaml", 'r') as file:
            eval_template = yaml.safe_load(file)
        eval_template['split'] = 'test'
        eval_template['checkpoint'] = os.path.join(model_root, 'checkpoints', f"grad_{epochs}.pt")
        for speaker in spk_dirs:
            spk_root = os.path.join(filelist_dir, speaker)
            train_evals, val_evals, test_evals = n_evals(spk_root)  
            eval_template['n_evaluations'] = test_evals     
            eval_template['spk'] = spk_dict[speaker]
            with open(os.path.join(eval_path, f"{speaker}.yaml"), 'w') as file:
                yaml.dump(eval_template, file)
            run_dir = os.path.join(model_out, 'HFGN_inference', config_name, speaker)
#             inf_template = f'''{sh_header}
            inf_template =f'''
python "{speechdiff_dir}/generate_tts_preds.py" --config-name="config_template" hydra.run.dir="{run_dir}/B1" +data="{config_name}" ++data.train_filelist_path="{os.path.join(spk_root, 'B1.txt')}" +eval="{config_name}/{speaker}" ++eval.split="train" ++eval.out_dir="{run_dir}/B1" ++eval.n_evaluations={train_evals}
python "{speechdiff_dir}/generate_tts_preds.py" --config-name="config_template" hydra.run.dir="{run_dir}/B3" +data="{config_name}" ++data.dev_filelist_path="{os.path.join(spk_root, 'B3.txt')}" +eval="{config_name}/{speaker}" ++eval.split="dev" ++eval.out_dir="{run_dir}/B3" ++eval.n_evaluations={val_evals}
python "{speechdiff_dir}/generate_tts_preds.py" --config-name="config_template" hydra.run.dir="{run_dir}/B2" +data="{config_name}" ++data.test_filelist_path="{os.path.join(spk_root, 'B2.txt')}" +eval="{config_name}/{speaker}" ++eval.split="test" ++eval.out_dir="{run_dir}/B2" ++eval.n_evaluations={test_evals}
python "{speechdiff_dir}/evaluate_tts.py" --config-name="config_template" hydra.run.dir="{run_dir}/test/evaluation" +data="{config_name}"  ++data.test_filelist_path="{os.path.join(spk_root, 'B2.txt')}" +eval="{config_name}/{speaker}" +eval.pred_filelist_path="{run_dir}/B2/test_preds.txt"
            '''
            with open(os.path.join(sh_out, f"inf_{epochs}", f'{speaker}.sh'), 'w') as file:
                file.write(inf_template.format(TEXT=speaker))


    

    # with open(sh_template, 'r') as file:
    #     sh_header = file.read()
    with open(os.path.join(filelist_dir, 'speaker_id_dict.json'), 'r') as file:
        spk_dict = json.load(file)
    epochs = 1000
    os.makedirs(sh_out, exist_ok=True)
    model_root = os.path.join(model_out, config_name)
    os.makedirs( model_root, exist_ok=True)
    data_conf()
    eval_conf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create config and .sh files for speechdiff.")
    parser.add_argument("speechdiff_dir", type=str, help="Path to the speechdiff directory") 
    parser.add_argument("--sh_out", "-o", type=str, default="output", help="Output directory, i.e. where the .sh files will be saved")
    parser.add_argument("--model_out", "-m", type=str, default='model_out', help='output directory for the speechdiff model ckpoints')
    parser.add_argument("--config_name", "-c", type=str, default='config_name', help='output directory for the config files')
    # parser.add_argument("--sh_template", "-t", type=str, default="sh_template", help="path to your sh template")
    parser.add_argument("--filelist_dir", "-f", type=str, default="filelist", help="path to your filelists dir")
    parser.add_argument("--speaker_dict", "-s", type = str, default='speaker_dict', help="path to the speaker: id dict")
    args = parser.parse_args()

    main(args.speechdiff_dir, args.sh_out, args.model_out, args.config_name, args.filelist_dir)