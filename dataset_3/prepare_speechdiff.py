import os
import argparse
from utils import get_subdirectories, n_evals, data_conf, eval_conf, get_sh_files_in_directory
import json
import subprocess


def main(speechdiff_dir, output_dir):
    #check if the speechdiff filelists dir exists
    filelist_dir = os.path.join(output_dir, 'filelists', 'speechdiff', 'TORGO')
    if filelist_dir:
        # load speaker id dict
        with open(os.path.join(filelist_dir, 'speaker_id_dict.json'), 'r') as file:
            spk_dict = json.load(file)
        epochs = 1000 # define number of epochs
        # define output directories
        sh_out = os.path.join(output_dir, 'shell_scripts')
        model_out = os.path.join(output_dir, 'Grad-TTS_TORGO')
        os.makedirs(sh_out, exist_ok=True)
        os.makedirs(model_out, exist_ok=True)
        # create data config files
        print('Creating training config files')
        data_conf(speechdiff_dir, filelist_dir, model_out, sh_out)
        #create synthesis and eval config files
        print('Creating synthesis and eval config files')
        eval_conf(speechdiff_dir, filelist_dir, model_out, sh_out, spk_dict, epochs)
    else:
        print(f"filelist dir {filelist_dir} does not exist")

    # check if train script exists and run
    train_script = os.path.join(sh_out, 'train.sh')
    if train_script:
        print('Training script found. Running...')
        subprocess.run(["bash", train_script])
    else:
        print(f'train script {train_script} does not exist')
    
    eval_scripts = get_sh_files_in_directory(os.path.join(sh_out, f'inf_epochs'))
    if eval_scripts:
        print('Executing eval scripts...')
        for script in eval_scripts:
            script_path = os.path.join(sh_out, 'inf_epochs', script)
            print(f'Running synthesis and eval for {script}')
            subprocess.run(["bash", script_path])
    else:
        print('No eval .sh files found')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the TORGO.")
    parser.add_argument("speechdiff_dir", type=str, help="Path to the speechdiff directory, this is located in TTDS/speech-diff") 
    parser.add_argument("--output_dir", "-o", type=str, default="output", help="Output directory, csv manifests and .txt summaries will be saved here") # output dir, this will save the csv manifests here

    args = parser.parse_args()

    main(args.speechdiff_dir, args.output_dir)