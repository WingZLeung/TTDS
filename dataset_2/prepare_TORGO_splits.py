import pandas as pd
import os
import argparse
import random


def main(csv_path, output_dir, random_seed=None):
    os.makedirs(output_dir, exist_ok=True)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create random TORGO splits.")
    parser.add_argument("csv_path", type=str, help="Path to the TORGO csv") 
    parser.add_argument("--output_dir", "-o", type=str, default="output", help="Output directory, where the random split csv will be saved")

    args = parser.parse_args()

    main(args.csv_path, args.output_dir, random_seed=42)