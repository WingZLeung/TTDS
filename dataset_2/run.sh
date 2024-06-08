#Read config file, if you have cd'd to TTDS/dataset it should read the config in the same dir
CONFIG_FILE="config.json"

if [ -f "$CONFIG_FILE" ]; then
    echo "$CONFIG_FILE found, reading configuration..."
else
    echo "Configuration file $CONFIG_FILE not found!"
    # exit 2
fi


# Read paths from the config file, update these on the config file as required
TORGO_DIR=$(jq -r '.torgo_dir' $CONFIG_FILE)
FILELIST_DIR=$(jq -r '.filelist_dir' $CONFIG_FILE)
SD_DIR=$(jq -r '.speechdiff_dir' $CONFIG_FILE)
SH_DIR=$(jq -r '.sh_dir' $CONFIG_FILE)
MODEL_DIR=$(jq -r '.model_dir' $CONFIG_FILE)


# run prepare_TORGO python script to check if TORGO exists (and download if not) and pre-process and create CSV file
python prepare_TORGO.py $TORGO_DIR --output_dir $FILELIST_DIR/TORGO

# check if the TORGO CSV has been created, if so prepare random TORGO splits

TORGO_CSV="$FILELIST_DIR/TORGO/TORGO.csv"
if [ -f "$TORGO_CSV" ]; then    
    echo "TORGO CSV found, preparing random data splits..."
    python prepare_TORGO_splits.py "$TORGO_CSV" --output_dir "$OUTPUT_DIR"
else
    echo "TORGO CSV not found at $TORGO_CSV!"
fi

#check if the TORGO random split csv exists, if so prepare speechdiff filelists
if [ -f "$FILELIST_DIR/TORGO/TORGO_split.csv" ]; then
    echo "Creating speech-diff filelists"
    python prepare_filelists.py $SD_DIR --csv_path $FILELIST_DIR/TORGO/TORGO_split.csv --output_dir $FILELIST_DIR/speechdiff
else
    echo "TORGO CSV split file does not exist"
    # exit 2
fi

# create speechdiff config file and shell scripts for training and inference
python prepare_configs.py $SD_DIR --sh_out $SH_DIR --model_out $MODEL_DIR --config_name 'TORGO_all' --filelist_dir $FILELIST_DIR/speechdiff --speaker_dict $FILELIST_DIR/speechdiff/speaker_id_dict.json

if [ -d "$SH_DIR"]; then
    echo "running training script"
    bash 
else
    echo "shell scripts dir $SH_DIR does not exist"
    exit 1