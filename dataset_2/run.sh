CONFIG_FILE="config.json"


TORGO_DIR=$(jq -r '.torgo_dir' $CONFIG_FILE)
FILELIST_DIR=$(jq -r '.filelist_dir' $CONFIG_FILE)
SD_DIR=$(jq -r '.speechdiff_dir' $CONFIG_FILE)

python prepare_TORGO.py $TORGO_DIR --output_dir $FILELIST_DIR/TORGO