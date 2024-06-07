DATASET_DIR="../"
TORGO_DIR="$DATASET_DIR/TORGO"
OUTPUT_DIR="../manifests/TORGO"
SD_DIR="../speech-diff"

# Check if the TORGO directory exists
if [ -d "$TORGO_DIR" ]; then
    echo "Directory 'TORGO' already exists in $DATASET_DIR. Skipping download and extraction."
else
    echo "Directory 'TORGO' does not exist. Creating it now..."
    mkdir "$TORGO_DIR"

    cd "$TORGO_DIR"
    echo "Downloading files..."
    wget -q --show-progress https://www.cs.toronto.edu/~complingweb/data/TORGO/F.tar.bz2
    wget -q --show-progress https://www.cs.toronto.edu/~complingweb/data/TORGO/FC.tar.bz2
    wget -q --show-progress https://www.cs.toronto.edu/~complingweb/data/TORGO/M.tar.bz2
    wget -q --show-progress https://www.cs.toronto.edu/~complingweb/data/TORGO/MC.tar.bz2

    echo "Extracting files..."
    tar -xf "F.tar.bz2"
    tar -xf "FC.tar.bz2"
    tar -xf "M.tar.bz2"
    tar -xf "MC.tar.bz2"

    echo "Cleaning up..."
    rm "F.tar.bz2"
    rm "FC.tar.bz2"
    rm "M.tar.bz2"
    rm "MC.tar.bz2"

    echo "Download and extraction completed."
fi

if [ -d "$TORGO_DIR" ]; then
    echo "preprocessing TORGO"
    python /users/acr22wl/TTDS/dataset/prepare_TORGO.py $TORGO_DIR --output_dir $OUTPUT_DIR
    echo "preparing random data splits"
    python /users/acr22wl/TTDS/dataset/prepare_TORGO_splits.py $OUTPUT_DIR/TORGO.csv --output_dir $OUTPUT_DIR
else
    echo "TORGO directory not found" 
fi

if [ -f "$OUTPUT_DIR/TORGO_split.csv" ]; then
    echo "Creating speech-diff filelists"
    python prepare_filelists.py $SD_DIR --csv_path $OUTPUT_DIR/TORGO_splits.csv --output_dir './manifests/speechdiff'
else
    echo "TORGO CSV split file does not exist"
fi