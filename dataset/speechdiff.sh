DATASET_DIR="/mnt/parscratch/users/acr22wl/TTDS_test"
TORGO_DIR="$DATASET_DIR/TORGO"
OUTPUT_DIR="/mnt/parscratch/users/acr22wl/TTDS_test/manifests/speechdiff/"
SD_DIR="./speech-diff"
CSV_DIR="/mnt/parscratch/users/acr22wl/TTDS_test/manifests/TORGO"

python prepare_filelists.py SD_DIR $CSV_DIR/TORGO_split.csv 