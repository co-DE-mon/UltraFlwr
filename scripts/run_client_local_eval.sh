#!/usr/bin/env bash

# navigate to directory
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd $SCRIPTPATH

cd ..

# Install FedYOLO from setup.py, uncomment if already installed
if [[ -f "setup.py" ]]; then
    echo "Installing FedYOLO package..."
    pip install --no-cache-dir -e .
else
    echo "Error: setup.py not found. Cannot install FedYOLO."
    exit 1
fi

BASE_PATH="$(pwd)"

echo "Base directory: $BASE_PATH"

DATASET_NAME="m2cai16"
MODEL_PATHS=("runs/detect/train48/weights/best.pt"
             "runs/detect/train49/weights/best.pt"
             "runs/detect/train50/weights/best.pt"
             "runs/detect/train51/weights/best.pt")
DATASET_PATHS=("${BASE_PATH}/datasets/${DATASET_NAME}/partitions/client_0/data.yaml"
          "${BASE_PATH}/datasets/${DATASET_NAME}/partitions/client_1/data.yaml"
          "${BASE_PATH}/datasets/${DATASET_NAME}/partitions/client_2/data.yaml"
          "${BASE_PATH}/datasets/${DATASET_NAME}/data.yaml")
LOG_DIR="logs_local_train_${DATASET_NAME}"

mkdir -p "$LOG_DIR"

for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH="${MODEL_PATHS[$i]}"
    DATASET_PATH="${DATASET_PATHS[$i]}"

    LOG_FILE="$LOG_DIR/clientwise_test_$(echo "$DATASET_PATH" | sed 's|/|_|g').log"

    echo "Starting training on $DATASET_PATH..."
    python3 scripts/local_eval_for_server_table.py --data "$DATASET_PATH" --model "$MODEL_PATH" | tee "$LOG_FILE"
    echo "Finished training on $DATASET_PATH."
    echo "---------------------------------------"
done

echo "All trainings completed."
