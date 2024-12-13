#!/bin/bash

# Get the root directory of the repo
LOCAL_HOME="$(git rev-parse --show-toplevel)"

# Navigate to the repo root
cd "$LOCAL_HOME"

# Read CLIENT_CONFIG from Python file
CLIENT_CONFIG_FILE="./FedYOLO/config.py"
if [[ ! -f "$CLIENT_CONFIG_FILE" ]]; then
    echo "Error: $CLIENT_CONFIG_FILE not found"
    exit 1
fi

# List of datasets and strategies
DATASET_NAME_LIST=("baseline" "bccd")
STRATEGY_LIST=("FedAvg" "FedMedian" "FedHeadAvg" "FedHeadMedian")

# Loop over each dataset and strategy
for DATASET_NAME in "${DATASET_NAME_LIST[@]}"; do
    for STRATEGY in "${STRATEGY_LIST[@]}"; do
        
        echo "===================================================================="
        echo "Running with DATASET_NAME=${DATASET_NAME} and STRATEGY=${STRATEGY}"
        echo "===================================================================="
        
        # Modify the config.py file
        sed -i "s/^DATASET_NAME = .*/DATASET_NAME = '${DATASET_NAME}'/" $CLIENT_CONFIG_FILE
        sed -i "s/^\s*'strategy': .*/    'strategy': '${STRATEGY}',/" $CLIENT_CONFIG_FILE
        
        # Run the base bash file
        bash "$LOCAL_HOME/scripts/run.sh"

        # newline
        echo ""
        echo ""
        
    done
done
