#!/bin/bash

# Navigate to directory
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd $SCRIPTPATH
cd ..

# Read CLIENT_CONFIG from Python file
CLIENT_CONFIG_FILE="./FedYOLO/config.py"
if [[ ! -f "$CLIENT_CONFIG_FILE" ]]; then
    echo "Error: $CLIENT_CONFIG_FILE not found"
    exit 1
fi

# List of datasets and strategies
DATASET_NAME_LIST=("baseline")
STRATEGY_LIST=("FedHeadAvg" "FedAvg" "FedMedian")

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
        bash $SCRIPTPATH/run.sh

        # newline
        echo ""
        echo ""
        
    done
done
