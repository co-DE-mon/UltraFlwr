#!/bin/bash

LOCAL_HOME="$(git rev-parse --show-toplevel)"

PYTHON_SCRIPT="$LOCAL_HOME/FedYOLO/test/test.py"
CONFIG_FILE="$LOCAL_HOME/FedYOLO/config.py"

# Extract configurations from the Python config file using Python
DATASET_NAME=$(python3 -c "import sys; sys.path.append('$LOCAL_HOME/FedYOLO'); import config; print(config.DATASET_NAME)")
CONFIG_strategy=$(python3 -c "import sys; sys.path.append('$LOCAL_HOME/FedYOLO'); import config; print(config.SERVER_CONFIG['strategy'])")
CLIENT_CONFIG_cid=$(python3 -c "import sys; sys.path.append('$LOCAL_HOME/FedYOLO'); import config; print(','.join(str(client['cid']) for client in config.CLIENT_CONFIG.values()))")

# Convert the extracted variables to arrays in bash
IFS=',' read -r -a datasets <<< "$DATASET_NAME"
IFS=',' read -r -a strategies <<< "$CONFIG_strategy"
IFS=',' read -r -a client_nums <<< "$CLIENT_CONFIG_cid"

# Define scoring styles
scoring_styles=("client-client" "client-server" "server-client" "server-server")
# scoring_styles=("server-client" "server-server")

# Loop through each combination of dataset, strategy, client number, and scoring style
for dataset in "${datasets[@]}"; do
  for strategy in "${strategies[@]}"; do
    for client_num in "${client_nums[@]}"; do
      for scoring_style in "${scoring_styles[@]}"; do
        echo "Running with dataset=$dataset, strategy=$strategy, client_num=$client_num, scoring_style=$scoring_style"
        python3 "$PYTHON_SCRIPT" --dataset_name "$dataset" --strategy_name "$strategy" --client_num "$client_num" --scoring_style "$scoring_style"
        echo ""
        echo ""
      done
    done
  done
done

#! server-server is being computed multiple times. Will Fix it later.