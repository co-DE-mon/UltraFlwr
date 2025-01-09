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

# Define client-dependent and independent scoring styles
client_dependent_styles=("client-client" "client-server" "server-client")
client_independent_styles=("server-server")

# Function to check if strategy contains head, neck, or backbone
should_skip_server() {
    local strategy=$1
    local strategy_lower=$(echo "$strategy" | tr '[:upper:]' '[:lower:]')
    if [[ $strategy_lower == *"head"* ]] || [[ $strategy_lower == *"neck"* ]] || [[ $strategy_lower == *"backbone"* ]]; then
      return 0  # true in bash
    else
        return 1  # false in bash
    fi
}

# First, run tests that depend on client numbers
for dataset in "${datasets[@]}"; do
  for strategy in "${strategies[@]}"; do
    # Check if we should skip server-based tests
    if ! should_skip_server "$strategy"; then
      # Run server-server only if strategy doesn't contain head, neck, or backbone
      for scoring_style in "${client_independent_styles[@]}"; do
        echo "Running with dataset=$dataset, strategy=$strategy, scoring_style=$scoring_style"
        python3 "$PYTHON_SCRIPT" --dataset_name "$dataset" --strategy_name "$strategy" --scoring_style "$scoring_style"
        echo ""
        echo ""
      done
    else
      echo "Skipping server-based tests for strategy=$strategy (contains head/neck/backbone)" #! Have to add reason in the README.md
      echo ""
    fi
    
    # Run client-dependent tests
    for client_num in "${client_nums[@]}"; do
      for scoring_style in "${client_dependent_styles[@]}"; do
        # Skip server-client tests if strategy contains head, neck, or backbone
        if should_skip_server "$strategy" && [[ $scoring_style == "server-client" ]]; then
          echo "Skipping server-client test for strategy=$strategy (contains head/neck/backbone)"
          echo ""
          continue
        fi
        
        echo "Running with dataset=$dataset, strategy=$strategy, client_num=$client_num, scoring_style=$scoring_style"
        python3 "$PYTHON_SCRIPT" --dataset_name "$dataset" --strategy_name "$strategy" --client_num "$client_num" --scoring_style "$scoring_style"
        echo ""
        echo ""
      done
    done
  done
done