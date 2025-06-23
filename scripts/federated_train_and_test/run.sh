#!/usr/bin/env bash

# navigate to directory
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$SCRIPTPATH"

cd ../../

# Default values for arguments
SERVER_SCRIPT="FedYOLO/train/yolo_server.py"
CLIENT_SCRIPT="FedYOLO/train/yolo_client.py"
SERVER_ADDRESS="127.0.0.1:8080"  # Changed to localhost for better connectivity

# Read CLIENT_CONFIG from Python file
CLIENT_CONFIG_FILE="./FedYOLO/config.py"
if [[ ! -f "$CLIENT_CONFIG_FILE" ]]; then
    echo "Error: $CLIENT_CONFIG_FILE not found"
    exit 1
fi

DATASET_NAME=$(./ultravenv/Scripts/python.exe -c "from FedYOLO.config import SPLITS_CONFIG; print(SPLITS_CONFIG['dataset_name'])")
STRATEGY_NAME=$(./ultravenv/Scripts/python.exe -c "from FedYOLO.config import SERVER_CONFIG; print(SERVER_CONFIG['strategy'])")

# Function to start the server
start_server() {
    # Free port 8080 before starting the server
    echo "Freeing port 8080..."
    # lsof -t -i:8080 | xargs kill -9 2>/dev/null
    echo "Starting server..."
    SERVER_LOG="logs/server_log_${DATASET_NAME}_${STRATEGY_NAME}.txt"
    ./ultravenv/Scripts/python.exe "$SERVER_SCRIPT" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    PIDS+=($SERVER_PID)
    echo "Server started with PID: $SERVER_PID. Logs: $SERVER_LOG"
}

# Function to start a client with its own config
start_client() {
    CLIENT_CID=$1
    CLIENT_DATA_PATH=$(./ultravenv/Scripts/python.exe -c "from FedYOLO.config import CLIENT_CONFIG; print(CLIENT_CONFIG[$CLIENT_CID]['data_path'])")
    CLIENT_LOG="logs/client_${CLIENT_CID}_log_${DATASET_NAME}_${STRATEGY_NAME}.txt"
    echo "Starting client $CLIENT_CID with data path: $CLIENT_DATA_PATH..."
    ./ultravenv/Scripts/python.exe "$CLIENT_SCRIPT" --cid="$CLIENT_CID" --data_path="$CLIENT_DATA_PATH" > "$CLIENT_LOG" 2>&1 &
    CLIENT_PID=$!
    PIDS+=($CLIENT_PID)
    echo "Client $CLIENT_CID started with PID: $CLIENT_PID. Logs: $CLIENT_LOG"
}

# Start the server
start_server

# Add a short delay to ensure server is up
sleep 2

# Start clients based on CLIENT_CONFIG
for CLIENT_CID in $(./ultravenv/Scripts/python.exe -c "from FedYOLO.config import CLIENT_CONFIG; print(' '.join(map(str, CLIENT_CONFIG.keys())))"); do
    start_client "$CLIENT_CID"
done

# Wait for all processes to finish
wait
