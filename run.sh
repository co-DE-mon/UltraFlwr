#!/bin/bash

# Default values for arguments
SERVER_SCRIPT="server/yolo_server.py"
CLIENT_SCRIPT="client/yolo_client.py"
SERVER_ADDRESS="127.0.0.1:8080"  # Changed to localhost for better connectivity

# Configurations for each client
CLIENT_1_CID=1
CLIENT_1_DATA_PATH="/home/localssk23/FedYOLO/datasets/pills/partitions/client_0/data.yaml"

CLIENT_2_CID=2
CLIENT_2_DATA_PATH="/home/localssk23/FedYOLO/datasets/pills/partitions/client_1/data.yaml"

# Function to start the server
start_server() {
    echo "Starting server..."
    python3 "$SERVER_SCRIPT" &
    SERVER_PID=$!
    echo "Server started with PID: $SERVER_PID"
}

# Function to start a client with its own config
start_client() {
    CLIENT_CID=$1
    CLIENT_DATA_PATH=$2
    echo "Starting client $CLIENT_CID with data path: $CLIENT_DATA_PATH..."
    python3 "$CLIENT_SCRIPT" --server_address="$SERVER_ADDRESS" --cid="$CLIENT_CID" --data_path="$CLIENT_DATA_PATH" &
    CLIENT_PID=$!
    echo "Client $CLIENT_CID started with PID: $CLIENT_PID"
}

# Start the server
start_server

# Add a short delay to ensure server is up
sleep 2

# Start client 1 with its own config
start_client "$CLIENT_1_CID" "$CLIENT_1_DATA_PATH"

# Start client 2 with its own config
start_client "$CLIENT_2_CID" "$CLIENT_2_DATA_PATH"

# Wait for all processes to finish
wait $SERVER_PID
wait $CLIENT_1_PID
wait $CLIENT_2_PID


# Manual way:

# Start the server
# New terminal
# python3 server/yolo_server.py

# Start client 0
# New terminal
# python3 client/yolo_client.py --server_address=0.0.0.0:8080 --cid=0 --data_path /home/localssk23/FedYOLO/datasets/pills/partitions/client_0/data.yaml

# Start client 1
# New terminal
# python3 client/yolo_client.py --server_address=0.0.0.0:8080 --cid=1 --data_path /home/localssk23/FedYOLO/datasets/pills/partitions/client_1/data.yaml