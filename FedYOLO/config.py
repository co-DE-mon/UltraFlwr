# config.py
import yaml
import platform
import os

def get_nc_from_yaml(yaml_path):
    """Get number of classes from data.yaml file."""
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data.get('nc', None)

def generate_client_config(num_clients, dataset_path):
    """Dynamically generate client configuration for n clients."""
    return {
        i: {
            'cid': i,
            'data_path': f"{dataset_path}/partitions/client_{i}/data.yaml"
        }
        for i in range(num_clients)
    }

# Base Configuration
if 'microsoft' in platform.uname().release.lower():  # WSL/Bash
    BASE = "/mnt/c/Users/Pranav Rustagi/Desktop/Project"
else:
    BASE = r"C:\Users\Pranav Rustagi\Desktop\Project"

HOME = os.path.join(BASE, "UltraFlwr")
DATASET_NAME = 'baseline'
DATASET_PATH = os.path.join(HOME, 'datasets', DATASET_NAME)
DATA_YAML = os.path.join(DATASET_PATH, 'data.yaml')
# print("DATA_YAML path:", DATA_YAML)
# print("File exists:", os.path.exists(DATA_YAML))
NC = get_nc_from_yaml(DATA_YAML)


# Number of clients can be easily modified here
NUM_CLIENTS = 2  # Change this to desired number of clients

# Generate equal ratios for n clients
CLIENT_RATIOS = [1/NUM_CLIENTS] * NUM_CLIENTS

SPLITS_CONFIG = {
    'dataset_name': DATASET_NAME,
    'num_classes': NC,
    'dataset': DATASET_PATH,
    'num_clients': NUM_CLIENTS,
    'ratio': CLIENT_RATIOS
}

# Dynamically generate client config
CLIENT_CONFIG = generate_client_config(NUM_CLIENTS, DATASET_PATH)

SERVER_CONFIG = {
    'server_address': "127.0.0.1:8080",
    'rounds': 2,
    'sample_fraction': 1.0,
    'min_num_clients': NUM_CLIENTS,
    'max_num_clients': NUM_CLIENTS * 2,  # Adjusted based on number of clients
    'strategy': 'FedAvg',
}

YOLO_CONFIG = {
    'batch_size': 8,
    'epochs': 1,
}