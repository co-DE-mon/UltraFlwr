import yaml

HOME = "/home/yang/Documents/GitHub/FedDet"
DATASET_NAME = 'baseline'
DATASET_PATH = f'{HOME}/datasets/{DATASET_NAME}'

#? Directly gettin the number of classes for a datset from the data.yaml file
def get_nc_from_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data.get('nc', None)

# Read nc from the first client's data.yaml file
DATA_YAML = f"{DATASET_PATH}/data.yaml"
NC = get_nc_from_yaml(DATA_YAML)

SPLITS_CONFIG = {
    'dataset_name': DATASET_NAME,
    'num_classes': NC,
    'dataset': DATASET_PATH,
    'ratio': [0.5, 0.5]  # Amount of data per client
}

CLIENT_CONFIG = {
    0: {
        'cid': 0,
        'data_path': f"{DATASET_PATH}/partitions/client_0/data.yaml"
    },
    1: {
        'cid': 1,
        'data_path': f"{DATASET_PATH}/partitions/client_1/data.yaml"
    }
}

SERVER_CONFIG = {
    'server_address': "0.0.0.0:8080",
    'rounds': 1,
    'sample_fraction': 1.0,
    'min_num_clients': 2,
}

YOLO_CONFIG = {
    'batch_size': 8,
    'epochs': 1,
    'detect_module_weights': True #! This needs to be integrated into flower potentially?? Or maybe actually a custom strategy where the weights things is don eon the server code. #wontfix for now
}
