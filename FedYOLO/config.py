HOME = "/home/localssk23/FedDet"

SPLITS_CONFIG = {
    'dataset_name': 'pills',
    'dataset': './datasets/pills',
    'ratio': [0.5, 0.5] # Amount of data per client
}

CLIENT_CONFIG = {
    0: {
        'cid': 0,
        'data_path': "/home/localssk23/FedDet/datasets/pills/partitions/client_0/data.yaml"
    },
    1: {
        'cid': 1,
        'data_path': "/home/localssk23/FedDet/datasets/pills/partitions/client_1/data.yaml"
    }
}

SERVER_CONFIG = {
    'server_address': "0.0.0.0:8080",
    'rounds': 2,
    'sample_fraction': 1.0,
    'min_num_clients': 2,
}

YOLO_CONFIG = {
    'batch_size': 8,
    'epochs': 2,
}