SPLITS_CONFIG = {
    'dataset': './datasets/pills',
    'ratio': [0.5, 0.5] # Amount of data per client
}

SERVER_CONFIG = {
    'server_address': "0.0.0.0:8080",
    'rounds': 2,
    'sample_fraction': 1.0,
    'min_num_clients': 2,
    'epochs': 2,
}