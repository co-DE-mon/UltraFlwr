home = '/home/localssk23/FedYOLO'

SPLITS_CONFIG = {
    'dataset': f'{home}/datasets/pills',
    'ratio': [0.5, 0.5] # Amount of data per client
}

SERVER_CONFIG = {
    'server_address': "0.0.0.0:8080",
    'rounds': 2,
    'sample_fraction': 1.0,
    'min_num_clients': 2,
}