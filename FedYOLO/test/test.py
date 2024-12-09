import torch

from ultralytics import YOLO
from extract_final_save_from_client import extract_results_path

import pandas as pd

from FedYOLO.config import HOME, SPLITS_CONFIG, SERVER_CONFIG

DATASET_NAME = SPLITS_CONFIG['dataset_name']
NUM_ROUNDS = SERVER_CONFIG['rounds']

#! HUGE DIFFERENCES BETWEEN SYSTEMS REGARDING FILES PATHS AND LOGGING. NEED TO IN-DEPTH TEST THIS.

#####################
# CLIENT EVALUATION #
#####################

def get_client_metrics(client_number, dataset_name, home_path):
    # Extract paths
    client_model_weights_path = extract_results_path(f"{home_path}/logs/client_{client_number}_log_{dataset_name}.txt")
    weights = f"{home_path}/{client_model_weights_path}/weights/best.pt"
    
    # Load and validate local model
    client_model = YOLO(weights)
    client_metrics = client_model.val(data=f'{home_path}/datasets/{dataset_name}/partitions/client_{client_number}/data.yaml', verbose=False)
    
    # Create local model metrics table
    client_table = pd.DataFrame({
        'Class': list(client_metrics.names.values()),
        'mAP@0.5:0.95': client_metrics.box.maps.tolist()
    })
    
    # Extract global model weights
    client_global_model_weights_path = extract_results_path(f"{home_path}/logs/client_{client_number}_log_{dataset_name}.txt")
    global_weights = f"{home_path}/{client_global_model_weights_path}/weights/best.pt"
    
    # Load and validate global model
    client_global_model = YOLO(global_weights)
    client_global_metrics = client_global_model.val(data=f'{home_path}/datasets/{dataset_name}/data.yaml', verbose=False)
    
    # Create global model metrics table
    client_global_table = pd.DataFrame({
        'Class': list(client_global_metrics.names.values()),
        'mAP@0.5:0.95': client_global_metrics.box.maps.tolist()
    })
    
    # Combine local and global model metrics
    combined_table = pd.merge(client_table, client_global_table, on='Class', how='inner')
    combined_table.columns = ['Class', 'mAP@0.5:0.95_local', 'mAP@0.5:0.95_global']

    del client_model
    del client_global_model
    
    return combined_table

print('##################')
print('# CLIENT RESULTS #')
print('##################')
print()
client_0_metrics_table = get_client_metrics(0, DATASET_NAME, HOME)
client_1_metrics_table = get_client_metrics(1, DATASET_NAME, HOME)
combined_table = pd.merge(client_0_metrics_table, client_1_metrics_table, on='Class', how='inner')
combined_table.columns = ['Class', 'mAP@0.5:0.95_local_0', 'mAP@0.5:0.95_global_0', 'mAP@0.5:0.95_local_1', 'mAP@0.5:0.95_global_1']
print()
print()
print('##############################')
print('# FINAL CONSOLIDATED METRICS #')
print('##############################')
print(combined_table.to_string(index=False))
print()
print()

#####################
# SERVER EVALUATION #
#####################
print('##################')
print('# SERVER RESULTS #')
print('##################')
print()

server_model = YOLO('/home/localssk23/FedDet/yolo11n_nc8.yaml')
server_model_weights_path = f"/home/localssk23/FedDet/weights/model_round_{NUM_ROUNDS}_WOW.pt"
server_model.model.load_state_dict(torch.load(server_model_weights_path)['model'].state_dict(), strict=False)

server_model_client0_metrics = server_model.val(data=f'{HOME}/datasets/{DATASET_NAME}/partitions/client_0/data.yaml', verbose=True)
server_model_client1_metrics = server_model.val(data=f'{HOME}/datasets/{DATASET_NAME}/partitions/client_1/data.yaml', verbose=True)
server_model_global_metrics = server_model.val(data=f'{HOME}/datasets/{DATASET_NAME}/data.yaml', verbose=True)

server_model_client0_table = pd.DataFrame({
    'Class': list(server_model_client0_metrics.names.values()),
    'mAP@0.5:0.95': server_model_client0_metrics.box.maps.tolist()
})

server_model_client1_table = pd.DataFrame({
    'Class': list(server_model_client1_metrics.names.values()),
    'mAP@0.5:0.95': server_model_client1_metrics.box.maps.tolist()
})

server_model_global_table = pd.DataFrame({
    'Class': list(server_model_global_metrics.names.values()),
    'mAP@0.5:0.95': server_model_global_metrics.box.maps.tolist()
})

server_model_combined_table = pd.merge(server_model_client0_table, server_model_client1_table, on='Class', how='inner')