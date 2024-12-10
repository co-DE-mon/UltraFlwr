import torch

from ultralytics import YOLO
from extract_final_save_from_client import extract_results_path

import pandas as pd

from FedYOLO.config import HOME, SPLITS_CONFIG, SERVER_CONFIG

DATASET_NAME = SPLITS_CONFIG['dataset_name']
NUM_ROUNDS = SERVER_CONFIG['rounds']

#! HUGE DIFFERENCES BETWEEN SYSTEMS REGARDING FILES PATHS AND LOGGING. NEED TO IN-DEPTH TEST THIS.

print('##################')
print('# CLIENT RESULTS #')
print('##################')
print()

def get_client_metrics(client_number, dataset_name):
    # Extract paths
    client_model_weights_path = extract_results_path(f"{HOME}/logs/client_{client_number}_log_{dataset_name}.txt")
    weights = f"{HOME}/{client_model_weights_path}/weights/best.pt"
    
    # Load and validate local model
    client_model = YOLO(weights)
    client_metrics = client_model.val(data=f'{HOME}/datasets/{dataset_name}/partitions/client_{client_number}/data.yaml', verbose=False)
    
    # Create local model metrics table
    client_table = pd.DataFrame({
        'Class': list(client_metrics.names.values()),
        'mAP@0.5:0.95': client_metrics.box.maps.tolist()
    })

    torch.cuda.empty_cache()
    
    # Extract global model weights
    client_global_model_weights_path = extract_results_path(f"{HOME}/logs/client_{client_number}_log_{dataset_name}.txt")
    global_weights = f"{HOME}/{client_global_model_weights_path}/weights/best.pt"
    
    # Load and validate global model
    client_global_model = YOLO(global_weights)
    client_global_metrics = client_global_model.val(data=f'{HOME}/datasets/{dataset_name}/data.yaml', verbose=False)
    
    # Create global model metrics table
    client_global_table = pd.DataFrame({
        'Class': list(client_global_metrics.names.values()),
        'mAP@0.5:0.95': client_global_metrics.box.maps.tolist()
    })

    torch.cuda.empty_cache()
    
    # Combine local and global model metrics
    combined_table = pd.merge(client_table, client_global_table, on='Class', how='inner')
    combined_table.columns = ['Class', 'mAP@0.5:0.95_local', 'mAP@0.5:0.95_global']

    del client_model
    del client_global_model
    
    return combined_table


client_0_metrics_table = get_client_metrics(0, DATASET_NAME)
client_1_metrics_table = get_client_metrics(1, DATASET_NAME)
combined_table = pd.merge(client_0_metrics_table, client_1_metrics_table, on='Class', how='inner')
combined_table.columns = ['Class', 'mAP@0.5:0.95_local_0', 'mAP@0.5:0.95_global_0', 'mAP@0.5:0.95_local_1', 'mAP@0.5:0.95_global_1']

print()
print()
print('#####################################')
print('# FINAL CONSOLIDATED CLIENT METRICS #')
print('#####################################')
print(combined_table.to_string(index=False))
print()
print()

# clear variables and free memory
del client_0_metrics_table
del client_1_metrics_table
del combined_table
torch.cuda.empty_cache()

print('##################')
print('# SERVER RESULTS #')
print('##################')
print()

torch.backends.cudnn.benchmark = True
torch.set_num_threads(1) # https://github.com/ultralytics/yolov5/issues/2960

def safe_get_metrics(metrics):
    try:
        classes = list(metrics.names.values())
        maps = metrics.box.maps.tolist() if len(metrics.box.maps) > 0 else [0] * len(classes)
        return classes, maps
    except AttributeError:
        # print(f"Warning: Metrics object doesn't have expected attributes. Returning empty lists.")
        return [], []

def create_safe_dataframe(metrics, client_name):
    classes, maps = safe_get_metrics(metrics)
    if len(classes) != len(maps):
        print(f"Warning: Mismatch in length for {client_name}. Classes: {len(classes)}, mAP: {len(maps)}")
        maps = maps + [0] * (len(classes) - len(maps))  # Pad with zeros if necessary
    return pd.DataFrame({
        'Class': classes,
        f'mAP@0.5:0.95_{client_name}': maps
    })

def get_server_metrics(weights_path, dataset_name):
    server_model = YOLO(weights_path)
    server_model_metrics = server_model.val(data=dataset_name, verbose=False, batch=16)
    del server_model
    torch.cuda.empty_cache()
    return server_model_metrics

server_model_weights_path = f"{HOME}/weights/model_round_{NUM_ROUNDS}_{DATASET_NAME}.pt"

server_model_client0_dataset = f'{HOME}/datasets/{DATASET_NAME}/partitions/client_0/data.yaml'
server_model_client0_table = get_server_metrics(server_model_weights_path, server_model_client0_dataset)
server_model_client0_table = create_safe_dataframe(server_model_client0_table, 'client_0')

server_model_client1_dataset = f'{HOME}/datasets/{DATASET_NAME}/partitions/client_1/data.yaml'
server_model_client1_table = get_server_metrics(server_model_weights_path, server_model_client1_dataset)
server_model_client1_table = create_safe_dataframe(server_model_client1_table, 'client_1')

server_model_global_dataset = f'{HOME}/datasets/{DATASET_NAME}/data.yaml'
server_model_global_table = get_server_metrics(server_model_weights_path, server_model_global_dataset)
server_model_global_table = create_safe_dataframe(server_model_global_table, 'global')

# Merge client based scores
client_based_table = pd.merge(server_model_client0_table, server_model_client1_table, on='Class', how='inner')
# Merge with global scores
server_model_combined_table = pd.merge(client_based_table, server_model_global_table, on='Class', how='inner')

# Fill NaN values with 0
server_model_combined_table = server_model_combined_table.fillna(0)

# Reorder columns if needed
column_order = ['Class', 'mAP@0.5:0.95_client_0', 'mAP@0.5:0.95_client_1', 'mAP@0.5:0.95_global']
server_model_combined_table = server_model_combined_table[column_order]

# Convert mAP values to percentage and round to 2 decimal places
for col in server_model_combined_table.columns[1:]:
    server_model_combined_table[col] = (server_model_combined_table[col] * 100).round(2)

print()
print()
print('#####################################')
print('# FINAL CONSOLIDATED SERVER METRICS #')
print('#####################################')
print(server_model_combined_table.to_string(index=False))
print()
print()