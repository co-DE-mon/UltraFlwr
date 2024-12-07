from ultralytics import YOLO
from extract_final_save_from_client import extract_results_path
import torch
from FedYOLO.config import HOME, SPLITS_CONFIG, SERVER_CONFIG

DATASET_NAME = SPLITS_CONFIG['dataset_name']
NUM_ROUNDS = SERVER_CONFIG['rounds']

#! HUGE DIFFERENCES BETWEEN SYSTEMS REGARDING FILES PATHS AND LOGGING. NEED TO IN-DEPTH TEST THIS.

print('################')
print('### CLIENT 1 ###')
print('################')
client0_model = YOLO()
client0_model_weights_path = extract_results_path(f"{HOME}/logs/client_0_log_{DATASET_NAME}.txt")
weights = HOME + '/' + client0_model_weights_path + "/weights/best.pt" # Can choose last as well
client0_model.load_state_dict(torch.load(weights), strict=False)   # Load the weights from the client
metrics = client0_model.val(data=f'{HOME}/datasets/{DATASET_NAME}/partitions/client_0/data.yaml')  # Validate the model
print()

print('################')
print('### CLIENT 2 ###')
print('################')
client1_model = YOLO()
client1_model_weights_path = extract_results_path(f"{HOME}/logs/client_1_log_{DATASET_NAME}.txt")
weights = HOME + '/' + client1_model_weights_path + "/weights/best.pt" # Can choose last as well
client1_model.load_state_dict(torch.load(weights), strict=False)   # Load the weights from the client
metrics = client1_model.val(data=f'{HOME}/datasets/{DATASET_NAME}/partitions/client_1/data.yaml')  # Validate the model
print()

print('##############')
print('### SERVER ###')
print('##############')
server_model = YOLO()
server_model_weights_path = f"{HOME}/weights/model_weights___round_{NUM_ROUNDS}_{DATASET_NAME}.pth" # Need to automatically take the last round
server_model.model.load_state_dict(torch.load(server_model_weights_path), strict=False)
metrics = server_model.val(data=f'{HOME}/datasets/{DATASET_NAME}/partitions/client_0/data.yaml')  # Validate the model