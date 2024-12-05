from ultralytics import YOLO
from extract_final_save_from_client import extract_results_path
import torch
import numpy as np

client1_model = YOLO()
client1_model_weights_path = extract_results_path("/home/yang/Documents/GitHub/FedDet/client_1_log.txt")
weights = client1_model_weights_path + "/weights/best.pt" # Can choose last as well
client1_model.load_state_dict(torch.load(weights), strict=False)   # Load the weights from the client
metrics = client1_model.val(data='/home/yang/Documents/GitHub/FedDet/datasets/pills/partitions/client_0/data.yaml')  # Validate the model
print(metrics)  # Print the metrics

server_model = YOLO()
server_model_weights_path = "/home/yang/Documents/GitHub/FedDet/model_weights_round_1.pth"


# Load the updated state_dict into the model
server_model.model.load_state_dict(torch.load(server_model_weights_path), strict=False)

print("Weights loaded successfully!")

metrics = server_model.val(data='/home/yang/Documents/GitHub/FedDet/datasets/pills/partitions/client_0/data.yaml')  # Validate the model