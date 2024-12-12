from ultralytics import YOLO
from extract_final_save_from_client import extract_results_path

from FedYOLO.config import HOME, SERVER_CONFIG

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='baseline')
parser.add_argument('--strategy_name', type=str, default='FedAvg')
parser.add_argument('--client_num', type=int, default=1)
parser.add_argument('--scoring_style', type=str, default="client-client")

args = parser.parse_args()

dataset_name = args.dataset_name
strategy_name = args.strategy_name
client_num = args.client_num
scoring_style = args.scoring_style
num_rounds = SERVER_CONFIG['rounds']

def client_client_metrics(client_number, dataset_name, strategy_name):

    logs_path = f"{HOME}/logs/client_{client_number}_log_{dataset_name}_{strategy_name}.txt"
    weights_path = extract_results_path(logs_path)
    weights = f"{HOME}/{weights_path}/weights/best.pt"
    model = YOLO(weights)
    table = model.val(data=f'{HOME}/datasets/{dataset_name}/partitions/client_{client_number}/data.yaml', verbose=True)
    return table

def client_server_metrics(client_number, dataset_name, strategy_name):

    logs_path = f"{HOME}/logs/client_{client_number}_log_{dataset_name}_{strategy_name}.txt"
    weights_path = extract_results_path(logs_path)
    weights = f"{HOME}/{weights_path}/weights/best.pt"
    model = YOLO(weights)
    table = model.val(data=f'{HOME}/datasets/{dataset_name}/data.yaml', verbose=True)
    return table

def server_client_metrics(client_number, dataset_name, strategy_name, num_rounds):

    weights_path = f"{HOME}/weights/model_round_{num_rounds}_{dataset_name}_Strategy_{strategy_name}.pt"
    server_model = YOLO(weights_path)
    normal_model = YOLO(weights_path)

    if strategy_name == 'FedHeadAvg': #! Need to make a config for this as well
        detection_weights = {k: v for k, v in server_model.model.state_dict().items() if k.startswith('model.detect')}
        normal_model.model.load_state_dict({**normal_model.model.state_dict(), **detection_weights}, strict=False)   
        server_model = normal_model 
    
    table = server_model.val(data=f'{HOME}/datasets/{dataset_name}/partitions/client_{client_number}/data.yaml', verbose=True)
    return table

def server_server_metrics(dataset_name, strategy_name, num_rounds):

    weights_path = f"{HOME}/weights/model_round_{num_rounds}_{dataset_name}_Strategy_{strategy_name}.pt"
    server_model = YOLO(weights_path)
    normal_model = YOLO(weights_path)

    if strategy_name == 'FedHeadAvg': #! Need to make a config for this as well
        detection_weights = {k: v for k, v in server_model.model.state_dict().items() if k.startswith('model.detect')}
        normal_model.model.load_state_dict({**normal_model.model.state_dict(), **detection_weights}, strict=False)   
        server_model = normal_model 
    
    table = server_model.val(data=f'{HOME}/datasets/{dataset_name}/data.yaml', verbose=True)
    return table

if scoring_style == "client-client":
    client_metrics_table = client_client_metrics(client_num, dataset_name, strategy_name)
elif scoring_style == "client-server":
    client_metrics_table = client_server_metrics(client_num, dataset_name, strategy_name)
elif scoring_style == "server-client":
    client_metrics_table = server_client_metrics(client_num, dataset_name, strategy_name, num_rounds)
elif scoring_style == "server-server":
    client_metrics_table = server_server_metrics(dataset_name, strategy_name, num_rounds)
else:
    raise ValueError(f"Invalid scoring_style: {scoring_style}")