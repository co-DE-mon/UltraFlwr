import argparse
import warnings
from collections import OrderedDict
import torch
import flwr as fl
from ultralytics import YOLO
from FedYOLO.config import SERVER_CONFIG, YOLO_CONFIG

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--cid", type=int, required=True)
parser.add_argument("--data_path", type=str, default="./client_0_assets/dummy_data_0/data.yaml")

NUM_CLIENTS = 3


def train(net, data_path, cid):
    net.train(data=data_path, epochs=YOLO_CONFIG['epochs'], workers=0, seed=cid, batch=YOLO_CONFIG['batch_size'])

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, data_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = YOLO()
        self.cid = cid
        self.data_path = data_path

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.net.model.state_dict().items()]

    def set_parameters(self, parameters, detect_module_weights=False): #! Need to make a config for this
        if detect_module_weights:
            params_dict = zip(self.net.model.state_dict().keys(), parameters)
            detection_weights = {k: torch.tensor(v) for k, v in params_dict if k.startswith('model.detect')}
            state_dict = OrderedDict(detection_weights)
        else:
            params_dict = zip(self.net.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        self.net.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.data_path, self.cid)
        return self.get_parameters(), 10, {}


def main():
    args = parser.parse_args()
    assert args.cid < NUM_CLIENTS
    fl.client.start_client(server_address=SERVER_CONFIG['server_address'], client=FlowerClient(args.cid, args.data_path))


if __name__ == "__main__":
    main()