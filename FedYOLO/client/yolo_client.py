import argparse
import warnings
from collections import OrderedDict
import torch
import flwr as fl
from ultralytics import YOLO

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--server_address", type=str, default="0.0.0.0:8080")
parser.add_argument("--cid", type=int, required=True)
parser.add_argument("--data_path", type=str, default="./client_0_assets/dummy_data_0/data.yaml")

NUM_CLIENTS = 3


def train(net, data_path, epochs, cid):
    net.train(data=data_path, epochs=epochs, workers=0, seed=cid)


def test(net, current_round, total_rounds, data_path):
    results = net.val(data=data_path)
    loss = results.results_dict.get('metrics/mAP50(B)')
    accuracy = results.results_dict.get('metrics/precision(B)')
    if current_round < total_rounds:
        net.train(data=data_path, workers=0, epochs=1)
    return loss, accuracy


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, data_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = YOLO()
        self.cid = cid
        self.data_path = data_path

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.model.state_dict().items()]

    def set_parameters(self, parameters, config, detect_module_weights=True):
        if detect_module_weights:
            params_dict = zip(self.net.model.state_dict().keys(), parameters)
            detection_weights = {k: torch.tensor(v) for k, v in params_dict if k.startswith('model.detect')}
            state_dict = OrderedDict(detection_weights)
        else:
            params_dict = zip(self.net.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        self.net.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        self.set_parameters(parameters, config)
        train(self.net, self.data_path, config['epochs'], self.cid)
        return self.get_parameters(config), 10, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters, config)
        loss, accuracy = test(self.net, config["current_round"], config["total_rounds"], self.data_path)
        return loss, len(parameters), {"accuracy": accuracy}


def main():
    args = parser.parse_args()
    assert args.cid < NUM_CLIENTS
    fl.client.start_client(server_address=args.server_address, client=FlowerClient(args.cid, args.data_path))


if __name__ == "__main__":
    main()