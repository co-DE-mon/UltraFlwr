import argparse
import warnings
from collections import OrderedDict

import torch

import flwr as fl

from ultralytics import YOLO


parser = argparse.ArgumentParser(description='YOLO client')
parser.add_argument(
    "--server_address",
    type=str,
    default="0.0.0.0:8080",
    help=f"gRPC server address (default '0.0.0.0:8080')",
)
parser.add_argument(
    "--cid",
    type=int,
    required=True,
    help="Client id. Should be an integer between 0 and NUM_CLIENTS",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=2,
    help="Number of rounds of federated learning (default: 2)",
)

warnings.filterwarnings("ignore", category=UserWarning)
NUM_CLIENTS = 3

# Need to avoid training or inferencing on the same model at the same time
# https://docs.ultralytics.com/guides/yolo-thread-safe-inference/#why-should-each-thread-have-its-own-yolo-model-instance

def train(net, epochs):
    net.train(data="./client_0_assets/dummy_data_0/data.yaml", epochs=epochs, workers=0, seed=1) # Flower does not support multi-threading

def test_and_continue(net):
    """Validate the model on the specified dataset."""
    results = net.val(data="./client_0_assets/dummy_data_0/data.yaml") #! this is messing up the training, need to set to trainmode
    val_mAP50 = results.results_dict.get('metrics/mAP50(B)')
    val_precision = results.results_dict.get('metrics/precision(B)')
    loss = val_mAP50
    accuracy = val_precision
    net.train(data="./client_0_assets/dummy_data_0/data.yaml", workers=0, epochs=1)
    return loss, accuracy

def test_final(net):
    """Validate the model on the specified dataset."""
    results = net.val(data="./client_0_assets/dummy_data_0/data.yaml") #! this is messing up the training, need to set to trainmode
    val_mAP50 = results.results_dict.get('metrics/mAP50(B)')
    val_precision = results.results_dict.get('metrics/precision(B)')
    loss = val_mAP50
    accuracy = val_precision
    return loss, accuracy

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.net = YOLO("./client_0_assets/yolov8n_0.pt")
        self.net = YOLO()
        self.get_parameters_count = 0
        self.set_parameters_count = 0

    def get_parameters(self, config):
        self.get_parameters_count += 1
        print(f"get_parameters called {self.get_parameters_count} times")
        return [val.cpu().numpy() for _, val in self.net.model.state_dict().items()]

    def set_parameters(self, parameters, config):
        self.set_parameters_count += 1
        print(f"set_parameters called {self.set_parameters_count} times")
        params_dict = zip(self.net.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters, config)
        train(self.net, config['epochs'])
        return self.get_parameters(config), 10, {} # 10 is replacing the number of samples trained on this client

    def evaluate(self, parameters, config):
        self.set_parameters(parameters, config)
        if config['current_round'] == 3:
            loss, accuracy = test_final(self.net)
        else:
            loss, accuracy = test_and_continue(self.net)
        # loss = 1.0
        # accuracy = 1.0
        return loss, len(parameters), {"accuracy": accuracy}


def main():
    args = parser.parse_args()
    print(args)

    assert args.cid < NUM_CLIENTS

    fl.client.start_client(
        server_address=args.server_address,
        client=FlowerClient(),
    )

if __name__ == "__main__":
    main()