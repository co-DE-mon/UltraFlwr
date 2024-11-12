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

warnings.filterwarnings("ignore", category=UserWarning)
NUM_CLIENTS = 3

# Need to avoid training or inferencing on the same model at the same time
# https://docs.ultralytics.com/guides/yolo-thread-safe-inference/#why-should-each-thread-have-its-own-yolo-model-instance

def train(net, epochs):
    net.train(data="./client_1_assets/dummy_data_2/data.yaml", epochs=epochs, workers=0)

def test(net):
    """Validate the model on the specified dataset."""
    results = net.val(data="./client_1_assets/dummy_data_3/data.yaml") # if don't work, use dummy_data_2
    val_mAP50 = results.results_dict.get('metrics/mAP50(B)')
    val_precision = results.results_dict.get('metrics/precision(B)')
    loss = val_mAP50
    accuracy = val_precision
    net.train()
    return loss, accuracy

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = YOLO("./client_1_assets/yolov8n_1.pt").to(self.device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.model.state_dict().items()]

    def set_parameters(self, parameters, config):
        params_dict = zip(self.net.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters, config)
        train(self.net, config['epochs'])
        return self.get_parameters(config), 10, {} # 10 is replacing the number of samples trained on this client

    def evaluate(self, parameters, config):
        self.set_parameters(parameters, config)
        loss, accuracy = test(self.net)
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