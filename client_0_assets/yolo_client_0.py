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
    "--data_path",
    type=str,
    default="./client_0_assets/dummy_data_0/data.yaml",
    help="Path to the dataset YAML file (default: ./client_0_assets/dummy_data_0/data.yaml)",
)

warnings.filterwarnings("ignore", category=UserWarning)
NUM_CLIENTS = 3

# Need to avoid training or inferencing on the same model at the same time
# https://docs.ultralytics.com/guides/yolo-thread-safe-inference/#why-should-each-thread-have-its-own-yolo-model-instance

def train(net, data_path, epochs, cid):
    net.train(data=data_path, epochs=epochs, workers=0, seed=cid) # Flower does not support multi-threading


def test(net, current_round, total_rounds, data_path):
    """
    Validate the model on the specified dataset and optionally set it to train mode.
    
    Args:
        net: The model/network object.
        current_round (int): The current federated learning round.
        total_rounds (int): The total number of federated learning rounds.
        data_path (str): Path to the dataset YAML file.

    Returns:
        tuple: A tuple containing loss (float) and accuracy (float).
    """
    try:
        # Validate the model
        results = net.val(data=data_path)
        val_mAP50 = results.results_dict.get('metrics/mAP50(B)')
        val_precision = results.results_dict.get('metrics/precision(B)')
        
        # Check if metrics are available
        if val_mAP50 is None or val_precision is None:
            raise KeyError("Validation metrics 'mAP50(B)' or 'precision(B)' not found in results.")
        
        # Use metrics as loss and accuracy
        loss = val_mAP50
        accuracy = val_precision

        # Set to trainmode if not the final round
        if current_round < total_rounds:
            net.train(data=data_path, workers=0, epochs=1)
        else:
            print("Final round.")
        
        return loss, accuracy

    except KeyError as e:
        print(f"Error accessing validation metrics: {e}")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, data_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.net = YOLO("./client_0_assets/yolov8n_0.pt")
        self.net = YOLO()
        self.get_parameters_count = 0
        self.set_parameters_count = 0
        self.cid = cid
        self.data_path = data_path

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
        train(self.net, self.data_path, config['epochs'], self.cid)
        return self.get_parameters(config), 10, {} # 10 is replacing the number of samples trained on this client

    def evaluate(self, parameters, config):
        self.set_parameters(parameters, config)
        loss, accuracy = test(self.net, config["current_round"], config["total_rounds"], self.data_path)
        # loss = 1.0
        # accuracy = 1.0
        return loss, len(parameters), {"accuracy": accuracy}


def main():
    args = parser.parse_args()
    print(args)

    assert args.cid < NUM_CLIENTS

    fl.client.start_client(
        server_address=args.server_address,
        client=FlowerClient(args.cid, args.data_path),
    )

if __name__ == "__main__":
    main()