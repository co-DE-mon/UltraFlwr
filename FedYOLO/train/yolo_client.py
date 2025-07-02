import argparse
from pathlib import Path
import warnings
from collections import OrderedDict
import torch
import flwr as fl
from ultralytics import YOLO
from FedYOLO.config import SERVER_CONFIG, YOLO_CONFIG, SPLITS_CONFIG, HOME
from FedYOLO.test.extract_final_save_from_client import extract_results_path
from ultralytics.utils.loss import ProximalDetectionLoss
from FedYOLO.train.client_utils import parameters_to_state_dict

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--cid", type=int, required=True)
parser.add_argument("--data_path", type=str, default="./client_0_assets/dummy_data_0/data.yaml")

NUM_CLIENTS = SERVER_CONFIG['max_num_clients']

def train(net, data_path, cid, strategy, proximal_mu=None):
    if proximal_mu is not None and "Prox" in strategy:
        # The proximal loss should already be set in the fit method
        # Just verify it's there
        if hasattr(net.model, 'loss') and isinstance(net.model.loss, ProximalDetectionLoss):
            print(f"[TRAIN] Using ProximalDetectionLoss with mu={net.model.loss.proximal_mu}")
        else:
            print(f"[TRAIN] Warning: ProximalDetectionLoss not properly set!")
    
    net.train(data=data_path, epochs=YOLO_CONFIG['epochs'], workers=0, seed=cid, batch=YOLO_CONFIG['batch_size'], project=strategy)
# Define get_section_parameters as a standalone function
from typing import Tuple
def get_section_parameters(state_dict: OrderedDict) -> Tuple[dict, dict, dict]:
    """Get parameters for each section of the model."""
    # Backbone parameters (early layers through conv layers)
    # backbone corresponds to:
    # (0): Conv
    # (1): Conv
    # (2): C3k2
    # (3): Conv
    # (4): C3k2
    # (5): Conv
    # (6): C3k2
    # (7): Conv
    # (8): C3k2
    backbone_weights = {
        k: v for k, v in state_dict.items()
        if not k.startswith(tuple(f'model.{i}' for i in range(9, 24)))
    }

    # Neck parameters
    # The neck consists of the following layers (by index in the Sequential container):
    # (9): SPPF
    # (10): C2PSA
    # (11): Upsample
    # (12): Concat
    # (13): C3k2
    # (14): Upsample
    # (15): Concat
    # (16): C3k2
    # (17): Conv
    # (18): Concat
    # (19): C3k2
    # (20): Conv
    # (21): Concat
    # (22): C3k2
    neck_weights = {
        k: v for k, v in state_dict.items()
        if k.startswith(tuple(f'model.{i}' for i in range(9, 23)))
    }

    # Head parameters (detection head)
    head_weights = {
        k: v for k, v in state_dict.items()
        if k.startswith('model.23')
    }

    return backbone_weights, neck_weights, head_weights

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, data_path, dataset_name, strategy_name):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = YOLO()
        self.cid = cid
        self.data_path = data_path
        self.dataset_name=dataset_name
        self.strategy_name=strategy_name
        self.proximal_loss = None

    def get_parameters(self):
        """Get relevant model parameters based on the strategy."""
        current_state_dict = self.net.model.state_dict()
        # Use the imported function
        backbone_weights, neck_weights, head_weights = get_section_parameters(current_state_dict)

        # Define strategy groups (same as in set_parameters) - Corrected lists
        backbone_strategies = [
            'FedAvg', 'FedBackboneAvg', 'FedBackboneHeadAvg', 'FedBackboneNeckAvg',
            'FedMedian', 'FedBackboneMedian', 'FedBackboneHeadMedian', 'FedBackboneNeckMedian',
            'FedProx', 'FedBackboneProx', 'FedBackboneHeadProx', 'FedBackboneNeckProx'
        ]
        neck_strategies = [
            'FedAvg', 'FedNeckAvg', 'FedNeckHeadAvg', 'FedBackboneNeckAvg',
            'FedMedian', 'FedNeckMedian', 'FedNeckHeadMedian', 'FedBackboneNeckMedian',
            'FedProx', 'FedNeckProx', 'FedNeckHeadProx', 'FedBackboneNeckProx'
        ]
        head_strategies = [
            'FedAvg', 'FedHeadAvg', 'FedNeckHeadAvg', 'FedBackboneHeadAvg',
            'FedMedian', 'FedHeadMedian', 'FedNeckHeadMedian', 'FedBackboneHeadMedian',
            'FedProx', 'FedHeadProx', 'FedNeckHeadProx', 'FedBackboneHeadProx'
        ]

        # Determine which parts to send based on strategy
        send_backbone = self.strategy_name in backbone_strategies
        send_neck = self.strategy_name in neck_strategies
        send_head = self.strategy_name in head_strategies

        relevant_parameters = []
        for k, v in current_state_dict.items():
            if (send_backbone and k in backbone_weights) or \
               (send_neck and k in neck_weights) or \
               (send_head and k in head_weights):
                relevant_parameters.append(v.cpu().numpy())
        
        return relevant_parameters

    def set_parameters(self, parameters):
        # Get current client model state and split into sections
        current_state_dict = self.net.model.state_dict()
        backbone_weights, neck_weights, head_weights = get_section_parameters(current_state_dict)
        
        # Define strategy groups
        backbone_strategies = [
            'FedAvg', 'FedBackboneAvg', 'FedBackboneHeadAvg', 'FedBackboneNeckAvg',
            'FedMedian', 'FedBackboneMedian', 'FedBackboneHeadMedian', 'FedBackboneNeckMedian',
            'FedProx', 'FedBackboneProx', 'FedBackboneHeadProx', 'FedBackboneNeckProx'
        ]
        neck_strategies = [
            'FedAvg', 'FedNeckAvg', 'FedNeckHeadAvg', 'FedBackboneNeckAvg',
            'FedMedian', 'FedNeckMedian', 'FedNeckHeadMedian', 'FedBackboneNeckMedian',
            'FedProx', 'FedNeckProx', 'FedNeckHeadProx', 'FedBackboneNeckProx'
        ]
        head_strategies = [
            'FedAvg', 'FedHeadAvg', 'FedNeckHeadAvg', 'FedBackboneHeadAvg',
            'FedMedian', 'FedHeadMedian', 'FedNeckHeadMedian', 'FedBackboneHeadMedian',
            'FedProx', 'FedHeadProx', 'FedNeckHeadProx', 'FedBackboneHeadProx'
        ]
        
        # Determine which parts to update
        update_backbone = self.strategy_name in backbone_strategies
        update_neck = self.strategy_name in neck_strategies
        update_head = self.strategy_name in head_strategies
        
        # Create the SAME key order that get_parameters() used
        relevant_keys = []
        for k in current_state_dict.keys():
            if (update_backbone and k in backbone_weights) or \
            (update_neck and k in neck_weights) or \
            (update_head and k in head_weights):
                relevant_keys.append(k)
        
        # Verify parameter count matches
        if len(parameters) != len(relevant_keys):
            print(f"ERROR: Expected {len(relevant_keys)} parameters, got {len(parameters)}")
            return
        
        # NOW zip with the correct keys
        params_dict = zip(relevant_keys, parameters)
        
        # Apply the parameters
        updated_weights = {}
        for k, v in params_dict:
            updated_weights[k] = torch.tensor(v)

        updated_state_dict = OrderedDict(updated_weights)
    
        if "Prox" in self.strategy_name:
            if not hasattr(self, 'current_proximal_mu'):
                self.current_proximal_mu = SERVER_CONFIG["proximal_mu"]
            if not hasattr(self, "proximal_loss"):
                self.proximal_loss = ProximalDetectionLoss(
                    model=self.net.model,
                    global_params=updated_state_dict,
                    proximal_mu=self.current_proximal_mu
                )
            else:
                self.proximal_loss.update_global_params(updated_state_dict)

            self.net.model.loss = self.proximal_loss

        self.net.model.load_state_dict(updated_state_dict, strict=False)

    def fit(self, parameters, config):
        if config["server_round"] != 1:
            del self.net
            torch.cuda.empty_cache()
            # get the path of the saved model weight
            logs_path = f"{HOME}/logs/client_{self.cid}_log_{self.dataset_name}_{self.strategy_name}.txt"
            weights_path = extract_results_path(logs_path)
            weights = f"{HOME}/{weights_path}/weights/best.pt"
            print(weights)

            self.net = YOLO(weights)

        self.set_parameters(parameters) # this needs to be modified so we only asign parts of the weights
        proximal_mu = config.get("proximal_mu", SERVER_CONFIG["proximal_mu"])
        print(f"[Client {self.cid}] Using proximal_mu = {proximal_mu}")  # Add this line for debugging

        if "Prox" in self.strategy_name:
            if self.proximal_loss is None:
                # Convert parameters to state_dict format
                param_keys = list(self.net.model.state_dict().keys())
                global_params_dict = dict(zip(param_keys, [torch.tensor(p) for p in parameters]))
                self.proximal_loss = ProximalDetectionLoss(
                        model=self.net.model,
                        global_params=global_params_dict,  # ✅ Proper format
                        proximal_mu=proximal_mu
                    )
            else:
                param_keys = list(self.net.model.state_dict().keys())
                global_params_dict = dict(zip(param_keys, [torch.tensor(p) for p in parameters]))
                self.proximal_loss.update_global_params(dict(zip(self.net.model.state_dict().keys(), parameters)))
                self.proximal_loss.proximal_mu = proximal_mu

        self.net.model.loss = self.proximal_loss
        print(f"[CLIENT {self.cid}] ProximalDetectionLoss updated with proximal_mu: {self.proximal_loss.proximal_mu}")


        train(self.net, self.data_path, self.cid, f"{self.strategy_name}_{self.dataset_name}_{self.cid}", proximal_mu=proximal_mu)
        return self.get_parameters(), 10, {}


def main():

    args = parser.parse_args()

    args.data_path = str(Path(args.data_path))
    assert args.cid < NUM_CLIENTS
    fl.client.start_client(
        server_address=SERVER_CONFIG['server_address'],
        client=FlowerClient(
            args.cid,
            args.data_path,
            SPLITS_CONFIG['dataset_name'],
            SERVER_CONFIG['strategy']
        ).to_client()
    )

if __name__ == "__main__":
    main()
    
