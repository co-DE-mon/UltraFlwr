import flwr as fl
from flwr.common import ndarrays_to_parameters
from ultralytics import YOLO

import sys
sys.path.append('/home/localssk23/FedYOLO/')
from config import SERVER_CONFIG

def fit_config(server_round: int):
    return {"epochs": 30}

def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.model.state_dict().items()]

ndarrays = get_parameters(YOLO())
parameters = ndarrays_to_parameters(ndarrays)

def main():
    def evaluate_config(server_round: int):
        return {
            "current_round": server_round,
            "total_rounds": SERVER_CONFIG["rounds"],
        }

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=SERVER_CONFIG["sample_fraction"],
        min_fit_clients=SERVER_CONFIG["min_num_clients"],
        on_fit_config_fn=fit_config,
        initial_parameters=parameters,
        on_evaluate_config_fn=evaluate_config,
    )

    fl.server.start_server(
        server_address=SERVER_CONFIG["server_address"],
        config=fl.server.ServerConfig(num_rounds=SERVER_CONFIG["rounds"]),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
