import flwr as fl
from flwr.common import ndarrays_to_parameters
from ultralytics import YOLO
from FedYOLO.config import SERVER_CONFIG, YOLO_CONFIG

def fit_config(server_round: int):
    return {"epochs": YOLO_CONFIG["epochs"]}

def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.model.state_dict().items()]

#! No Clue How to Get Server Weights
#! Because on the server side, the model is never even seen. Only the parameters are passed around. :(

net = YOLO()
ndarrays = get_parameters(net)
parameters = ndarrays_to_parameters(ndarrays)

def main():
    # Baseline strategy: Federated Averaging
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=SERVER_CONFIG["sample_fraction"],
        min_fit_clients=SERVER_CONFIG["min_num_clients"],
        on_fit_config_fn=fit_config,
        initial_parameters=parameters, #! Need to make a config for this
    )

    fl.server.start_server(
        server_address=SERVER_CONFIG["server_address"],
        config=fl.server.ServerConfig(num_rounds=SERVER_CONFIG["rounds"]),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
