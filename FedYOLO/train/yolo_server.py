import numpy as np

import flwr as fl
from flwr.common import ndarrays_to_parameters

from ultralytics import YOLO

from FedYOLO.train.server_utils import write_yolo_config
from FedYOLO.train.strategies import FedAvg, FedHeadAvg, FedMedian, FedHeadMedian

from FedYOLO.config import SERVER_CONFIG, YOLO_CONFIG, SPLITS_CONFIG, HOME


def fit_config(server_round: int) -> dict:
    """Return training configuration for each round."""
    return {"epochs": YOLO_CONFIG["epochs"]}


def get_parameters(net: YOLO) -> list[np.ndarray]:
    """Extract model parameters from YOLO model."""
    return [val.cpu().numpy() for _, val in net.model.state_dict().items()]


def initialize_yolo(dataset_name: str, num_classes: int) -> YOLO:
    """Initialize YOLO model with the specified dataset and number of classes."""
    write_yolo_config(dataset_name, num_classes)
    return YOLO(f"{HOME}/FedYOLO/yolo_configs/yolo11n_{dataset_name}.yaml")

def get_strategy(strategy_class, initial_parameters) -> fl.server.strategy.Strategy:
    """Initialize the strategy using the provided class and configuration."""

    strategies = {
        "FedAvg": FedAvg,
        "FedHeadAvg": FedHeadAvg,
        "FedMedian": FedMedian,
        "FedHeadMedian": FedHeadMedian,
    }

    strategy_class = strategies.get(strategy_class)
    if strategy_class is None:
        raise ValueError(f"Unknown strategy class: {strategy_class}")

    return strategy_class(
        fraction_fit=SERVER_CONFIG["sample_fraction"],
        min_fit_clients=SERVER_CONFIG["min_num_clients"],
        on_fit_config_fn=fit_config,
        initial_parameters=initial_parameters,
    )

def main() -> None:
    """Start the FL server with custom strategy."""
    yolo_model = initialize_yolo(SPLITS_CONFIG['dataset_name'], SPLITS_CONFIG['num_classes'])
    initial_parameters = ndarrays_to_parameters(get_parameters(yolo_model))
    
    strategy = get_strategy(SERVER_CONFIG["strategy"], initial_parameters)
    
    fl.server.start_server(
        server_address=SERVER_CONFIG["server_address"],
        config=fl.server.ServerConfig(num_rounds=SERVER_CONFIG["rounds"]),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()