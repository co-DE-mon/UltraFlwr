import io
import torch
import numpy as np
from datetime import datetime
from collections import OrderedDict
from typing import Optional, Union

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

from ultralytics import YOLO
from ultralytics.utils import __version__

from FedYOLO.config import SERVER_CONFIG, YOLO_CONFIG, SPLITS_CONFIG, HOME

def fit_config(server_round: int) -> dict:
    """Return training configuration for each round."""
    return {"epochs": YOLO_CONFIG["epochs"]}


def get_parameters(net: YOLO) -> list[np.ndarray]:
    """Extract model parameters from YOLO model."""
    return [val.cpu().numpy() for _, val in net.model.state_dict().items()]


def save_model_checkpoint(server_round: int, model=None) -> None:
    """Save model training checkpoints with additional metadata."""
    buffer = io.BytesIO()
    torch.save(
        {
            "epoch": 0,
            "best_fitness": 0,
            "model": model,
            "ema": 0,
            "updates": 0,
            "optimizer": 0,
            "train_args": 0,
            "train_metrics": 0,
            "train_results": 0,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": __version__,
            "license": "AGPL-3.0 (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        },
        buffer,
    )
    ckpt_path = f"{HOME}/weights/model_round_{server_round}_{SPLITS_CONFIG['dataset_name']}.pt"
    with open(ckpt_path, "wb") as f:
        f.write(buffer.getvalue())


class SaveModelStrategy(fl.server.strategy.FedAvg):
    """Custom FL strategy to save aggregated YOLO model checkpoints."""

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint."""
        net = YOLO(f"{HOME}/yolo11n_{SPLITS_CONFIG['dataset_name']}.yaml")

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")
            aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            save_model_checkpoint(server_round, model=net.model)

        return aggregated_parameters, aggregated_metrics

def main() -> None:
    """Start the FL server with custom strategy."""
    initial_parameters = ndarrays_to_parameters(get_parameters(YOLO()))
    strategy = SaveModelStrategy(
        fraction_fit=SERVER_CONFIG["sample_fraction"],
        min_fit_clients=SERVER_CONFIG["min_num_clients"],
        on_fit_config_fn=fit_config,
        initial_parameters=initial_parameters,
    )

    fl.server.start_server(
        server_address=SERVER_CONFIG["server_address"],
        config=fl.server.ServerConfig(num_rounds=SERVER_CONFIG["rounds"]),
        strategy=strategy,
    )

def write_yolo_config(dataset_name, num_classes=None):
    content = f"""# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLO11 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect

# Parameters
nc: {str(num_classes)} # number of classes

# Dataset
dataset: {dataset_name}

scales: # model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo11.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024] # summary: 319 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
  s: [0.50, 0.50, 1024] # summary: 319 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
  m: [0.50, 1.00, 512] # summary: 409 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
  l: [1.00, 1.00, 512] # summary: 631 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
  x: [1.00, 1.50, 512] # summary: 631 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs

# YOLO11n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9
  - [-1, 2, C2PSA, [1024]] # 10

# YOLO11n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, False]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, False]] # 16 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, False]] # 19 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 2, C3k2, [1024, True]] # 22 (P5/32-large)

  - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)
"""

    filename = f"{HOME}/yolo11n_{dataset_name}.yaml"
    with open(filename, "w") as file:
        file.write(content)
    
    print(f"YAML configuration file '{filename}' has been created.")

if __name__ == "__main__":
    dataset_name = SPLITS_CONFIG['dataset_name']
    num_classes = SPLITS_CONFIG['num_classes']
    write_yolo_config(dataset_name, num_classes)
    main()

#1. There is no direct way to define and update YOLO when the number of classes change.