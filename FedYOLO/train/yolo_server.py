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

from FedYOLO.config import SERVER_CONFIG, YOLO_CONFIG


def fit_config(server_round: int) -> dict:
    """Return training configuration for each round."""
    return {"epochs": YOLO_CONFIG["epochs"]}


def get_parameters(net: YOLO) -> list[np.ndarray]:
    """Extract model parameters from YOLO model."""
    return [val.cpu().numpy() for _, val in net.model.state_dict().items()]


def set_parameters(net: YOLO, ndarrays: list[np.ndarray], server_round: int) -> OrderedDict:
    """Set model parameters from a list of ndarrays."""
    state_dict = net.model.state_dict()
    total_skipped = 0

    for (name, param), ndarray in zip(state_dict.items(), ndarrays):
        new_tensor = torch.tensor(ndarray)
        if param.shape == new_tensor.shape:
            state_dict[name] = new_tensor
        else:
            total_skipped += 1

    print(f"Skipped {total_skipped} layers")
    net.model.load_state_dict(state_dict, strict=False)
    torch.save(state_dict, f"/home/localssk23/FedDet/weights/model_round_{server_round}.pt")
    return state_dict


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
    ckpt_path = f"/home/localssk23/FedDet/weights/model_round_{server_round}_WOW.pt"
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
        net = YOLO('/home/localssk23/FedDet/yolo11n_nc8.yaml')

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


if __name__ == "__main__":
    main()