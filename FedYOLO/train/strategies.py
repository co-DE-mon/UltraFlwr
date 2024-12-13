import torch
from collections import OrderedDict
from typing import Optional, Union

import flwr as fl
from flwr.common import parameters_to_ndarrays, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

from ultralytics import YOLO

from FedYOLO.train.server_utils import save_model_checkpoint
from FedYOLO.config import SPLITS_CONFIG, HOME

class BaseSaveStrategy(fl.server.strategy.FedAvg):
    """Base class for custom FL strategies to save aggregated YOLO model checkpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = f"{HOME}/yolo11n_{SPLITS_CONFIG['dataset_name']}.yaml"

    def load_and_update_model(self, aggregated_parameters: Parameters, update_head_only: bool = False) -> YOLO:
        """Load YOLO model and update weights with aggregated parameters."""
        net = YOLO(self.model_path)
        aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
        params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
        
        if update_head_only:
            detection_weights = {k: torch.tensor(v) for k, v in params_dict if k.startswith('model.detect')}
        else:
            detection_weights = {k: torch.tensor(v) for k, v in params_dict}
        
        state_dict = OrderedDict(detection_weights)
        net.load_state_dict(state_dict, strict=False)
        return net

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint."""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")
            net = self.load_and_update_model(aggregated_parameters, update_head_only=self.update_head_only)
            save_model_checkpoint(server_round, model=net.model)

        return aggregated_parameters, aggregated_metrics


class FedAvg(BaseSaveStrategy):
    """Custom FL strategy to save aggregated YOLO model checkpoints."""
    update_head_only = False


class FedMedian(BaseSaveStrategy, fl.server.strategy.FedMedian):
    """Custom FL strategy to save aggregated YOLO model checkpoints."""
    update_head_only = False


class FedHeadAvg(BaseSaveStrategy):
    """Custom FL strategy to save head-only aggregated YOLO model checkpoints."""
    update_head_only = True


class FedHeadMedian(BaseSaveStrategy, fl.server.strategy.FedMedian):
    """Custom FL strategy to save head-only aggregated YOLO model checkpoints."""
    update_head_only = True