import time
import torch
from collections import OrderedDict
from typing import Optional, Union

import flwr as fl
from flwr.common import parameters_to_ndarrays, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from ultralytics import YOLO

from FedYOLO.train.server_utils import save_model_checkpoint
from FedYOLO.config import SPLITS_CONFIG, HOME

class BaseYOLOSaveStrategy(fl.server.strategy.FedAvg):
    """Base class for custom FL strategies to save YOLO model checkpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = f"{HOME}/FedYOLO/yolo_configs/yolo11n_{SPLITS_CONFIG['dataset_name']}.yaml"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        time.sleep(30) # wait for clients to initialise
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def get_section_parameters(self, state_dict: OrderedDict) -> tuple[dict, dict, dict]:
        """Get parameters for each section of the model."""
        # Backbone parameters (early layers through conv layers)
        backbone_weights = {
            k: v for k, v in state_dict.items() 
            if not k.startswith(('model.17', 'model.20', 'model.21', 'model.22', 'model.23'))
        }
        
        # Neck parameters (SPPF and FPN layers)
        neck_weights = {
            k: v for k, v in state_dict.items() 
            if k.startswith(('model.17', 'model.20', 'model.21', 'model.22'))
        }
        
        # Head parameters (detection head)
        head_weights = {
            k: v for k, v in state_dict.items() 
            if k.startswith('model.23')
        }
        
        return backbone_weights, neck_weights, head_weights

    def load_and_update_model(self, aggregated_parameters: Parameters) -> YOLO:
        """Load YOLO model and update weights with aggregated parameters."""
        net = YOLO(self.model_path)
        aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
        params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
        
        state_dict = net.state_dict()
        backbone_weights, neck_weights, head_weights = self.get_section_parameters(state_dict)
        
        updated_weights = {}
        
        for k, v in params_dict:
            # Only update sections based on strategy configuration
            if (self.update_backbone and k in backbone_weights) or \
               (self.update_neck and k in neck_weights) or \
               (self.update_head and k in head_weights):
                updated_weights[k] = torch.tensor(v)
        
        state_dict = OrderedDict(updated_weights)
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
            net = self.load_and_update_model(aggregated_parameters)
            save_model_checkpoint(server_round, model=net.model)

        return aggregated_parameters, aggregated_metrics

# FedAvg variations
class FedAvg(BaseYOLOSaveStrategy):
    """Federated averaging of all model parameters."""
    update_backbone = True
    update_neck = True
    update_head = True

class FedHeadAvg(BaseYOLOSaveStrategy):
    """Federated averaging of detection head only."""
    update_backbone = False
    update_neck = False
    update_head = True

class FedNeckAvg(BaseYOLOSaveStrategy):
    """Federated averaging of neck (SPPF and FPN) only."""
    update_backbone = False
    update_neck = True
    update_head = False

class FedBackboneAvg(BaseYOLOSaveStrategy):
    """Federated averaging of backbone only."""
    update_backbone = True
    update_neck = False
    update_head = False

class FedNeckHeadAvg(BaseYOLOSaveStrategy):
    """Federated averaging of neck and head."""
    update_backbone = False
    update_neck = True
    update_head = True

# FedMedian variations
class FedMedian(BaseYOLOSaveStrategy, fl.server.strategy.FedMedian):
    """Federated median of all model parameters."""
    update_backbone = True
    update_neck = True
    update_head = True

class FedHeadMedian(BaseYOLOSaveStrategy, fl.server.strategy.FedMedian):
    """Federated median of detection head only."""
    update_backbone = False
    update_neck = False
    update_head = True

class FedNeckMedian(BaseYOLOSaveStrategy, fl.server.strategy.FedMedian):
    """Federated median of neck (SPPF and FPN) only."""
    update_backbone = False
    update_neck = True
    update_head = False

class FedBackboneMedian(BaseYOLOSaveStrategy, fl.server.strategy.FedMedian):
    """Federated median of backbone only."""
    update_backbone = True
    update_neck = False
    update_head = False

class FedNeckHeadMedian(BaseYOLOSaveStrategy, fl.server.strategy.FedMedian):
    """Federated median of neck and head."""
    update_backbone = False
    update_neck = True
    update_head = True