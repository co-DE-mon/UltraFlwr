import flwr as fl
from flwr.common import ndarrays_to_parameters
from ultralytics import YOLO
from FedYOLO.config import SERVER_CONFIG
from collections import OrderedDict
import torch
from typing import Optional, Union
import numpy as np
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, Scalar

# net = YOLO()

def fit_config(server_round: int):
    return {"epochs": SERVER_CONFIG["epochs"]}

def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.model.state_dict().items()]

# class SaveModelStrategy(fl.server.strategy.FedAvg):
#     def aggregate_fit(
#         self,
#         server_round: int,
#         results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
#         failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
#     ) -> tuple[Optional[Parameters], dict[str, Scalar]]:

#         # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
#         aggregated_parameters, aggregated_metrics = super().aggregate_fit(
#             server_round, results, failures
#         )

#         if aggregated_parameters is not None:
#             # Convert `Parameters` to `list[np.ndarray]`
#             aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(
#                 aggregated_parameters
#             )

#             # # Save aggregated_ndarrays to disk
#             # print(f"Saving round {server_round} aggregated_ndarrays...")
#             # np.savez(f"round-{server_round}-weights.npz", *aggregated_ndarrays)

#             # Convert `list[np.ndarray]` to PyTorch `state_dict`
#             params_dict = zip(net.model.state_dict().keys(), aggregated_ndarrays)
#             state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#             net.model.load_state_dict(state_dict, strict=False)

#             # Save the model to disk
#             torch.save(net.model.state_dict(), f"model_round_{server_round}.pth")


#         return aggregated_parameters, aggregated_metrics

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[Union[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Save the parameters as an OrderedDict directly
            state_dict = OrderedDict({f"layer_{i}": torch.tensor(v) for i, v in enumerate(aggregated_ndarrays)})

            # Save the state_dict to a file
            torch.save(state_dict, f"model_weights_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics

# ndarrays = get_parameters(net)
# parameters = ndarrays_to_parameters(ndarrays)

def main():
    # Baseline strategy: Federated Averaging
    strategy = SaveModelStrategy(
        fraction_fit=SERVER_CONFIG["sample_fraction"],
        min_fit_clients=SERVER_CONFIG["min_num_clients"],
        on_fit_config_fn=fit_config,
        # initial_parameters=parameters,
    )

    fl.server.start_server(
        server_address=SERVER_CONFIG["server_address"],
        config=fl.server.ServerConfig(num_rounds=SERVER_CONFIG["rounds"]),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
