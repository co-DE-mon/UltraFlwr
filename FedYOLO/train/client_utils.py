import torch
from collections import OrderedDict

def parameters_to_state_dict(parameters, model):
    """
    Convert Flower parameters (a list of NumPy arrays) to a PyTorch state_dict.
    """
    param_dict = OrderedDict()
    model_keys = list(model.state_dict().keys())
    assert len(model_keys) == len(parameters), \
        f"Mismatch: model has {len(model_keys)} params, but got {len(parameters)}"

    for k, v in zip(model_keys, parameters):
        param_dict[k] = torch.tensor(v, dtype=model.state_dict()[k].dtype)

    return param_dict
