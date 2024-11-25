from ultralytics import YOLO
from collections import OrderedDict
import torch

# Load a model
# model = YOLO("yolo11n.pt")
model = YOLO()

# Train the model
train_results = model.train(
    data="/home/yang/Documents/GitHub/FedYOLO/client_0_assets/dummy_data_1/data.yaml",  # path to dataset YAML
    epochs=30,  # number of training epochs
    imgsz=640,  # training image size
    device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.model.state_dict().items()]

weights = get_parameters(model)

def set_parameters(parameters, net):
    params_dict = zip(net.model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.model.load_state_dict(state_dict, strict=True)

set_parameters(weights, model)

train_results = model.train(
    data="/home/yang/Documents/GitHub/FedYOLO/client_0_assets/dummy_data_1/data.yaml",  # path to dataset YAML
    epochs=30,  # number of training epochs
    imgsz=640,  # training image size
    device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)