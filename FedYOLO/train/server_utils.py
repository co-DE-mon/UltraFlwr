import torch 

from datetime import datetime

from ultralytics.utils import __version__

from FedYOLO.config import SERVER_CONFIG, SPLITS_CONFIG, HOME

def save_model_checkpoint(server_round: int, model=None) -> None:
    """Save model training checkpoints with additional metadata."""
    checkpoint = {
        "epoch": 0,
        "best_fitness": 0,
        "model": model,  # Save the entire model, not just state_dict
        "ema": None,
        "updates": 0,
        "optimizer": None,
        "train_args": {},
        "train_metrics": {},
        "train_results": {},
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": __version__,
        "license": "AGPL-3.0 (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
    }
    
    ckpt_path = f"{HOME}/weights/model_round_{server_round}_{SPLITS_CONFIG['dataset_name']}_Strategy_{SERVER_CONFIG['strategy']}.pt"
    torch.save(checkpoint, ckpt_path)#! For now, we do not deal with this. Very difficult to log checkpoint across systems.

def write_yolo_config(dataset_name, num_classes=None):
    base_yaml = f"{HOME}/FedYOLO/train/yolov11.yaml"

    with open(base_yaml, "r") as file:
        base_yaml_content = file.read()
      
    if num_classes is not None:
        base_yaml_content = base_yaml_content.replace("nc: 80", f"nc: {num_classes}")

    filename = f"{HOME}/FedYOLO/yolo_configs/yolo11n_{dataset_name}.yaml"
    with open(filename, "w") as file:
        file.write(base_yaml_content)
    
    print(f"YAML configuration file '{filename}' has been created.")
