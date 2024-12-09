from ultralytics import YOLO

import pandas as pd

from FedYOLO.config import DATA_YAML

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data=DATA_YAML,  # path to dataset YAML
    epochs=1,  # number of training epochs
    imgsz=640,  # training image size
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
model_metrics = model.val()

metrics_table = pd.DataFrame({
    'Class': list(model_metrics.names.values()),
    'mAP@0.5:0.95': model_metrics.box.maps.tolist()
})

print(metrics_table)