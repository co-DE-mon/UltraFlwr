import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, help="Path to data")
args = parser.parse_args()

model = YOLO()

results = model.train(data=args.data, batch=8)

metrics = model.val(data=args.data)
