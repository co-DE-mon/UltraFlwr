# FedDet
Running baseline for federated YOLOv11.

## Create Env

```bash
conda create -n <ENV_NAME> python=3.10
conda activate <ENV_NAME>
python -m pip install flwr
pip install ultralytics
```

## Project Layout

# Running Scripts

### Download Dataset

1. `cd` into the `datasets` dir
2. Make a new folder with a dataset name. [Ex: `mkdir pills`]
3. `cd` into `pills` 
4. Download/Place the dataset. [Ex: `curl -L "https://universe.roboflow.com/ds/ojBLb70TPf?key=XwOAnIyCjF" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip`]

### Create Partitions

5. In your `pills` folder, `mkdir partitions`
6. Make the config changes according to your dataset
7. run '/home/localssk23/FedYOLO/scripts/fed_split.py'

### To Run

8. Start a New terminal
9. `python3 server/yolo_server.py`

### Start client 0

10. Start a New terminal
11. ```python
    python3 client/yolo_client.py 
    --server_address=0.0.0.0:8080 
    --cid=0 
    --data_path /home/localssk23/FedYOLO/datasets/pills/partitions/client_0/data.yaml```

### Start client 1

12. Start a New terminal
13. ```python
    python3 client/yolo_client.py 
    --server_address=0.0.0.0:8080 
    --cid=1 
    --data_path /home/localssk23/FedYOLO/datasets/pills/partitions/client_1/data.yaml``

[ Change the cid according to your client num ]

# Notes

1. If you want to build your custom data, just follow the style of roboflow downloads as mentioned in above steps.

# Questions
1. Where Do we see the runs? Do we need a nice way to understand that?
2. How to make a nice way to build a `run.sh`. The one right now is too hard to interpret.
3. Do we need a requirements.txt file?

## The Research Hat

- A open-source implementation of FedYOLO and data partitioner in Flower
- Minimise the impact of non-IID data (need reading)
- Cross-domain detection (Simulated -> Phantom)
- Deployment in OR in real-time
