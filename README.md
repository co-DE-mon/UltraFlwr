# FedYOLO
Running baseline for federated YOLOv8.

## Create Env

```bash
conda create -n <ENV_NAME> python=3.10
conda activate <ENV_NAME>
python -m pip install flwr
pip install ultralytics
```

## Project Layout

```
FedYOLO
└── client_0_assets
|   ├── yolo_client_0.py
|   ├── yolov8n_0.pt
|   ├── dummy_data_0
|       ├── data.yaml
|       └── ...
|   └── dummy_data_1
└── client_1_assets
|   ├── yolo_client_1.py
|   ├── yolov8n_1.pt
|   ├── dummy_data_2
|   └── dummy_data_3
└── server_assets
    └── yolo_server.py
```

## Running Scripts

Always start the server first:

```bash
python3 server_assets/yolo_server.py --rounds 2 --min_num_clients 2 --sample_fraction 1.0
```

Then start the clients:

```bash
python3 client_0_assets/yolo_client_0.py --server_address=0.0.0.0:8080 --cid=0
python3 client_1_assets/yolo_client_1.py --server_address=0.0.0.0:8080 --cid=1
```

## Note

When running the scripts, if you encounter the following error, stating dataset images not found:

```bash
Note dataset download directory is '<SOME/DIRECTORY>'. You can update this in '<SOME/FILE.json>'
```

Please update the `datasets_dir` field in the `.json` file to the full path to the root directory of this repo. Such as:

```bash
"datasets_dir": "/home/<USERNAME>/Documents/GitHub/FedYOLO",
```

## The Research Hat

- A open-source implementation of FedYOLO and data partitioner in Flower
- Minimise the impact of non-IID data (need reading)
- Cross-domain detection (Simulated -> Phantom)
- Deployment in OR in real-time
