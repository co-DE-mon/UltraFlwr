# UltraFlwr: Federated Object Detection 
Official repository for *Federated Medical and Surgical Object Detection on the Edge*.

UltraFlwr provides random and equally sized YOLO compatible data partitioning, federated training, and flexible testing capabilities. It integrates [Ultralytics](https://github.com/Ultralytics/Ultralytics) YOLO off-the-shelf within the [Flower](https://github.com/adap/flower) framework.

Developed by anonymized authors.

--------

Inspiration from existing issues:

1. Exact Ask in Ultralytics Library | [Issue](https://github.com/orgs/Ultralytics/discussions/9440)
2. Problem of loading the YOLO state | dict [Issue](https://github.com/Ultralytics/Ultralytics/issues/8804) 
    - (Similar issue raised by us: [Issue](https://github.com/Ultralytics/Ultralytics/issues/18097))
3. Need to easily integrate flower strategies with Ultralytics | [Issue](https://github.com/Ultralytics/Ultralytics/issues/14535) 
4. Request from mmlab support in flower indicates a want from the community to be able to do federated object detection | [Issue](https://github.com/adap/flower/issues/4521)

Inspiration from actual need:

1. Ultralytics allows the easy change of final heads (during inference) for multiple tasks.
2. The Ultralytics style datasets are also well supported for easy off-the-shelf testing (and coco benchmarking).
3. Allow flower strategies become smoothly integrated with Ultralytics' YOLO.
4. Create detection specifc partial aggregation strategies, such as *YOLO-PA*.

## Benchmarks

Comprehensive benchmarks are included in [Benchmarks.md](Benchmarks.md).

## Usage (Training)

We provide usage guides using [pills dataset](https://universe.roboflow.com/roboflow-100/pills-sxdht) under three settings:

1. Single machine simulation
   ​	a. Using Python venv
   ​	b. Using Docker
2. Multiple edge computing devices
   ​	a. Using Docker

### Python venv

Check `FedYOLO/config.py` to see the default configurations 

1. Make a custom environment: `python -m venv ultraflwr`
2. Clone the repository
3. `cd` into the repository: `cd UltraFlwr`   
4. pip install the requirements: `pip install -e .`

#### Preparing Datasets

5. `cd` into the datasets folder: `cd datasets`
6. Make a directory for a specific dataset: `mkdir pills`
7. `cd` into the dataset folder: `cd pills`
8. Get data-set from Roboflow
9. Create a directory for the client specific datasets: `mkdir partitions`
10. Create the partitions
    - Go to the base of the clone: `cd ../../`
    - Create the splits: `python FedYOLO/data_partitioner/fed_split.py` 
        - To choose the dataset, change the `DATASET_NAME` parameter in the `FedYOLO/config.py` file

#### To Build Custom Dataset
Follow the style of roboflow downloads as mentioned in above steps.

![sample_dataset](./assets/sample_dataset.png)

#### Training

11. For one-off: `./scripts/run.sh`
    - For  Multiple Runs: `./scripts/benchmark.sh`
    - For normal YOLO training on entire server dataset: `python scripts/scripts/benchmark_yolo_normal_setting.py `

### Running on Nvidia DGX with run.ai and Docker (need to add anonymization for submission)
1. Download and partition the datasets as needed, refer to [Preparing Datasets](####preparing-datasets)
2. Navigate to the root of the project
3. Build the docker image and push it to a registry with `bash create_docker_image.sh`
4. Configure the `FedYOLO/config.py` as needed
5. We ran our experiments with `scripts/benchmark.sh`, please create your script or modify it accordingly
6. Update the `run_ai.sh` according to your needs
7. Submit job with `bash run_ai.sh`

### Running on Edge Devices with Docker
Updates coming soon.
## Usage (Testing)

12. For testing and getting client-wise global and local scores: `./FedYOLO/test/test.sh`
    - This automatically prints out tables in Ultralytics style.
13. To collect tables (suitable for latex) for all scores across all global and local data and models: `python /FedYOLO/test/master_table.py`

## Baseline Tasks
- [x] Proper Federated Training using off-the-shelf Flower strategies.
- [x] Inference Code for Local and Global datasets using client models.
- [x] Inference Code for Local and Global datasets using server model.
- [x] Fast Prototyping through simple bash script launch and logging.
- [x] Propose new custom strategy in the Flwr framework. Our proposal: YOLO-PA
- [x] Dynamically adapt entire code base to any number of clients and not rely on manually changing code base.

## To-Dos
- [ ] Develop scripts more sophisticated/adaptable data splits.
