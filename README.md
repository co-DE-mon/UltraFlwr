# UltraFlwr: Federated Object Detection 
This repository contains code to all [Ultralytics](https://github.com/Ultralytics/Ultralytics) YOLO models off-the-shelf within the [Flower](https://github.com/adap/flower) framework.

Developed by [Yang-Li86](https://github.com/Yang-Li86) and [aymuos15](https://aymuos15.github.io/) :D

--------

Inspiration from existing issues:

1. Exact Ask in Ultralytics Library | [Issue](https://github.com/orgs/Ultralytics/discussions/9440)
2. Problem of loading the YOLO state | dict [Issue](https://github.com/Ultralytics/Ultralytics/issues/8804) 
    - (Similar issue raised by us: [Issue](https://github.com/Ultralytics/Ultralytics/issues/18097))
3. Need to easily integrate flower strategies with Ultralytics | [Issue](https://github.com/Ultralytics/Ultralytics/issues/14535) 
4. Request from mmlab support in flower indicates a want from the community to be able to do federated object detection | [Issue](https://github.com/adap/flower/issues/4521)

Inspiration from actual need:

5. Ultralytics allows the easy change of heads (during inference) for multiple tasks.
6. The Ultralytics style datasets are also well supported for easy off-the-shelf testing (and coco benchmarking).
7. Allow flower strategies become smoothlessly integrated with Ultralytics' YOLO.
8. Create detection specifc partial aggregation stratigies. Our initial proposal: **YOLO-PA**

## Usage Guide with [Pills Data-set](https://universe.roboflow.com/roboflow-100/pills-sxdht)

Check `FedYOLO/config.py` to see the default configurations 

1. (Highly Recommended) Make a custom environment: `python -m venv ultraflwr`
2. Clone the repository: `git clone https://github.com/Yang-Li86/UltraFlwr.git`
3. `cd` into the repository: `cd UltraFlower`
4. pip install the requirements: `pip install -e .`

### Preparing datasets

5. `cd` into the data-sets folder: `cd datasets`
6. Make a directory for a specific data-set: `mkdir pills`
7. `cd` into the data-set folder: `cd pills`
8. Get data-set from roboflow: `curl -L "https://universe.roboflow.com/ds/ojBLb70TPf?key=XwOAnIyCjF" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip`
9. Create a directory for the client specific data-sets: `mkdir partitions`
10. Create the partitions
    - Go to the base of the clone: `cd ../../`
    - Create the splits: `python FedYOLO/data_partitioner/fed_split.py` 
        - To choose the dataset, change the `DATASET_NAME` parameter in the `FedYOLO/config.py` file

#### To build Custom Data-set
Follow the style of roboflow downloads as mentioned in above steps.

![sample_dataset](./assets/sample_dataset.png)

### Training

11. For One-off: `./scripts/run.sh`
    - For Strating Multiple Runs: `./scripts/benchmark.sh`
    - For normal YOLO training on entire server dataset: `python scripts/scripts/benchmark_yolo_normal_setting.py `

### Testing

12. For testing and getting client-wise global and local scores: `./FedYOLO/test/test.sh`
    - This automatically prints out tables in Ultralytics style.
13. To collect tables (suitable for latex) for all scores across all global and local data and models: `python /FedYOLO/test/master_table.py`

## Baseline Tasks
- [x] Proper Federated Training using off-the-shelf flower strategies.
- [x] Inference Code for Local and Global data-sets using client models.
- [x] Inference Code for Local and Global data-sets using server model.
- [x] Fast Prototyping through simple bash script launch and logging.
- [x] Propose new custom strategy in the flwr framework. Our proposal: YOLO-PA
- [x] Dynamically adapt entire code base to any number of clients and not rely on manually changing code base.

## To-Dos
- [ ] Develop scripts more sophisticated/adaptable data splits.
