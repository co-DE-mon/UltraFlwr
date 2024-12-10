# UltraFlwr: Federated Object Detection Using [Ultralytics](https://github.com/Ultralytics/Ultralytics) and [Flower](https://github.com/adap/flower)

This repository contains code to run YOLOv11 off-the-shelf within the flower framework.

Developed by [Yang-Li86](https://github.com/Yang-Li86) and [aymuos15](https://aymuos15.github.io/) :D

Inspiration from existing issues:
1. Exact Ask in Ultralytics Library | [Issue](https://github.com/orgs/Ultralytics/discussions/9440)
2. Problem of loading the YOLO state | dict [Issue](https://github.com/Ultralytics/Ultralytics/issues/8804) 
    - (Similar issue raised by us: [Issue](https://github.com/Ultralytics/Ultralytics/issues/18097))
3. There is a need to easily integrate flower strategies with Ultralytics | [Issue](https://github.com/Ultralytics/Ultralytics/issues/14535) 
4. Request from mmlab support in flower indicates a want from the community to be able to do federated object detection | [Issue](https://github.com/adap/flower/issues/4521)
5. Unlike mmlab, Ultralytics allow the easy change of heads (during inference) for multiple tasks. Therefore, there is a definite need for this to be integrated within flower as well.
6. The Ultralytics style datasets are also well supported for easy off-the-shelf testing (and coco benchmarking)
7. Another primary objective was to create detection specifc stratigies. Our proposal: FedHeadAvg

## Step by Step Guide to End-to-End Training/Testing with the Example [Pills Data-set](https://universe.roboflow.com/roboflow-100/pills-sxdht)

Check `config.py` to see the default configurations [!IMPORTANT! Before starting make sure to make the relevant changes here]

1. (Highly Recommended) Make a custom environment: `python -m venv fedlytics`
2. Clone the repository: `git clone https://github.com/Yang-Li86/FedDet.git`
3. `cd` into the repository: `cd FedDet`
4. pip install the requirements: `pip install -e .`
5. Create the `weights` and `logs` folders: `mkdir weights logs`
6. `cd` into the data-sets folder: `cd datasets`
7. Make a directory for a specific data-set: `mkdir pills`
8. `cd` into the data-set folder: `cd pills`
9. Get data-set from roboflow: `curl -L "https://universe.roboflow.com/ds/ojBLb70TPf?key=XwOAnIyCjF" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip`
10. Create a directory for the client specific data-sets: `mkdir partitions`
11. Create the partitions
    - Go to the base of the clone: `cd ../../`
    - Create the splits: `python FedYOLO/data_partitioner/fed_split.py` (configs) 
12. For federated training: `./scripts/run.sh`
13. For testing and getting client-wise global and local scores: `python /FedYOLO/test/test.py`
    - This automatically prints out tables in Ultralytics style.
14. [To collect results] Run `FedYOLO/test/test.py` to collect (our style) of tables for mAP scores across all global and local data and models.

## To build Custom Data-set
Follow the style of roboflow downloads as mentioned in above steps.

![sample_dataset](./assets/sample_dataset.png)

## Baseline Tasks
- [x] Proper Federated Training using off-the-shelf flower strategies.
- [x] Inference Code for Local and Global data-sets using client models.
- [x] Fast Prototyping through simple bash script launch and logging.
- [x] Inference Code for Local and Global data-sets using server model.
- [x] Specialised version of FedHeadAvg for YOLO. FedHeadAvg where only the detection module is being updated.
- [x] Propose new custom strategy in the flwr framework: Our proposal: FedHeadAvg

## To-Dos
- [ ] Make better prints for server model inference.
- [ ] Dynamically adapt entire code base to any number of clients and not rely on manually changing code base. [This needs discussion. Maybe not recommended in the longer run]
- [ ] Develop scripts more sophisticated/adaptable data splits.