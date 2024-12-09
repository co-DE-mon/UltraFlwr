# UltraFlower: Federated Object Detection Using [Ultralytics](https://github.com/ultralytics/ultralytics) and [Flower](https://github.com/adap/flower)

This repository contains code to run YOLOv11 off-the-shelf within the flower framework.

Inspiration from existing issues:
1. Exact Ask in Ultralytics Library | [Issue](https://github.com/orgs/ultralytics/discussions/9440)

2. Problem of loading the YOLO state | dict [Issue](https://github.com/ultralytics/ultralytics/issues/8804) 
        
    - (Similar issue raised by us: [Issue](https://github.com/ultralytics/ultralytics/issues/18097))

3. Request from mmlab support in flower indicates a want from the community to be able to do federated object detection | [Issue](https://github.com/adap/flower/issues/4521)

## Step by Step Guide to fo End-to-End Training/Testing with the Example [Pills Dataset](https://universe.roboflow.com/roboflow-100/pills-sxdht)

Check `config.py` to see the default configurations

1. (Highly Recommended) Make a custom environment: `python -m venv fedlytics`
2. Clone the repository: `git clone https://github.com/Yang-Li86/FedDet.git`
3. `cd` into the repository: `cd FedDet`
4. pip install the requirements: `pip install -e .`
5. `cd` into the datasets folder: `cd datasets`
6. Make a directory for a specific dataset: `mkdir pills`
8. `cd` into the dataset folder: `cd pills`
9. Get dataset from roboflow: `curl -L "https://universe.roboflow.com/ds/ojBLb70TPf?key=XwOAnIyCjF" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip`
10. Create a directory for the client specific datasets: `mkdir partitions`
11. Create the partitions
    - Go to the base of the clone: `cd ../../`
    - Create the splits: python FedYOLO/data_partitioner/fed_split.py (configs) 
12. For federated training: `./scrips/run.sh`
13. For testing and getting client-wise global and local scores: `python /FedYOLO/test/test.py`
    - This automatically prints out tables in ultralytics style.

## To build Custom Dataset
Follow the style of roboflow downloads as mentioned in above steps.

## Basline Tasks
- [x] Proper Federated Training using off-the-shelf flower strategies.
- [x] Inference Code for Local and Global datasets using client models.
- [x] Fast Prototyping through simple bash script launch and logging.
- [x] Inference Code for Local and Global datasets using server model.

## To-Dos
- [ ] Make better prints for server model inference.
- [ ] Dynamically adapt entire codebase to any number of clients and not rely on manually changing codebase.
- [ ] Develop scripts more sophisticated/adaptable data splits.