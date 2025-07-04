# UltraFlwr: Federated Object Detection 
UltraFlwr provides random and equally sized YOLO compatible data partitioning, federated training, and flexible testing capabilities. It integrates [Ultralytics](https://github.com/Ultralytics/Ultralytics) YOLO off-the-shelf within the [Flower](https://github.com/adap/flower) framework.

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

Comprehensive benchmarks are included in the [benchmarks](benchmarks) folder.

## Usage (Training)

We provide usage guides using [pills dataset](https://universe.roboflow.com/roboflow-100/pills-sxdht) under these settings:

1. [Single machine simulation using Python virtual environment](docs/local_venv.md)
2. [Single machine simulation using Docker](docs/local_docker.md)

A deployment demo video is shown below, click to watch it!

[![Watch the video](https://img.youtube.com/vi/raHjqcyYcBs/0.jpg)](https://www.youtube.com/watch?v=raHjqcyYcBs)

## Usage (Testing)

For testing and getting client-wise global and local scores: `./FedYOLO/test/test.sh`
- This automatically prints out tables in Ultralytics style.

To collect tables (suitable for latex) for all scores across all global and local data and models: `python /FedYOLO/test/master_table.py`

## Baseline Tasks
- [x] Proper Federated Training using off-the-shelf Flower strategies.
- [x] Inference Code for Local and Global datasets using client models.
- [x] Inference Code for Local and Global datasets using server model.
- [x] Fast Prototyping through simple bash script launch and logging.
- [x] Propose new custom strategy in the Flwr framework. Our proposal: YOLO-PA
- [x] Dynamically adapt entire code base to any number of clients and not rely on manually changing code base.

## To-Dos
- [ ] Develop scripts more sophisticated/adaptable data splits.

## Contribution Guideline
We are working on formulating rules but feel free to raise issues and PRs.

## Citation

Please cite the following work if you use our code: 

```
@article{li2025ultraflwr,
  title={UltraFlwr--An Efficient Federated Medical and Surgical Object Detection Framework},
  author={Li, Yang and Kundu, Soumya Snigdha and Boels, Maxence and Mahmoodi, Toktam and Ourselin, Sebastien and Vercauteren, Tom and Dasgupta, Prokar and Shapey, Jonathan and Granados, Alejandro},
  journal={arXiv preprint arXiv:2503.15161},
  year={2025}
}
```
