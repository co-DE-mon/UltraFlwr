# m2cai Benchmarks

Table 1. Client 0 Results: Models are client-side models trained with FL using FedAvg, FedMedian, and our YOLO-PA strategies. They are evaluated on the test split of client 0 data partition.

| Strategy              | Precision | Recall   | mAP50    | mAP50-95 |
| --------------------- | --------- | -------- | -------- | -------- |
| FedAvg                | 0.909998  | 0.867596 | 0.921152 | 0.593580 |
| FedBackboneAvg        | 0.941696  | 0.840179 | 0.920046 | 0.588518 |
| FedNeckAvg            | 0.923654  | 0.862818 | 0.920600 | 0.578222 |
| FedHeadAvg            | 0.917678  | 0.756270 | 0.863326 | 0.546805 |
| FedNeckHeadAvg        | 0.851178  | 0.805271 | 0.848999 | 0.550404 |
| FedBackboneHeadAvg    | 0.916293  | 0.874296 | 0.929095 | 0.595477 |
| FedBackboneNeckAvg    | 0.951910  | 0.870933 | 0.935262 | 0.605878 |
| FedMedian             | 0.961613  | 0.857902 | 0.928927 | 0.597616 |
| FedBackboneMedian     | 0.738272  | 0.812384 | 0.785613 | 0.459211 |
| FedNeckMedian         | 0.657941  | 0.559808 | 0.624793 | 0.307048 |
| FedHeadMedian         | 0.961996  | 0.842291 | 0.916869 | 0.572493 |
| FedNeckHeadMedian     | 0.932255  | 0.852845 | 0.922770 | 0.588437 |
| FedBackboneHeadMedian | 0.919781  | 0.914226 | 0.942652 | 0.595907 |
| FedBackboneNeckMedian | 0.912086  | 0.890253 | 0.942151 | 0.589349 |



Table 2. Client 1 Results: Models are client-side models trained with FL using FedAvg, FedMedian, and our YOLO-PA strategies. They are evaluated on the test split of client 1 data partition.

| Strategy              | Precision | Recall   | mAP50    | mAP50-95 |
| --------------------- | --------- | -------- | -------- | -------- |
| FedAvg                | 0.927739  | 0.860144 | 0.904919 | 0.562492 |
| FedBackboneAvg        | 0.910199  | 0.814715 | 0.878648 | 0.534324 |
| FedNeckAvg            | 0.891645  | 0.838169 | 0.884384 | 0.511650 |
| FedHeadAvg            | 0.923722  | 0.832817 | 0.897231 | 0.507505 |
| FedNeckHeadAvg        | 0.920539  | 0.819848 | 0.871784 | 0.523050 |
| FedBackboneHeadAvg    | 0.892313  | 0.879699 | 0.913557 | 0.553811 |
| FedBackboneNeckAvg    | 0.938050  | 0.877665 | 0.915550 | 0.545659 |
| FedMedian             | 0.940446  | 0.835153 | 0.906835 | 0.539717 |
| FedBackboneMedian     | 0.861136  | 0.869036 | 0.920052 | 0.532586 |
| FedNeckMedian         | 0.922859  | 0.819646 | 0.919105 | 0.508062 |
| FedHeadMedian         | 0.854615  | 0.844093 | 0.883564 | 0.520424 |
| FedNeckHeadMedian     | 0.945300  | 0.799227 | 0.896843 | 0.536235 |
| FedBackboneHeadMedian | 0.949054  | 0.875188 | 0.936616 | 0.564371 |
| FedBackboneNeckMedian | 0.877478  | 0.886497 | 0.922764 | 0.559496 |



Table 3. Client 2 Results: Models are client-side models trained with FL using FedAvg, FedMedian, and our YOLO-PA strategies. They are evaluated on the test split of client 2 data partition.

| Strategy              | Precision | Recall   | mAP50    | mAP50-95 |
| --------------------- | --------- | -------- | -------- | -------- |
| FedAvg                | 0.919511  | 0.889783 | 0.948495 | 0.593204 |
| FedBackboneAvg        | 0.940078  | 0.923731 | 0.958623 | 0.597342 |
| FedNeckAvg            | 0.897398  | 0.914295 | 0.948108 | 0.599396 |
| FedHeadAvg            | 0.920315  | 0.859463 | 0.931894 | 0.539246 |
| FedNeckHeadAvg        | 0.907510  | 0.902073 | 0.940090 | 0.578465 |
| FedBackboneHeadAvg    | 0.906081  | 0.928365 | 0.951915 | 0.588570 |
| FedBackboneNeckAvg    | 0.905806  | 0.935630 | 0.954696 | 0.609719 |
| FedMedian             | 0.944392  | 0.913533 | 0.960658 | 0.613109 |
| FedBackboneMedian     | 0.960505  | 0.897152 | 0.954147 | 0.610119 |
| FedNeckMedian         | 0.844109  | 0.693211 | 0.821567 | 0.460991 |
| FedHeadMedian         | 0.942191  | 0.874805 | 0.939109 | 0.568963 |
| FedNeckHeadMedian     | 0.925078  | 0.908564 | 0.946967 | 0.581127 |
| FedBackboneHeadMedian | 0.940060  | 0.929443 | 0.957733 | 0.579589 |
| FedBackboneNeckMedian | 0.945484  | 0.916287 | 0.959154 | 0.598151 |



Table 4. Server Results: Server models evaluated on client 0 test data
| Model                                    | Precision | Recall  | mAP50   | mAP50-95 |
|------------------------------------------|-----------|---------|---------|---------|
| Server FedAvg                 | 0.925092  | 0.916779 | 0.952448 | 0.613323 |
| Server FedMedian          | 0.912367  | 0.865742 | 0.921694 | 0.571779 |


Table 5. Server Results: Server models evaluated on client 1 test data
| Model                                    | Precision | Recall  | mAP50   | mAP50-95 |
|------------------------------------------|-----------|---------|---------|---------|
| Server FedMedian              | 0.959555  | 0.855646 | 0.911382 | 0.544038 |
| Server FedAvg                | 0.962412  | 0.858136 | 0.927283 | 0.572605 |

Table 6. Server Results: Server models evaluated on client 2 test data
| Model                                    | Precision | Recall  | mAP50   | mAP50-95 |
|------------------------------------------|-----------|---------|---------|---------|
| Server FedAvg                  | 0.952714  | 0.893077 | 0.940282 | 0.601530 |
| Server FedMedian              | 0.929704  | 0.945285 | 0.961993 | 0.618636 |