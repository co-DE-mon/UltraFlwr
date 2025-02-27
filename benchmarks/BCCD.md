# BCCD Benchmarks

Table 1. Client 0 Results: Models are client-side models trained with FL using FedAvg, FedMedian, and our YOLO-PA strategies. They are evaluated on the test split of client 0 data partition.

| Strategy              | Precision | Recall   | mAP50    | mAP50-95 |
| --------------------- | --------- | -------- | -------- | -------- |
| FedAvg                | 0.761784  | 0.865658 | 0.826452 | 0.545225 |
| FedBackboneAvg        | 0.789151  | 0.869418 | 0.809367 | 0.508151 |
| FedNeckAvg            | 0.830585  | 0.803097 | 0.824176 | 0.507456 |
| FedHeadAvg            | 0.844142  | 0.806578 | 0.843425 | 0.546554 |
| FedNeckHeadAvg        | 0.800573  | 0.886091 | 0.847051 | 0.549952 |
| FedBackboneHeadAvg    | 0.815800  | 0.841861 | 0.850693 | 0.539845 |
| FedBackboneNeckAvg    | 0.813590  | 0.823074 | 0.841593 | 0.552806 |
| FedMedian             | 0.818896  | 0.824055 | 0.819358 | 0.544958 |
| FedBackboneMedian     | 0.803589  | 0.835679 | 0.831968 | 0.531088 |
| FedNeckMedian         | 0.836883  | 0.839723 | 0.837638 | 0.535185 |
| FedHeadMedian         | 0.833399  | 0.842641 | 0.841661 | 0.520734 |
| FedNeckHeadMedian     | 0.821522  | 0.854615 | 0.842101 | 0.539765 |
| FedBackboneHeadMedian | 0.798655  | 0.846128 | 0.797034 | 0.526568 |
| FedBackboneNeckMedian | 0.836053  | 0.842280 | 0.827996 | 0.543761 |




Table 2. Client 1 Results: Models are client-side models trained with FL using FedAvg, FedMedian, and our YOLO-PA strategies. They are evaluated on the test split of client 1 data partition.

| Model                 | Precision | Recall   | mAP50    | mAP50-95 |
| --------------------- | --------- | -------- | -------- | -------- |
| FedAvg                | 0.896519  | 0.874857 | 0.938992 | 0.642139 |
| FedBackboneAvg        | 0.865148  | 0.919527 | 0.921894 | 0.635436 |
| FedNeckAvg            | 0.835919  | 0.918635 | 0.900290 | 0.623707 |
| FedHeadAvg            | 0.850899  | 0.944454 | 0.927359 | 0.613435 |
| FedNeckHeadAvg        | 0.864851  | 0.888227 | 0.898387 | 0.600362 |
| FedBackboneHeadAvg    | 0.808845  | 0.963255 | 0.921109 | 0.629824 |
| FedBackboneNeckAvg    | 0.838410  | 0.886967 | 0.906020 | 0.616203 |
| FedMedian             | 0.873356  | 0.908136 | 0.928158 | 0.628665 |
| FedBackboneMedian     | 0.833399  | 0.913016 | 0.907741 | 0.620052 |
| FedNeckMedian         | 0.841952  | 0.929661 | 0.895768 | 0.601987 |
| FedHeadMedian         | 0.843578  | 0.908136 | 0.913914 | 0.629619 |
| FedNeckHeadMedian     | 0.828718  | 0.923207 | 0.919327 | 0.620477 |
| FedBackboneHeadMedian | 0.898874  | 0.873724 | 0.943760 | 0.651822 |
| FedBackboneNeckMedian | 0.869177  | 0.900262 | 0.906083 | 0.640509 |




Table 3. Client 2 Results: Models are client-side models trained with FL using FedAvg, FedMedian, and our YOLO-PA strategies. They are evaluated on the test split of client 2 data partition.

| Strategy              | Precision | Recall   | mAP50    | mAP50-95 |
| --------------------- | --------- | -------- | -------- | -------- |
| FedAvg                | 0.860370  | 0.893927 | 0.930201 | 0.638893 |
| FedBackboneAvg        | 0.891892  | 0.869575 | 0.904465 | 0.615867 |
| FedNeckAvg            | 0.859561  | 0.888686 | 0.914764 | 0.619177 |
| FedHeadAvg            | 0.841799  | 0.881173 | 0.889379 | 0.579815 |
| FedNeckHeadAvg        | 0.800591  | 0.917403 | 0.865318 | 0.601458 |
| FedBackboneHeadAvg    | 0.850357  | 0.939468 | 0.919798 | 0.629423 |
| FedBackboneNeckAvg    | 0.844003  | 0.920946 | 0.930709 | 0.650420 |
| FedMedian             | 0.794022  | 0.933719 | 0.886042 | 0.623385 |
| FedBackboneMedian     | 0.843552  | 0.882159 | 0.913897 | 0.630331 |
| FedNeckMedian         | 0.841652  | 0.899882 | 0.907607 | 0.623018 |
| FedHeadMedian         | 0.879912  | 0.845017 | 0.901691 | 0.647069 |
| FedNeckHeadMedian     | 0.945676  | 0.819615 | 0.924617 | 0.620542 |
| FedBackboneHeadMedian | 0.866151  | 0.926731 | 0.946950 | 0.656613 |
| FedBackboneNeckMedian | 0.879944  | 0.898348 | 0.925611 | 0.656142 |



Table 4. Server Results: Server models evaluated on client 0 test data

| Model            | Precision | Recall   | mAP50    | mAP50-95 |
| ---------------- | --------- | -------- | -------- | -------- |
| Server FedAvg    | 0.794261  | 0.840834 | 0.844830 | 0.564221 |
| Server FedMedian | 0.782140  | 0.819527 | 0.811777 | 0.542210 |

Table 5. Server Results: Server models evaluated on client 1 test data

| Model            | Precision | Recall   | mAP50    | mAP50-95 |
| ---------------- | --------- | -------- | -------- | -------- |
| Server FedAvg    | 0.844542  | 0.936311 | 0.933927 | 0.669218 |
| Server FedMedian | 0.865164  | 0.905512 | 0.922297 | 0.620364 |

Table 6. Server Results: Server models evaluated on client 2 test data

| Model            | Precision | Recall   | mAP50    | mAP50-95 |
| ---------------- | --------- | -------- | -------- | -------- |
| Server FedAvg    | 0.854314  | 0.918362 | 0.921988 | 0.649132 |
| Server FedMedian | 0.828181  | 0.923343 | 0.906461 | 0.636952 |