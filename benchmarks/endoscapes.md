# endoscapes Benchmarks

**YOLOv11n does well when detection tools, but performance drops significantly as many detection classes are anatomical structures, causing the overall performance to degrade on endoscapes.**

Table 1. Client 0 Results: Models are client-side models trained with FL using FedAvg, FedMedian, and our YOLO-PA strategies. They are evaluated on the test split of client 0 data partition.

| Strategy              | Precision | Recall   | mAP50    | mAP50-95 |
| --------------------- | --------- | -------- | -------- | -------- |
| FedAvg                | 0.602757  | 0.516503 | 0.484247 | 0.289188 |
| FedBackboneAvg        | 0.595871  | 0.547123 | 0.494512 | 0.282241 |
| FedNeckAvg            | 0.655771  | 0.469089 | 0.507938 | 0.289291 |
| FedHeadAvg            | 0.692477  | 0.511350 | 0.543236 | 0.302905 |
| FedNeckHeadAvg        | 0.631979  | 0.446830 | 0.493152 | 0.286784 |
| FedBackboneHeadAvg    | 0.562533  | 0.504225 | 0.470178 | 0.288628 |
| FedBackboneNeckAvg    | 0.673192  | 0.441421 | 0.488798 | 0.282584 |
| FedMedian             | 0.676166  | 0.454714 | 0.512801 | 0.302663 |
| FedBackboneMedian     | 0.676166  | 0.454714 | 0.512801 | 0.302663 |
| FedNeckMedian         | 0.676166  | 0.454714 | 0.512801 | 0.302663 |
| FedHeadMedian         | 0.676166  | 0.454714 | 0.512801 | 0.302663 |
| FedNeckHeadMedian     | 0.559860  | 0.494672 | 0.460603 | 0.278616 |
| FedBackboneHeadMedian | 0.562195  | 0.466901 | 0.451238 | 0.267191 |
| FedBackboneNeckMedian | 0.562195  | 0.466901 | 0.451238 | 0.267191 |



Table 2. Client 1 Results: Models are client-side models trained with FL using FedAvg, FedMedian, and our YOLO-PA strategies. They are evaluated on the test split of client 1 data partition.

| Model                 | Precision | Recall   | mAP50    | mAP50-95 |
| --------------------- | --------- | -------- | -------- | -------- |
| FedAvg                | 0.624373  | 0.548548 | 0.563037 | 0.357277 |
| FedBackboneAvg        | 0.553745  | 0.555418 | 0.530567 | 0.351192 |
| FedNeckAvg            | 0.568981  | 0.580175 | 0.553598 | 0.356679 |
| FedHeadAvg            | 0.693481  | 0.525618 | 0.564988 | 0.361515 |
| FedNeckHeadAvg        | 0.673949  | 0.541871 | 0.551328 | 0.354318 |
| FedBackboneHeadAvg    | 0.615255  | 0.576381 | 0.556042 | 0.356470 |
| FedBackboneNeckAvg    | 0.620402  | 0.529820 | 0.553061 | 0.364018 |
| FedMedian             | 0.598132  | 0.541123 | 0.546795 | 0.362951 |
| FedBackboneMedian     | 0.598132  | 0.541123 | 0.546795 | 0.362951 |
| FedNeckMedian         | 0.598132  | 0.541123 | 0.546795 | 0.362951 |
| FedHeadMedian         | 0.598132  | 0.541123 | 0.546795 | 0.362951 |
| FedNeckHeadMedian     | 0.586908  | 0.571101 | 0.564165 | 0.358918 |
| FedBackboneHeadMedian | 0.625771  | 0.591806 | 0.581700 | 0.372619 |
| FedBackboneNeckMedian | 0.625771  | 0.591806 | 0.581700 | 0.372619 |



Table 3. Client 2 Results: Models are client-side models trained with FL using FedAvg, FedMedian, and our YOLO-PA strategies. They are evaluated on the test split of client 2 data partition.

| Model                 | Precision | Recall   | mAP50    | mAP50-95 |
| --------------------- | --------- | -------- | -------- | -------- |
| FedAvg                | 0.627565  | 0.552336 | 0.544887 | 0.332147 |
| FedBackboneAvg        | 0.624376  | 0.538275 | 0.534336 | 0.326509 |
| FedNeckAvg            | 0.545541  | 0.561614 | 0.531516 | 0.334373 |
| FedHeadAvg            | 0.671247  | 0.550257 | 0.534977 | 0.333335 |
| FedNeckHeadAvg        | 0.591778  | 0.536807 | 0.515199 | 0.335027 |
| FedBackboneHeadAvg    | 0.630081  | 0.530796 | 0.527076 | 0.328041 |
| FedBackboneNeckAvg    | 0.570677  | 0.548278 | 0.528549 | 0.330984 |
| FedMedian             | 0.681657  | 0.538202 | 0.560254 | 0.343622 |
| FedBackboneMedian     | 0.681657  | 0.538202 | 0.560254 | 0.343622 |
| FedNeckMedian         | 0.681657  | 0.538202 | 0.560254 | 0.343622 |
| FedHeadMedian         | 0.681657  | 0.538202 | 0.560254 | 0.343622 |
| FedNeckHeadMedian     | 0.637640  | 0.539447 | 0.538821 | 0.341846 |
| FedBackboneHeadMedian | 0.559352  | 0.566942 | 0.542194 | 0.331822 |
| FedBackboneNeckMedian | 0.559352  | 0.566942 | 0.542194 | 0.331822 |



Table 4. Server Results: Server models evaluated on client 0 test data

| Model                       | Precision | Recall   | mAP50    | mAP50-95 |
| --------------------------- | --------- | -------- | -------- | -------- |
| Server FedAvg     | 0.605853  | 0.477718 | 0.511679 | 0.308567 |
| Server FedMedian  | 0.672707  | 0.503576 | 0.525363 | 0.305743 |


Table 5. Server Results: Server models evaluated on client 1 test data
| Model                       | Precision | Recall   | mAP50    | mAP50-95 |
| --------------------------- | --------- | -------- | -------- | -------- |
| Server FedAvg     | 0.632505  | 0.589636 | 0.594684 | 0.382455 |
| Server FedMedian  | 0.639991  | 0.570412 | 0.576128 | 0.372062 |

Table 6. Server Results: Server models evaluated on client 2 test data
| Model                       | Precision | Recall   | mAP50    | mAP50-95 |
| --------------------------- | --------- | -------- | -------- | -------- |
| Server FedAvg     | 0.666322  | 0.571207 | 0.588171 | 0.359206 |
| Server FedMedian  | 0.602383  | 0.596650 | 0.586077 | 0.362020 |



---



Data class distribution used in the experiments:

Class Distribution for Train Split

| Class | Global Count | Client 0 Count | Client 1 Count | Client 2 Count |
| ----- | ------------ | -------------- | -------------- | -------------- |
| 0     | 433          | 199            | 104            | 130            |
| 1     | 425          | 163            | 120            | 142            |
| 2     | 636          | 217            | 215            | 204            |
| 3     | 952          | 330            | 283            | 339            |
| 4     | 1174         | 394            | 387            | 393            |
| 5     | 1946         | 625            | 643            | 678            |

Class Distribution for Validation Split

| Class | Global Count | Client 0 Count | Client 1 Count | Client 2 Count |
| ----- | ------------ | -------------- | -------------- | -------------- |
| 0     | 138          | 5              | 64             | 69             |
| 1     | 133          | 26             | 50             | 57             |
| 2     | 191          | 32             | 81             | 78             |
| 3     | 267          | 55             | 112            | 100            |
| 4     | 357          | 115            | 121            | 121            |
| 5     | 647          | 231            | 198            | 218            |

Class Distribution for Test Split

| Class | Global Count | Client 0 Count | Client 1 Count | Client 2 Count |
| ----- | ------------ | -------------- | -------------- | -------------- |
| 0     | 130          | 40             | 50             | 40             |
| 1     | 124          | 25             | 40             | 59             |
| 2     | 192          | 71             | 43             | 78             |
| 3     | 258          | 90             | 81             | 87             |
| 4     | 285          | 96             | 93             | 96             |
| 5     | 496          | 168            | 157            | 171            |

File structure:

```

```

