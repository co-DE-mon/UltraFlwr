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

---

Data class distribution used in the experiments:

Class Distribution for Train Split

| Class | Global Count | Client 0 Count | Client 1 Count | Client 2 Count |
|-------|--------------|----------------|----------------|----------------|
|   0   |     707      |      289       |      274       |      144       |
|   1   |     215      |       16       |      111       |       88       |
|   2   |     165      |       63       |       52       |       50       |
|   3   |     197      |      110       |       41       |       46       |
|   4   |     220      |      108       |       43       |       69       |
|   5   |     228      |       31       |       80       |      117       |
|   6   |     241      |       86       |       93       |       62       |

Class Distribution for Validation Split

| Class | Global Count | Client 0 Count | Client 1 Count | Client 2 Count |
|-------|--------------|----------------|----------------|----------------|
|   0   |     422      |      169       |      170       |       83       |
|   1   |     140      |       20       |       85       |       35       |
|   2   |      79      |       32       |       20       |       27       |
|   3   |     107      |       60       |       20       |       27       |
|   4   |     116      |       63       |       24       |       29       |
|   5   |     173      |       17       |       51       |      105       |
|   6   |     139      |       48       |       54       |       37       |

Class Distribution for Test Split

| Class | Global Count | Client 0 Count | Client 1 Count | Client 2 Count |
|-------|--------------|----------------|----------------|----------------|
|   0   |     293      |      116       |      125       |       52       |
|   1   |      95      |       18       |       43       |       34       |
|   2   |      64      |       32       |       19       |       13       |
|   3   |      84      |       40       |       17       |       27       |
|   4   |      64      |       23       |       12       |       29       |
|   5   |      84      |       13       |       25       |       46       |
|   6   |      96      |       33       |       42       |       21       |




File structure:
```
partitions % tree
.
├── client_0
│   ├── data.yaml
│   ├── test
│   │   ├── images
│   │   │   ├── BloodImage_00038_jpg.rf.63d04b5c9db95f32fa7669f72e4903ca.jpg
│   │   │   ├── BloodImage_00044_jpg.rf.589ee3d351cb6d9a3f7b7a942da5370a.jpg
│   │   │   ├── BloodImage_00062_jpg.rf.1cecc20a21ac39cb54cf532081a1e893.jpg
│   │   │   ├── BloodImage_00090_jpg.rf.5267690cb6a13608d39b0424bef3c9b4.jpg
│   │   │   ├── BloodImage_00099_jpg.rf.744666666386a07e242e214e041945ba.jpg
│   │   │   ├── BloodImage_00112_jpg.rf.e4b507506c4a70882bb23cb743061a66.jpg
│   │   │   ├── BloodImage_00113_jpg.rf.250f3f0288ad89f4f961529434d99713.jpg
│   │   │   ├── BloodImage_00120_jpg.rf.01566ae891eda007b18994a74255367c.jpg
│   │   │   ├── BloodImage_00133_jpg.rf.4f9b4435c673ed96c9deeb985c805d24.jpg
│   │   │   ├── BloodImage_00134_jpg.rf.f026a98ce4257048a617a01c70aac485.jpg
│   │   │   ├── BloodImage_00154_jpg.rf.c2a4d782d9505a7e89d4a71cdb38461e.jpg
│   │   │   └── BloodImage_00160_jpg.rf.63a1db217fa927dad3f2d2488b5e7862.jpg
│   │   ├── labels
│   │   │   ├── BloodImage_00038_jpg.rf.63d04b5c9db95f32fa7669f72e4903ca.txt
│   │   │   ├── BloodImage_00044_jpg.rf.589ee3d351cb6d9a3f7b7a942da5370a.txt
│   │   │   ├── BloodImage_00062_jpg.rf.1cecc20a21ac39cb54cf532081a1e893.txt
│   │   │   ├── BloodImage_00090_jpg.rf.5267690cb6a13608d39b0424bef3c9b4.txt
│   │   │   ├── BloodImage_00099_jpg.rf.744666666386a07e242e214e041945ba.txt
│   │   │   ├── BloodImage_00112_jpg.rf.e4b507506c4a70882bb23cb743061a66.txt
│   │   │   ├── BloodImage_00113_jpg.rf.250f3f0288ad89f4f961529434d99713.txt
│   │   │   ├── BloodImage_00120_jpg.rf.01566ae891eda007b18994a74255367c.txt
│   │   │   ├── BloodImage_00133_jpg.rf.4f9b4435c673ed96c9deeb985c805d24.txt
│   │   │   ├── BloodImage_00134_jpg.rf.f026a98ce4257048a617a01c70aac485.txt
│   │   │   ├── BloodImage_00154_jpg.rf.c2a4d782d9505a7e89d4a71cdb38461e.txt
│   │   │   └── BloodImage_00160_jpg.rf.63a1db217fa927dad3f2d2488b5e7862.txt
│   │   └── labels.cache
│   ├── train
│   │   ├── images
│   │   │   ├── BloodImage_00001_jpg.rf.d702f2b1212a2ed897b5607804109acf.jpg
│   │   │   ├── BloodImage_00002_jpg.rf.c10bc2092c5ffbbb18f5d8db2e5a00b7.jpg
│   │   │   ├── BloodImage_00003_jpg.rf.4c25ef46042b85efe0700671af1fba87.jpg
│   │   │   ├── BloodImage_00005_jpg.rf.5b7970d683a751cab17ed07b190488c3.jpg
│   │   │   ├── BloodImage_00006_jpg.rf.5843ee9ebfa219fd22679db5ecee8035.jpg
│   │   │   ├── BloodImage_00007_jpg.rf.7f3cae9502e84fc765fadf5c2c003c14.jpg
│   │   │   ├── BloodImage_00008_jpg.rf.1c8b2fef1372ccd864dc23b6fe935a8e.jpg
│   │   │   ├── BloodImage_00009_jpg.rf.0c18d19d1e59d1fe3e53f7a18380cc6b.jpg
│   │   │   ├── BloodImage_00010_jpg.rf.720639748c66ecb094ef9e4f0413f4e1.jpg
│   │   │   ├── BloodImage_00011_jpg.rf.36cb18cf5e06d50c880aa1485bf91e51.jpg
│   │   │   ├── BloodImage_00013_jpg.rf.2d36bc6975822a7799b350b580a946c2.jpg
│   │   │   ├── BloodImage_00014_jpg.rf.c079cb4d4b27bd90cf2b456109d906c5.jpg
│   │   │   ├── BloodImage_00015_jpg.rf.ccb967d23f035411d36b337cb8d7fe93.jpg
│   │   │   ├── BloodImage_00016_jpg.rf.04c8657ee6b0eb3fd19c804baa1e27c0.jpg
│   │   │   ├── BloodImage_00018_jpg.rf.2bd85354fc1b2e42446be452727a71a0.jpg
│   │   │   ├── BloodImage_00019_jpg.rf.e0d51910e182dca5b4e6afae15cf35a7.jpg
│   │   │   ├── BloodImage_00020_jpg.rf.5be1260f16e85f6d3b3f940a38cd7392.jpg
│   │   │   ├── BloodImage_00022_jpg.rf.acc5dc23d9f1e5c53981a2eda2cd9827.jpg
│   │   │   ├── BloodImage_00023_jpg.rf.6dea1d883100d129b1b7ded9add2c24f.jpg
│   │   │   ├── BloodImage_00024_jpg.rf.323c51bbfa65bf029bef80b250107268.jpg
│   │   │   ├── BloodImage_00028_jpg.rf.1a054fd7ba2ad7f7b3e3245212a14792.jpg
│   │   │   ├── BloodImage_00029_jpg.rf.9d8e026ce7b048d2d094f2f7f229f368.jpg
│   │   │   ├── BloodImage_00030_jpg.rf.71385416b4e288babd3e52de3941fc6b.jpg
│   │   │   ├── BloodImage_00031_jpg.rf.bcce96daec2fe43241ed711e793cb3d4.jpg
│   │   │   ├── BloodImage_00032_jpg.rf.99717e8579fd0c8d1fb28dfb5e96dae0.jpg
│   │   │   ├── BloodImage_00033_jpg.rf.ed29a91d8384959065269d3584e4acb7.jpg
│   │   │   ├── BloodImage_00034_jpg.rf.2ffb51d919ecc92d369cffbfe88f41c3.jpg
│   │   │   ├── BloodImage_00035_jpg.rf.ac506be1c23d0f3a09d8e340a008f885.jpg
│   │   │   ├── BloodImage_00036_jpg.rf.f9b3f1851acea0c511bdbb05aa0e4200.jpg
│   │   │   ├── BloodImage_00037_jpg.rf.f10b2a6625522717d70040e245a43103.jpg
│   │   │   ├── BloodImage_00039_jpg.rf.e52773eb869d63407dac92ddc09aa183.jpg
│   │   │   ├── BloodImage_00040_jpg.rf.954258c34647369f74e3a37e1e675acb.jpg
│   │   │   ├── BloodImage_00041_jpg.rf.80dc89cfe56e555cef5aa8742dec9ed7.jpg
│   │   │   ├── BloodImage_00042_jpg.rf.e0e7f2299512850aa9e1496cc40b35a0.jpg
│   │   │   ├── BloodImage_00043_jpg.rf.f11aab205044a15d8585d565a04bec19.jpg
│   │   │   ├── BloodImage_00045_jpg.rf.293583cdffcb84f3c126ad5fb0a5c10e.jpg
│   │   │   ├── BloodImage_00046_jpg.rf.fc4a2bb5179e91b09d0667faf7e6dfa3.jpg
│   │   │   ├── BloodImage_00047_jpg.rf.8398043d05fecb77cf4e650d2208fb0d.jpg
│   │   │   ├── BloodImage_00048_jpg.rf.728c77ec6eebf109593bfdb9597ea65f.jpg
│   │   │   ├── BloodImage_00049_jpg.rf.d34eff6d6dde615eeae0515a9f94ea88.jpg
│   │   │   ├── BloodImage_00052_jpg.rf.8b1cb0fdfb126b5629795cd4c7dca0b2.jpg
│   │   │   ├── BloodImage_00053_jpg.rf.d7ae83eb982e69dfb0c48b5cb4437e55.jpg
│   │   │   ├── BloodImage_00054_jpg.rf.e0701bbac7b3bf5dc3ea26f45b445e2c.jpg
│   │   │   ├── BloodImage_00055_jpg.rf.b47dffc3486b0b7c7c1a067421672d51.jpg
│   │   │   ├── BloodImage_00056_jpg.rf.319effd6211d38e7645bfa787d3c2dab.jpg
│   │   │   ├── BloodImage_00058_jpg.rf.16d6f2ede839c8876e3339c91f4e73ef.jpg
│   │   │   ├── BloodImage_00059_jpg.rf.ef31566ba8899bf8358f53c7ce29063c.jpg
│   │   │   ├── BloodImage_00064_jpg.rf.bf19df7bdde5d68567661f7a9d2d586e.jpg
│   │   │   ├── BloodImage_00065_jpg.rf.ddc13d1753ba439a75be985268c15813.jpg
│   │   │   ├── BloodImage_00066_jpg.rf.26aa8e7bf8c8e2601b56d24d5d8a61fb.jpg
│   │   │   ├── BloodImage_00067_jpg.rf.244c43eed192e0dcd27ba9bd92b9b18e.jpg
│   │   │   ├── BloodImage_00068_jpg.rf.cea435e947d1e8661d14f90e0cc71f6a.jpg
│   │   │   ├── BloodImage_00072_jpg.rf.4dea0e09c770a4de243d19921135ce15.jpg
│   │   │   ├── BloodImage_00073_jpg.rf.195c2eb474e50e3707a12cb06739508b.jpg
│   │   │   ├── BloodImage_00074_jpg.rf.d76fb021c6e3ea3606b74b9f38a982da.jpg
│   │   │   ├── BloodImage_00077_jpg.rf.ea1ac390cf353b6825fe2f86375db707.jpg
│   │   │   ├── BloodImage_00078_jpg.rf.4f5978ebeafc09a920f9407f8acd2d5a.jpg
│   │   │   ├── BloodImage_00079_jpg.rf.2a2c70686066bf5f4634e747dc90dd58.jpg
│   │   │   ├── BloodImage_00081_jpg.rf.229ebb7e6402a2ac724a5a662d5b7e99.jpg
│   │   │   ├── BloodImage_00082_jpg.rf.66e0043d7cda781d565786264330eb55.jpg
│   │   │   ├── BloodImage_00083_jpg.rf.0307243d1591caafc332240f3e4e39b6.jpg
│   │   │   ├── BloodImage_00087_jpg.rf.4ed67af94b2d74d0b6db05f10d1fbf57.jpg
│   │   │   ├── BloodImage_00088_jpg.rf.19b4ccde27cd787d761ac913158ea7f8.jpg
│   │   │   ├── BloodImage_00089_jpg.rf.7277987610ce4b8057fc2255ec1deceb.jpg
│   │   │   ├── BloodImage_00091_jpg.rf.6d67a1893a8efb19a33afa4b62e87e93.jpg
│   │   │   ├── BloodImage_00094_jpg.rf.9eee88888a92b5208516e8b880cf46b6.jpg
│   │   │   ├── BloodImage_00095_jpg.rf.16971a175ba68f23bebe78d5075d540b.jpg
│   │   │   ├── BloodImage_00097_jpg.rf.c38c2cbdc33259648be15cfd9a43b2b7.jpg
│   │   │   ├── BloodImage_00098_jpg.rf.065b4cca0dc147f1f2768e62f90d059f.jpg
│   │   │   ├── BloodImage_00100_jpg.rf.73c6e7f83c1a68208d5c08d359038a8f.jpg
│   │   │   ├── BloodImage_00101_jpg.rf.6c604197453717b53def341cb276878a.jpg
│   │   │   ├── BloodImage_00103_jpg.rf.9907b3dbc5f0c74428d3c7971b644222.jpg
│   │   │   ├── BloodImage_00106_jpg.rf.fd43315c4bf62f68262040e9253e4a99.jpg
│   │   │   ├── BloodImage_00111_jpg.rf.2177a3168c80354610c6a7c996981595.jpg
│   │   │   ├── BloodImage_00114_jpg.rf.8907da31dbf9aac7107d884322074dcf.jpg
│   │   │   ├── BloodImage_00115_jpg.rf.ace3f09def7d9e9a78c5891273f3d9fc.jpg
│   │   │   ├── BloodImage_00117_jpg.rf.b35d5c3f818b6732815da9fb1a3bffa1.jpg
│   │   │   ├── BloodImage_00124_jpg.rf.f6f4e0138db2ebd88cce18a056627c2d.jpg
│   │   │   ├── BloodImage_00125_jpg.rf.2eecb21bba4ce18441b6228e460fc7a8.jpg
│   │   │   ├── BloodImage_00132_jpg.rf.209495d7f38eb9b80bc009ce3a9b5f96.jpg
│   │   │   ├── BloodImage_00136_jpg.rf.a86a515dfbe0ce2bd57cc87fed8ea862.jpg
│   │   │   ├── BloodImage_00137_jpg.rf.264ee70ffac6681843fa4dbcfb36b11f.jpg
│   │   │   ├── BloodImage_00139_jpg.rf.5e4411dde64c7638efc96e37c67ff2cb.jpg
│   │   │   ├── BloodImage_00140_jpg.rf.e36558fb41aa42903ede72a60b51a73d.jpg
│   │   │   └── BloodImage_00142_jpg.rf.4dc589baf76059f3d5b11226b77cdd43.jpg
│   │   ├── labels
│   │   │   ├── BloodImage_00001_jpg.rf.d702f2b1212a2ed897b5607804109acf.txt
│   │   │   ├── BloodImage_00002_jpg.rf.c10bc2092c5ffbbb18f5d8db2e5a00b7.txt
│   │   │   ├── BloodImage_00003_jpg.rf.4c25ef46042b85efe0700671af1fba87.txt
│   │   │   ├── BloodImage_00005_jpg.rf.5b7970d683a751cab17ed07b190488c3.txt
│   │   │   ├── BloodImage_00006_jpg.rf.5843ee9ebfa219fd22679db5ecee8035.txt
│   │   │   ├── BloodImage_00007_jpg.rf.7f3cae9502e84fc765fadf5c2c003c14.txt
│   │   │   ├── BloodImage_00008_jpg.rf.1c8b2fef1372ccd864dc23b6fe935a8e.txt
│   │   │   ├── BloodImage_00009_jpg.rf.0c18d19d1e59d1fe3e53f7a18380cc6b.txt
│   │   │   ├── BloodImage_00010_jpg.rf.720639748c66ecb094ef9e4f0413f4e1.txt
│   │   │   ├── BloodImage_00011_jpg.rf.36cb18cf5e06d50c880aa1485bf91e51.txt
│   │   │   ├── BloodImage_00013_jpg.rf.2d36bc6975822a7799b350b580a946c2.txt
│   │   │   ├── BloodImage_00014_jpg.rf.c079cb4d4b27bd90cf2b456109d906c5.txt
│   │   │   ├── BloodImage_00015_jpg.rf.ccb967d23f035411d36b337cb8d7fe93.txt
│   │   │   ├── BloodImage_00016_jpg.rf.04c8657ee6b0eb3fd19c804baa1e27c0.txt
│   │   │   ├── BloodImage_00018_jpg.rf.2bd85354fc1b2e42446be452727a71a0.txt
│   │   │   ├── BloodImage_00019_jpg.rf.e0d51910e182dca5b4e6afae15cf35a7.txt
│   │   │   ├── BloodImage_00020_jpg.rf.5be1260f16e85f6d3b3f940a38cd7392.txt
│   │   │   ├── BloodImage_00022_jpg.rf.acc5dc23d9f1e5c53981a2eda2cd9827.txt
│   │   │   ├── BloodImage_00023_jpg.rf.6dea1d883100d129b1b7ded9add2c24f.txt
│   │   │   ├── BloodImage_00024_jpg.rf.323c51bbfa65bf029bef80b250107268.txt
│   │   │   ├── BloodImage_00028_jpg.rf.1a054fd7ba2ad7f7b3e3245212a14792.txt
│   │   │   ├── BloodImage_00029_jpg.rf.9d8e026ce7b048d2d094f2f7f229f368.txt
│   │   │   ├── BloodImage_00030_jpg.rf.71385416b4e288babd3e52de3941fc6b.txt
│   │   │   ├── BloodImage_00031_jpg.rf.bcce96daec2fe43241ed711e793cb3d4.txt
│   │   │   ├── BloodImage_00032_jpg.rf.99717e8579fd0c8d1fb28dfb5e96dae0.txt
│   │   │   ├── BloodImage_00033_jpg.rf.ed29a91d8384959065269d3584e4acb7.txt
│   │   │   ├── BloodImage_00034_jpg.rf.2ffb51d919ecc92d369cffbfe88f41c3.txt
│   │   │   ├── BloodImage_00035_jpg.rf.ac506be1c23d0f3a09d8e340a008f885.txt
│   │   │   ├── BloodImage_00036_jpg.rf.f9b3f1851acea0c511bdbb05aa0e4200.txt
│   │   │   ├── BloodImage_00037_jpg.rf.f10b2a6625522717d70040e245a43103.txt
│   │   │   ├── BloodImage_00039_jpg.rf.e52773eb869d63407dac92ddc09aa183.txt
│   │   │   ├── BloodImage_00040_jpg.rf.954258c34647369f74e3a37e1e675acb.txt
│   │   │   ├── BloodImage_00041_jpg.rf.80dc89cfe56e555cef5aa8742dec9ed7.txt
│   │   │   ├── BloodImage_00042_jpg.rf.e0e7f2299512850aa9e1496cc40b35a0.txt
│   │   │   ├── BloodImage_00043_jpg.rf.f11aab205044a15d8585d565a04bec19.txt
│   │   │   ├── BloodImage_00045_jpg.rf.293583cdffcb84f3c126ad5fb0a5c10e.txt
│   │   │   ├── BloodImage_00046_jpg.rf.fc4a2bb5179e91b09d0667faf7e6dfa3.txt
│   │   │   ├── BloodImage_00047_jpg.rf.8398043d05fecb77cf4e650d2208fb0d.txt
│   │   │   ├── BloodImage_00048_jpg.rf.728c77ec6eebf109593bfdb9597ea65f.txt
│   │   │   ├── BloodImage_00049_jpg.rf.d34eff6d6dde615eeae0515a9f94ea88.txt
│   │   │   ├── BloodImage_00052_jpg.rf.8b1cb0fdfb126b5629795cd4c7dca0b2.txt
│   │   │   ├── BloodImage_00053_jpg.rf.d7ae83eb982e69dfb0c48b5cb4437e55.txt
│   │   │   ├── BloodImage_00054_jpg.rf.e0701bbac7b3bf5dc3ea26f45b445e2c.txt
│   │   │   ├── BloodImage_00055_jpg.rf.b47dffc3486b0b7c7c1a067421672d51.txt
│   │   │   ├── BloodImage_00056_jpg.rf.319effd6211d38e7645bfa787d3c2dab.txt
│   │   │   ├── BloodImage_00058_jpg.rf.16d6f2ede839c8876e3339c91f4e73ef.txt
│   │   │   ├── BloodImage_00059_jpg.rf.ef31566ba8899bf8358f53c7ce29063c.txt
│   │   │   ├── BloodImage_00064_jpg.rf.bf19df7bdde5d68567661f7a9d2d586e.txt
│   │   │   ├── BloodImage_00065_jpg.rf.ddc13d1753ba439a75be985268c15813.txt
│   │   │   ├── BloodImage_00066_jpg.rf.26aa8e7bf8c8e2601b56d24d5d8a61fb.txt
│   │   │   ├── BloodImage_00067_jpg.rf.244c43eed192e0dcd27ba9bd92b9b18e.txt
│   │   │   ├── BloodImage_00068_jpg.rf.cea435e947d1e8661d14f90e0cc71f6a.txt
│   │   │   ├── BloodImage_00072_jpg.rf.4dea0e09c770a4de243d19921135ce15.txt
│   │   │   ├── BloodImage_00073_jpg.rf.195c2eb474e50e3707a12cb06739508b.txt
│   │   │   ├── BloodImage_00074_jpg.rf.d76fb021c6e3ea3606b74b9f38a982da.txt
│   │   │   ├── BloodImage_00077_jpg.rf.ea1ac390cf353b6825fe2f86375db707.txt
│   │   │   ├── BloodImage_00078_jpg.rf.4f5978ebeafc09a920f9407f8acd2d5a.txt
│   │   │   ├── BloodImage_00079_jpg.rf.2a2c70686066bf5f4634e747dc90dd58.txt
│   │   │   ├── BloodImage_00081_jpg.rf.229ebb7e6402a2ac724a5a662d5b7e99.txt
│   │   │   ├── BloodImage_00082_jpg.rf.66e0043d7cda781d565786264330eb55.txt
│   │   │   ├── BloodImage_00083_jpg.rf.0307243d1591caafc332240f3e4e39b6.txt
│   │   │   ├── BloodImage_00087_jpg.rf.4ed67af94b2d74d0b6db05f10d1fbf57.txt
│   │   │   ├── BloodImage_00088_jpg.rf.19b4ccde27cd787d761ac913158ea7f8.txt
│   │   │   ├── BloodImage_00089_jpg.rf.7277987610ce4b8057fc2255ec1deceb.txt
│   │   │   ├── BloodImage_00091_jpg.rf.6d67a1893a8efb19a33afa4b62e87e93.txt
│   │   │   ├── BloodImage_00094_jpg.rf.9eee88888a92b5208516e8b880cf46b6.txt
│   │   │   ├── BloodImage_00095_jpg.rf.16971a175ba68f23bebe78d5075d540b.txt
│   │   │   ├── BloodImage_00097_jpg.rf.c38c2cbdc33259648be15cfd9a43b2b7.txt
│   │   │   ├── BloodImage_00098_jpg.rf.065b4cca0dc147f1f2768e62f90d059f.txt
│   │   │   ├── BloodImage_00100_jpg.rf.73c6e7f83c1a68208d5c08d359038a8f.txt
│   │   │   ├── BloodImage_00101_jpg.rf.6c604197453717b53def341cb276878a.txt
│   │   │   ├── BloodImage_00103_jpg.rf.9907b3dbc5f0c74428d3c7971b644222.txt
│   │   │   ├── BloodImage_00106_jpg.rf.fd43315c4bf62f68262040e9253e4a99.txt
│   │   │   ├── BloodImage_00111_jpg.rf.2177a3168c80354610c6a7c996981595.txt
│   │   │   ├── BloodImage_00114_jpg.rf.8907da31dbf9aac7107d884322074dcf.txt
│   │   │   ├── BloodImage_00115_jpg.rf.ace3f09def7d9e9a78c5891273f3d9fc.txt
│   │   │   ├── BloodImage_00117_jpg.rf.b35d5c3f818b6732815da9fb1a3bffa1.txt
│   │   │   ├── BloodImage_00124_jpg.rf.f6f4e0138db2ebd88cce18a056627c2d.txt
│   │   │   ├── BloodImage_00125_jpg.rf.2eecb21bba4ce18441b6228e460fc7a8.txt
│   │   │   ├── BloodImage_00132_jpg.rf.209495d7f38eb9b80bc009ce3a9b5f96.txt
│   │   │   ├── BloodImage_00136_jpg.rf.a86a515dfbe0ce2bd57cc87fed8ea862.txt
│   │   │   ├── BloodImage_00137_jpg.rf.264ee70ffac6681843fa4dbcfb36b11f.txt
│   │   │   ├── BloodImage_00139_jpg.rf.5e4411dde64c7638efc96e37c67ff2cb.txt
│   │   │   ├── BloodImage_00140_jpg.rf.e36558fb41aa42903ede72a60b51a73d.txt
│   │   │   └── BloodImage_00142_jpg.rf.4dc589baf76059f3d5b11226b77cdd43.txt
│   │   └── labels.cache
│   └── valid
│       ├── images
│       │   ├── BloodImage_00000_jpg.rf.3aa7a653c80726cbb25447cb697ad7a4.jpg
│       │   ├── BloodImage_00004_jpg.rf.5abe41b92c2d446545da27876795e4ec.jpg
│       │   ├── BloodImage_00012_jpg.rf.427cf7c80c315e412e52ca2bcc0daa30.jpg
│       │   ├── BloodImage_00017_jpg.rf.6ea3d63ae24abe3c42a9b6192dbba5a4.jpg
│       │   ├── BloodImage_00021_jpg.rf.a5330c5a9f79a73c6e526244171cf77b.jpg
│       │   ├── BloodImage_00026_jpg.rf.7937b64ebbc36c4e543219b113a85501.jpg
│       │   ├── BloodImage_00050_jpg.rf.41b081af31e41aec98eb77297ced3caa.jpg
│       │   ├── BloodImage_00057_jpg.rf.f7f41df6357d9be574288e87bf5a48d7.jpg
│       │   ├── BloodImage_00063_jpg.rf.a1a159550894e64ac69bc4c16ac58141.jpg
│       │   ├── BloodImage_00069_jpg.rf.db092f500d30b49d143816202c511b9b.jpg
│       │   ├── BloodImage_00070_jpg.rf.b4fb3f14cc12d19aa8e7d2282fea7617.jpg
│       │   ├── BloodImage_00071_jpg.rf.5fc48a8a54bdf5acd454fc2deb53d5c9.jpg
│       │   ├── BloodImage_00075_jpg.rf.61ef0160b81e5058d559b0cd90050820.jpg
│       │   ├── BloodImage_00076_jpg.rf.8f437a1c177a8400c0a4343ffce1935f.jpg
│       │   ├── BloodImage_00086_jpg.rf.df8471af37e3a15f2eff9d3c7c5e6db0.jpg
│       │   ├── BloodImage_00092_jpg.rf.e11aa418b64c8b6a3ba41ca184dc3b26.jpg
│       │   ├── BloodImage_00093_jpg.rf.dc8a5b88bd84b2f60c36a92e01650b30.jpg
│       │   ├── BloodImage_00104_jpg.rf.2ec5cb75581036d33efb68c2898f15f1.jpg
│       │   ├── BloodImage_00107_jpg.rf.00b6520c219ea6e922de40ca6bfa4009.jpg
│       │   ├── BloodImage_00108_jpg.rf.bd98a88ae6f4c36685e9afe60a66dc07.jpg
│       │   ├── BloodImage_00109_jpg.rf.52f1c330446c3c424684e4d12a6c8bfa.jpg
│       │   ├── BloodImage_00110_jpg.rf.0d0e768475c16aaca2b7cd8d88647bef.jpg
│       │   ├── BloodImage_00123_jpg.rf.95abb6aab1bd67cf8339c23c060f4329.jpg
│       │   └── BloodImage_00126_jpg.rf.b9ced9374b652616c9abfa563340ad8f.jpg
│       ├── labels
│       │   ├── BloodImage_00000_jpg.rf.3aa7a653c80726cbb25447cb697ad7a4.txt
│       │   ├── BloodImage_00004_jpg.rf.5abe41b92c2d446545da27876795e4ec.txt
│       │   ├── BloodImage_00012_jpg.rf.427cf7c80c315e412e52ca2bcc0daa30.txt
│       │   ├── BloodImage_00017_jpg.rf.6ea3d63ae24abe3c42a9b6192dbba5a4.txt
│       │   ├── BloodImage_00021_jpg.rf.a5330c5a9f79a73c6e526244171cf77b.txt
│       │   ├── BloodImage_00026_jpg.rf.7937b64ebbc36c4e543219b113a85501.txt
│       │   ├── BloodImage_00050_jpg.rf.41b081af31e41aec98eb77297ced3caa.txt
│       │   ├── BloodImage_00057_jpg.rf.f7f41df6357d9be574288e87bf5a48d7.txt
│       │   ├── BloodImage_00063_jpg.rf.a1a159550894e64ac69bc4c16ac58141.txt
│       │   ├── BloodImage_00069_jpg.rf.db092f500d30b49d143816202c511b9b.txt
│       │   ├── BloodImage_00070_jpg.rf.b4fb3f14cc12d19aa8e7d2282fea7617.txt
│       │   ├── BloodImage_00071_jpg.rf.5fc48a8a54bdf5acd454fc2deb53d5c9.txt
│       │   ├── BloodImage_00075_jpg.rf.61ef0160b81e5058d559b0cd90050820.txt
│       │   ├── BloodImage_00076_jpg.rf.8f437a1c177a8400c0a4343ffce1935f.txt
│       │   ├── BloodImage_00086_jpg.rf.df8471af37e3a15f2eff9d3c7c5e6db0.txt
│       │   ├── BloodImage_00092_jpg.rf.e11aa418b64c8b6a3ba41ca184dc3b26.txt
│       │   ├── BloodImage_00093_jpg.rf.dc8a5b88bd84b2f60c36a92e01650b30.txt
│       │   ├── BloodImage_00104_jpg.rf.2ec5cb75581036d33efb68c2898f15f1.txt
│       │   ├── BloodImage_00107_jpg.rf.00b6520c219ea6e922de40ca6bfa4009.txt
│       │   ├── BloodImage_00108_jpg.rf.bd98a88ae6f4c36685e9afe60a66dc07.txt
│       │   ├── BloodImage_00109_jpg.rf.52f1c330446c3c424684e4d12a6c8bfa.txt
│       │   ├── BloodImage_00110_jpg.rf.0d0e768475c16aaca2b7cd8d88647bef.txt
│       │   ├── BloodImage_00123_jpg.rf.95abb6aab1bd67cf8339c23c060f4329.txt
│       │   └── BloodImage_00126_jpg.rf.b9ced9374b652616c9abfa563340ad8f.txt
│       └── labels.cache
├── client_1
│   ├── data.yaml
│   ├── test
│   │   ├── images
│   │   │   ├── BloodImage_00190_jpg.rf.257a9f96afccdbed515a290b694f4c15.jpg
│   │   │   ├── BloodImage_00191_jpg.rf.9fecdaf56689fc80d667ef8d8da6bc27.jpg
│   │   │   ├── BloodImage_00204_jpg.rf.0555bc62812f0987a35f05f0960dd7c4.jpg
│   │   │   ├── BloodImage_00227_jpg.rf.816711b066fae3bdf16851eaebc13eb5.jpg
│   │   │   ├── BloodImage_00235_jpg.rf.6028248c6b2b38ea0d4045d289d56ca3.jpg
│   │   │   ├── BloodImage_00241_jpg.rf.00d6593ca59122287542bf819f62fb43.jpg
│   │   │   ├── BloodImage_00254_jpg.rf.3783a73c7c92da8010897e8fb9d14448.jpg
│   │   │   ├── BloodImage_00265_jpg.rf.acdcdc5c22ee42608c69240af0c5d732.jpg
│   │   │   ├── BloodImage_00266_jpg.rf.ec59f2e7492adb14ff2eb59f3c94c92f.jpg
│   │   │   ├── BloodImage_00275_jpg.rf.850c4258ce35168b019485bf3b0229ee.jpg
│   │   │   ├── BloodImage_00278_jpg.rf.c271aa5245a39cc10463d8d5f9ee4bf8.jpg
│   │   │   └── BloodImage_00284_jpg.rf.fb0089e655be201ea27173598ef30825.jpg
│   │   ├── labels
│   │   │   ├── BloodImage_00190_jpg.rf.257a9f96afccdbed515a290b694f4c15.txt
│   │   │   ├── BloodImage_00191_jpg.rf.9fecdaf56689fc80d667ef8d8da6bc27.txt
│   │   │   ├── BloodImage_00204_jpg.rf.0555bc62812f0987a35f05f0960dd7c4.txt
│   │   │   ├── BloodImage_00227_jpg.rf.816711b066fae3bdf16851eaebc13eb5.txt
│   │   │   ├── BloodImage_00235_jpg.rf.6028248c6b2b38ea0d4045d289d56ca3.txt
│   │   │   ├── BloodImage_00241_jpg.rf.00d6593ca59122287542bf819f62fb43.txt
│   │   │   ├── BloodImage_00254_jpg.rf.3783a73c7c92da8010897e8fb9d14448.txt
│   │   │   ├── BloodImage_00265_jpg.rf.acdcdc5c22ee42608c69240af0c5d732.txt
│   │   │   ├── BloodImage_00266_jpg.rf.ec59f2e7492adb14ff2eb59f3c94c92f.txt
│   │   │   ├── BloodImage_00275_jpg.rf.850c4258ce35168b019485bf3b0229ee.txt
│   │   │   ├── BloodImage_00278_jpg.rf.c271aa5245a39cc10463d8d5f9ee4bf8.txt
│   │   │   └── BloodImage_00284_jpg.rf.fb0089e655be201ea27173598ef30825.txt
│   │   └── labels.cache
│   ├── train
│   │   ├── images
│   │   │   ├── BloodImage_00143_jpg.rf.3d6cc38d06ee5ad7856c965d819cf2b2.jpg
│   │   │   ├── BloodImage_00144_jpg.rf.a3944bb4f6d72390c0043e0ff7138e5e.jpg
│   │   │   ├── BloodImage_00145_jpg.rf.3053bb2ea20050aded47c62d31766c64.jpg
│   │   │   ├── BloodImage_00147_jpg.rf.922f24aecdc26dd8165f897a4ee02826.jpg
│   │   │   ├── BloodImage_00148_jpg.rf.5dbdae5ee92018cabfef817ae96d88db.jpg
│   │   │   ├── BloodImage_00149_jpg.rf.6c3341c91f81cd2086403b77c571fd7d.jpg
│   │   │   ├── BloodImage_00150_jpg.rf.6449dff4812a9a6f430a4a7f5853ac2c.jpg
│   │   │   ├── BloodImage_00152_jpg.rf.5840888b497ba4f60174e17b5c62c622.jpg
│   │   │   ├── BloodImage_00156_jpg.rf.fe572186390c601ff6d64ae7c79ce5d1.jpg
│   │   │   ├── BloodImage_00157_jpg.rf.479c123e5583617fddeb18f3ec477b21.jpg
│   │   │   ├── BloodImage_00158_jpg.rf.f7724626cc249fcf1bb8a031c4f00d10.jpg
│   │   │   ├── BloodImage_00159_jpg.rf.aca04d5f87e8666ecacb56ff2d8d64cc.jpg
│   │   │   ├── BloodImage_00162_jpg.rf.36af3fa94f239d77668d63d3816f214c.jpg
│   │   │   ├── BloodImage_00163_jpg.rf.10d0f078d6f9fabdacb5e35ba7e301e5.jpg
│   │   │   ├── BloodImage_00164_jpg.rf.7b57dc64e26226501d24c80128890e79.jpg
│   │   │   ├── BloodImage_00165_jpg.rf.ea1062188dc6107aaf184afeb2c25e7c.jpg
│   │   │   ├── BloodImage_00166_jpg.rf.ed92901387d097b6f009c1e2ba5b8d1f.jpg
│   │   │   ├── BloodImage_00167_jpg.rf.e91bbcbf657ad27167c16123ec4fe2bf.jpg
│   │   │   ├── BloodImage_00168_jpg.rf.700981f577f08882a8e07f89d0ed2fad.jpg
│   │   │   ├── BloodImage_00169_jpg.rf.cc6d411b83e99bb4940d21a21daa1225.jpg
│   │   │   ├── BloodImage_00170_jpg.rf.f8660bd505ac96c28a3aacca8f5021d6.jpg
│   │   │   ├── BloodImage_00171_jpg.rf.7cc75df374b62663fbb3d78001d6b297.jpg
│   │   │   ├── BloodImage_00172_jpg.rf.1f7d1cced720a8a6497ae2b1404baec4.jpg
│   │   │   ├── BloodImage_00174_jpg.rf.6ba0937610b41271846f603fa0a1719e.jpg
│   │   │   ├── BloodImage_00175_jpg.rf.ead23ac106920eef89b4a847e120afdc.jpg
│   │   │   ├── BloodImage_00176_jpg.rf.ee54eb7f0a5f9c31ca2f9349d8c8788b.jpg
│   │   │   ├── BloodImage_00177_jpg.rf.106f44d2fc8af2a4b10923aff7e10cff.jpg
│   │   │   ├── BloodImage_00178_jpg.rf.3ee03e327ef350ade59d97001057e2b1.jpg
│   │   │   ├── BloodImage_00179_jpg.rf.9221c3d4c33e730d66b171b0b391b10f.jpg
│   │   │   ├── BloodImage_00180_jpg.rf.f33a017035e656480ca5fee99c468ba4.jpg
│   │   │   ├── BloodImage_00184_jpg.rf.da57069a5e8dd93a53f3e6ca2b5e5e9e.jpg
│   │   │   ├── BloodImage_00189_jpg.rf.411bfea1e3fc7205a82d3ed4c2bfa55e.jpg
│   │   │   ├── BloodImage_00192_jpg.rf.6eb1be4495880aa63bac2c3c0f83293f.jpg
│   │   │   ├── BloodImage_00193_jpg.rf.c9383369f0d81ae5fafc7a343e0e1dfc.jpg
│   │   │   ├── BloodImage_00195_jpg.rf.9195f71efe46a2281fcfefac3f3cffe9.jpg
│   │   │   ├── BloodImage_00196_jpg.rf.a18bf688a25ad88793c1483cca202e74.jpg
│   │   │   ├── BloodImage_00197_jpg.rf.384cd993a7f935e5e567889b62bc64f0.jpg
│   │   │   ├── BloodImage_00198_jpg.rf.6e646f3b1c05b6e922dfbbab3e10f236.jpg
│   │   │   ├── BloodImage_00199_jpg.rf.a8552f25787a8b0fe26d6d9c86037f76.jpg
│   │   │   ├── BloodImage_00200_jpg.rf.40a9a85811c5f4c96fee214db545fa87.jpg
│   │   │   ├── BloodImage_00201_jpg.rf.5a2603a7894db7759ae939fbfb3cd4cd.jpg
│   │   │   ├── BloodImage_00202_jpg.rf.51344f7c6b1264d75d92526ede56dc16.jpg
│   │   │   ├── BloodImage_00203_jpg.rf.115fb77f1a16fafab02c268413e9d583.jpg
│   │   │   ├── BloodImage_00206_jpg.rf.db97566d514a1a34c610bd3f5492a956.jpg
│   │   │   ├── BloodImage_00207_jpg.rf.dc3c5696f64f79af75b1d2435cfe64e4.jpg
│   │   │   ├── BloodImage_00208_jpg.rf.64f53f21241c04c530e5e21a74ebbe8a.jpg
│   │   │   ├── BloodImage_00209_jpg.rf.41271f310b73e8874ee2283bfd1ec57a.jpg
│   │   │   ├── BloodImage_00210_jpg.rf.46c6f6f63fe18c882ab02b7335797323.jpg
│   │   │   ├── BloodImage_00212_jpg.rf.9a40dbf065b86f8f016210cb7d0b45ad.jpg
│   │   │   ├── BloodImage_00214_jpg.rf.57f3b4693d2403575edd17efbb659df3.jpg
│   │   │   ├── BloodImage_00215_jpg.rf.0fd631a7bf59ca5c45e917b1b3242ccc.jpg
│   │   │   ├── BloodImage_00218_jpg.rf.48ff87de6fc7feade9ae09dc7119bba2.jpg
│   │   │   ├── BloodImage_00219_jpg.rf.05bca517d8fa02e385fdbf8d6ef18b93.jpg
│   │   │   ├── BloodImage_00220_jpg.rf.7b39d588266b020dfe15e07ffa1957fd.jpg
│   │   │   ├── BloodImage_00222_jpg.rf.80e1ba91312ae4e120af56a9935a6767.jpg
│   │   │   ├── BloodImage_00223_jpg.rf.6c56bb9119b89f45c357bb45d4dc322b.jpg
│   │   │   ├── BloodImage_00224_jpg.rf.cb09b3e08c44e1027385f1e243a2de03.jpg
│   │   │   ├── BloodImage_00226_jpg.rf.397eea892c42e753514d23b18d7b1e59.jpg
│   │   │   ├── BloodImage_00229_jpg.rf.3d7c28ad711e53748d7bc8a3e1303b40.jpg
│   │   │   ├── BloodImage_00230_jpg.rf.effb8985eb1735f463f281438eef3034.jpg
│   │   │   ├── BloodImage_00231_jpg.rf.cb826df690d18f2d7a4f39f0d3397bef.jpg
│   │   │   ├── BloodImage_00232_jpg.rf.9409125a3bcfc13306b85c0be5e85f4e.jpg
│   │   │   ├── BloodImage_00233_jpg.rf.950d2e3d11ea3b7bed120d0048ffe8af.jpg
│   │   │   ├── BloodImage_00234_jpg.rf.b7cf2a5b00777f2d2956e42ecee12126.jpg
│   │   │   ├── BloodImage_00236_jpg.rf.6b8b34573f2ef1bc250ea2c1825156e4.jpg
│   │   │   ├── BloodImage_00237_jpg.rf.93ab8101ea0898948bf73d519eafc904.jpg
│   │   │   ├── BloodImage_00239_jpg.rf.51c744d135f1bcb27f30dcbd190d92ba.jpg
│   │   │   ├── BloodImage_00240_jpg.rf.1fb953fc9e166fb47dd4ab07d8529d58.jpg
│   │   │   ├── BloodImage_00242_jpg.rf.86b7a35aa15698e7b53c98fd4aef2b7b.jpg
│   │   │   ├── BloodImage_00243_jpg.rf.4ac20502da26cf12c367cac90a517b3e.jpg
│   │   │   ├── BloodImage_00244_jpg.rf.8d47adf2dc728f4e0406b474f7d54ccb.jpg
│   │   │   ├── BloodImage_00247_jpg.rf.cb733c5cc360e7218ec5ad095c73f9dc.jpg
│   │   │   ├── BloodImage_00248_jpg.rf.76d2b0acb84359a0ce09b529ec5aae33.jpg
│   │   │   ├── BloodImage_00249_jpg.rf.cb55a4a3ff2397d5fd193c0493c64611.jpg
│   │   │   ├── BloodImage_00250_jpg.rf.12516222ad772dc4ee0fcd8c7e5cb689.jpg
│   │   │   ├── BloodImage_00251_jpg.rf.ee764f36f42a5c328aa4a70990a79c8f.jpg
│   │   │   ├── BloodImage_00253_jpg.rf.a7a7dd010daca5e07f8bc289142c6b7d.jpg
│   │   │   ├── BloodImage_00255_jpg.rf.398f0c38bcfb8813e3ce3e283abac9dc.jpg
│   │   │   ├── BloodImage_00256_jpg.rf.94c1a3062aeebb1df2b67c20e0532932.jpg
│   │   │   ├── BloodImage_00257_jpg.rf.b905d2edb475685ae36d88e33b405561.jpg
│   │   │   ├── BloodImage_00260_jpg.rf.d9aa0111d492a04b8ca8f24d0f9aed00.jpg
│   │   │   ├── BloodImage_00261_jpg.rf.9b6dcb3b7a59fd289d3392f106ad37b8.jpg
│   │   │   ├── BloodImage_00262_jpg.rf.94805769ed783f2c8f4cadb66bd1620d.jpg
│   │   │   ├── BloodImage_00264_jpg.rf.db2e3d0fd7569640d407e1dacf90a456.jpg
│   │   │   └── BloodImage_00267_jpg.rf.d636a81cd23566cca0cad3ddc6bf49a8.jpg
│   │   ├── labels
│   │   │   ├── BloodImage_00143_jpg.rf.3d6cc38d06ee5ad7856c965d819cf2b2.txt
│   │   │   ├── BloodImage_00144_jpg.rf.a3944bb4f6d72390c0043e0ff7138e5e.txt
│   │   │   ├── BloodImage_00145_jpg.rf.3053bb2ea20050aded47c62d31766c64.txt
│   │   │   ├── BloodImage_00147_jpg.rf.922f24aecdc26dd8165f897a4ee02826.txt
│   │   │   ├── BloodImage_00148_jpg.rf.5dbdae5ee92018cabfef817ae96d88db.txt
│   │   │   ├── BloodImage_00149_jpg.rf.6c3341c91f81cd2086403b77c571fd7d.txt
│   │   │   ├── BloodImage_00150_jpg.rf.6449dff4812a9a6f430a4a7f5853ac2c.txt
│   │   │   ├── BloodImage_00152_jpg.rf.5840888b497ba4f60174e17b5c62c622.txt
│   │   │   ├── BloodImage_00156_jpg.rf.fe572186390c601ff6d64ae7c79ce5d1.txt
│   │   │   ├── BloodImage_00157_jpg.rf.479c123e5583617fddeb18f3ec477b21.txt
│   │   │   ├── BloodImage_00158_jpg.rf.f7724626cc249fcf1bb8a031c4f00d10.txt
│   │   │   ├── BloodImage_00159_jpg.rf.aca04d5f87e8666ecacb56ff2d8d64cc.txt
│   │   │   ├── BloodImage_00162_jpg.rf.36af3fa94f239d77668d63d3816f214c.txt
│   │   │   ├── BloodImage_00163_jpg.rf.10d0f078d6f9fabdacb5e35ba7e301e5.txt
│   │   │   ├── BloodImage_00164_jpg.rf.7b57dc64e26226501d24c80128890e79.txt
│   │   │   ├── BloodImage_00165_jpg.rf.ea1062188dc6107aaf184afeb2c25e7c.txt
│   │   │   ├── BloodImage_00166_jpg.rf.ed92901387d097b6f009c1e2ba5b8d1f.txt
│   │   │   ├── BloodImage_00167_jpg.rf.e91bbcbf657ad27167c16123ec4fe2bf.txt
│   │   │   ├── BloodImage_00168_jpg.rf.700981f577f08882a8e07f89d0ed2fad.txt
│   │   │   ├── BloodImage_00169_jpg.rf.cc6d411b83e99bb4940d21a21daa1225.txt
│   │   │   ├── BloodImage_00170_jpg.rf.f8660bd505ac96c28a3aacca8f5021d6.txt
│   │   │   ├── BloodImage_00171_jpg.rf.7cc75df374b62663fbb3d78001d6b297.txt
│   │   │   ├── BloodImage_00172_jpg.rf.1f7d1cced720a8a6497ae2b1404baec4.txt
│   │   │   ├── BloodImage_00174_jpg.rf.6ba0937610b41271846f603fa0a1719e.txt
│   │   │   ├── BloodImage_00175_jpg.rf.ead23ac106920eef89b4a847e120afdc.txt
│   │   │   ├── BloodImage_00176_jpg.rf.ee54eb7f0a5f9c31ca2f9349d8c8788b.txt
│   │   │   ├── BloodImage_00177_jpg.rf.106f44d2fc8af2a4b10923aff7e10cff.txt
│   │   │   ├── BloodImage_00178_jpg.rf.3ee03e327ef350ade59d97001057e2b1.txt
│   │   │   ├── BloodImage_00179_jpg.rf.9221c3d4c33e730d66b171b0b391b10f.txt
│   │   │   ├── BloodImage_00180_jpg.rf.f33a017035e656480ca5fee99c468ba4.txt
│   │   │   ├── BloodImage_00184_jpg.rf.da57069a5e8dd93a53f3e6ca2b5e5e9e.txt
│   │   │   ├── BloodImage_00189_jpg.rf.411bfea1e3fc7205a82d3ed4c2bfa55e.txt
│   │   │   ├── BloodImage_00192_jpg.rf.6eb1be4495880aa63bac2c3c0f83293f.txt
│   │   │   ├── BloodImage_00193_jpg.rf.c9383369f0d81ae5fafc7a343e0e1dfc.txt
│   │   │   ├── BloodImage_00195_jpg.rf.9195f71efe46a2281fcfefac3f3cffe9.txt
│   │   │   ├── BloodImage_00196_jpg.rf.a18bf688a25ad88793c1483cca202e74.txt
│   │   │   ├── BloodImage_00197_jpg.rf.384cd993a7f935e5e567889b62bc64f0.txt
│   │   │   ├── BloodImage_00198_jpg.rf.6e646f3b1c05b6e922dfbbab3e10f236.txt
│   │   │   ├── BloodImage_00199_jpg.rf.a8552f25787a8b0fe26d6d9c86037f76.txt
│   │   │   ├── BloodImage_00200_jpg.rf.40a9a85811c5f4c96fee214db545fa87.txt
│   │   │   ├── BloodImage_00201_jpg.rf.5a2603a7894db7759ae939fbfb3cd4cd.txt
│   │   │   ├── BloodImage_00202_jpg.rf.51344f7c6b1264d75d92526ede56dc16.txt
│   │   │   ├── BloodImage_00203_jpg.rf.115fb77f1a16fafab02c268413e9d583.txt
│   │   │   ├── BloodImage_00206_jpg.rf.db97566d514a1a34c610bd3f5492a956.txt
│   │   │   ├── BloodImage_00207_jpg.rf.dc3c5696f64f79af75b1d2435cfe64e4.txt
│   │   │   ├── BloodImage_00208_jpg.rf.64f53f21241c04c530e5e21a74ebbe8a.txt
│   │   │   ├── BloodImage_00209_jpg.rf.41271f310b73e8874ee2283bfd1ec57a.txt
│   │   │   ├── BloodImage_00210_jpg.rf.46c6f6f63fe18c882ab02b7335797323.txt
│   │   │   ├── BloodImage_00212_jpg.rf.9a40dbf065b86f8f016210cb7d0b45ad.txt
│   │   │   ├── BloodImage_00214_jpg.rf.57f3b4693d2403575edd17efbb659df3.txt
│   │   │   ├── BloodImage_00215_jpg.rf.0fd631a7bf59ca5c45e917b1b3242ccc.txt
│   │   │   ├── BloodImage_00218_jpg.rf.48ff87de6fc7feade9ae09dc7119bba2.txt
│   │   │   ├── BloodImage_00219_jpg.rf.05bca517d8fa02e385fdbf8d6ef18b93.txt
│   │   │   ├── BloodImage_00220_jpg.rf.7b39d588266b020dfe15e07ffa1957fd.txt
│   │   │   ├── BloodImage_00222_jpg.rf.80e1ba91312ae4e120af56a9935a6767.txt
│   │   │   ├── BloodImage_00223_jpg.rf.6c56bb9119b89f45c357bb45d4dc322b.txt
│   │   │   ├── BloodImage_00224_jpg.rf.cb09b3e08c44e1027385f1e243a2de03.txt
│   │   │   ├── BloodImage_00226_jpg.rf.397eea892c42e753514d23b18d7b1e59.txt
│   │   │   ├── BloodImage_00229_jpg.rf.3d7c28ad711e53748d7bc8a3e1303b40.txt
│   │   │   ├── BloodImage_00230_jpg.rf.effb8985eb1735f463f281438eef3034.txt
│   │   │   ├── BloodImage_00231_jpg.rf.cb826df690d18f2d7a4f39f0d3397bef.txt
│   │   │   ├── BloodImage_00232_jpg.rf.9409125a3bcfc13306b85c0be5e85f4e.txt
│   │   │   ├── BloodImage_00233_jpg.rf.950d2e3d11ea3b7bed120d0048ffe8af.txt
│   │   │   ├── BloodImage_00234_jpg.rf.b7cf2a5b00777f2d2956e42ecee12126.txt
│   │   │   ├── BloodImage_00236_jpg.rf.6b8b34573f2ef1bc250ea2c1825156e4.txt
│   │   │   ├── BloodImage_00237_jpg.rf.93ab8101ea0898948bf73d519eafc904.txt
│   │   │   ├── BloodImage_00239_jpg.rf.51c744d135f1bcb27f30dcbd190d92ba.txt
│   │   │   ├── BloodImage_00240_jpg.rf.1fb953fc9e166fb47dd4ab07d8529d58.txt
│   │   │   ├── BloodImage_00242_jpg.rf.86b7a35aa15698e7b53c98fd4aef2b7b.txt
│   │   │   ├── BloodImage_00243_jpg.rf.4ac20502da26cf12c367cac90a517b3e.txt
│   │   │   ├── BloodImage_00244_jpg.rf.8d47adf2dc728f4e0406b474f7d54ccb.txt
│   │   │   ├── BloodImage_00247_jpg.rf.cb733c5cc360e7218ec5ad095c73f9dc.txt
│   │   │   ├── BloodImage_00248_jpg.rf.76d2b0acb84359a0ce09b529ec5aae33.txt
│   │   │   ├── BloodImage_00249_jpg.rf.cb55a4a3ff2397d5fd193c0493c64611.txt
│   │   │   ├── BloodImage_00250_jpg.rf.12516222ad772dc4ee0fcd8c7e5cb689.txt
│   │   │   ├── BloodImage_00251_jpg.rf.ee764f36f42a5c328aa4a70990a79c8f.txt
│   │   │   ├── BloodImage_00253_jpg.rf.a7a7dd010daca5e07f8bc289142c6b7d.txt
│   │   │   ├── BloodImage_00255_jpg.rf.398f0c38bcfb8813e3ce3e283abac9dc.txt
│   │   │   ├── BloodImage_00256_jpg.rf.94c1a3062aeebb1df2b67c20e0532932.txt
│   │   │   ├── BloodImage_00257_jpg.rf.b905d2edb475685ae36d88e33b405561.txt
│   │   │   ├── BloodImage_00260_jpg.rf.d9aa0111d492a04b8ca8f24d0f9aed00.txt
│   │   │   ├── BloodImage_00261_jpg.rf.9b6dcb3b7a59fd289d3392f106ad37b8.txt
│   │   │   ├── BloodImage_00262_jpg.rf.94805769ed783f2c8f4cadb66bd1620d.txt
│   │   │   ├── BloodImage_00264_jpg.rf.db2e3d0fd7569640d407e1dacf90a456.txt
│   │   │   └── BloodImage_00267_jpg.rf.d636a81cd23566cca0cad3ddc6bf49a8.txt
│   │   └── labels.cache
│   └── valid
│       ├── images
│       │   ├── BloodImage_00127_jpg.rf.3f6dcb7c8f7c879500ca02ca0cf0560b.jpg
│       │   ├── BloodImage_00130_jpg.rf.957bac5f70900739553002f2785a05f6.jpg
│       │   ├── BloodImage_00135_jpg.rf.013734f914cafcbb4697bb45f3b51b14.jpg
│       │   ├── BloodImage_00141_jpg.rf.1cdac383e5c96688efd3be47f4a68a7b.jpg
│       │   ├── BloodImage_00161_jpg.rf.ef350baedf3cbe5f75dea76b02b2ff15.jpg
│       │   ├── BloodImage_00182_jpg.rf.97ea83e97b1a0a2a4fc276cc1c21aaac.jpg
│       │   ├── BloodImage_00183_jpg.rf.c50a3e8b469ed94ff2868c7cd91ca226.jpg
│       │   ├── BloodImage_00187_jpg.rf.b7313efba5142b0ab554bf9858174c2e.jpg
│       │   ├── BloodImage_00205_jpg.rf.889fc1f80d77cf8f4adf32ca5fab8858.jpg
│       │   ├── BloodImage_00211_jpg.rf.526b43172b7bb6aa3943255906bbb2e3.jpg
│       │   ├── BloodImage_00216_jpg.rf.195f7ad40674b8de4777ef80ab918882.jpg
│       │   ├── BloodImage_00217_jpg.rf.5536b798a73000bea3a6c6db9fc613e4.jpg
│       │   ├── BloodImage_00221_jpg.rf.949f34eb0477e5d7c19f94cb4e32d5a5.jpg
│       │   ├── BloodImage_00225_jpg.rf.1ee0fa1432d3a5484a4d946a5a0bbda3.jpg
│       │   ├── BloodImage_00228_jpg.rf.111186a1cfd183edefe928d8d0205179.jpg
│       │   ├── BloodImage_00245_jpg.rf.826f18f0f09221813c3895972842198a.jpg
│       │   ├── BloodImage_00246_jpg.rf.0db836a82f49eb424a68ed4f13af0f76.jpg
│       │   ├── BloodImage_00252_jpg.rf.cd5ca37fc581f373cada4297a6c0e5ef.jpg
│       │   ├── BloodImage_00258_jpg.rf.f0081a194a94d488b855f6f9692d8c74.jpg
│       │   ├── BloodImage_00259_jpg.rf.da944c073232164e4ccae44e6b183860.jpg
│       │   ├── BloodImage_00263_jpg.rf.1836ce652322fbd8d57ba81f0008e694.jpg
│       │   ├── BloodImage_00270_jpg.rf.e495ce06a313608e169c9720480c4cc7.jpg
│       │   ├── BloodImage_00272_jpg.rf.b8586a591688b3fb33cfc8de105c8add.jpg
│       │   └── BloodImage_00273_jpg.rf.35dfe89de85b458f719aeaaa68846f7d.jpg
│       ├── labels
│       │   ├── BloodImage_00127_jpg.rf.3f6dcb7c8f7c879500ca02ca0cf0560b.txt
│       │   ├── BloodImage_00130_jpg.rf.957bac5f70900739553002f2785a05f6.txt
│       │   ├── BloodImage_00135_jpg.rf.013734f914cafcbb4697bb45f3b51b14.txt
│       │   ├── BloodImage_00141_jpg.rf.1cdac383e5c96688efd3be47f4a68a7b.txt
│       │   ├── BloodImage_00161_jpg.rf.ef350baedf3cbe5f75dea76b02b2ff15.txt
│       │   ├── BloodImage_00182_jpg.rf.97ea83e97b1a0a2a4fc276cc1c21aaac.txt
│       │   ├── BloodImage_00183_jpg.rf.c50a3e8b469ed94ff2868c7cd91ca226.txt
│       │   ├── BloodImage_00187_jpg.rf.b7313efba5142b0ab554bf9858174c2e.txt
│       │   ├── BloodImage_00205_jpg.rf.889fc1f80d77cf8f4adf32ca5fab8858.txt
│       │   ├── BloodImage_00211_jpg.rf.526b43172b7bb6aa3943255906bbb2e3.txt
│       │   ├── BloodImage_00216_jpg.rf.195f7ad40674b8de4777ef80ab918882.txt
│       │   ├── BloodImage_00217_jpg.rf.5536b798a73000bea3a6c6db9fc613e4.txt
│       │   ├── BloodImage_00221_jpg.rf.949f34eb0477e5d7c19f94cb4e32d5a5.txt
│       │   ├── BloodImage_00225_jpg.rf.1ee0fa1432d3a5484a4d946a5a0bbda3.txt
│       │   ├── BloodImage_00228_jpg.rf.111186a1cfd183edefe928d8d0205179.txt
│       │   ├── BloodImage_00245_jpg.rf.826f18f0f09221813c3895972842198a.txt
│       │   ├── BloodImage_00246_jpg.rf.0db836a82f49eb424a68ed4f13af0f76.txt
│       │   ├── BloodImage_00252_jpg.rf.cd5ca37fc581f373cada4297a6c0e5ef.txt
│       │   ├── BloodImage_00258_jpg.rf.f0081a194a94d488b855f6f9692d8c74.txt
│       │   ├── BloodImage_00259_jpg.rf.da944c073232164e4ccae44e6b183860.txt
│       │   ├── BloodImage_00263_jpg.rf.1836ce652322fbd8d57ba81f0008e694.txt
│       │   ├── BloodImage_00270_jpg.rf.e495ce06a313608e169c9720480c4cc7.txt
│       │   ├── BloodImage_00272_jpg.rf.b8586a591688b3fb33cfc8de105c8add.txt
│       │   └── BloodImage_00273_jpg.rf.35dfe89de85b458f719aeaaa68846f7d.txt
│       └── labels.cache
└── client_2
    ├── data.yaml
    ├── test
    │   ├── images
    │   │   ├── BloodImage_00289_jpg.rf.e4e288486cea56079f9a7c913d4b450b.jpg
    │   │   ├── BloodImage_00301_jpg.rf.105fe61dd7143cb22960ed9829c67727.jpg
    │   │   ├── BloodImage_00302_jpg.rf.9a22d757ab1acdd7ac35da2a8a0a2586.jpg
    │   │   ├── BloodImage_00325_jpg.rf.5aaa15454be96e1334cc7e22399bbee4.jpg
    │   │   ├── BloodImage_00334_jpg.rf.5f7249b59c3c4a0043325619a929116f.jpg
    │   │   ├── BloodImage_00336_jpg.rf.0c43658168a0602346cb8c597cde2cb4.jpg
    │   │   ├── BloodImage_00337_jpg.rf.fe254f7319162b61c5fbf7ce0cb0f534.jpg
    │   │   ├── BloodImage_00350_jpg.rf.db4df841149322ca365bda6df243a4c8.jpg
    │   │   ├── BloodImage_00359_jpg.rf.56ffe9f91e22e7d456aa886ce6ca117d.jpg
    │   │   ├── BloodImage_00369_jpg.rf.49cee9c3a29b86001acadc99962652d6.jpg
    │   │   ├── BloodImage_00385_jpg.rf.3b93195bce5adeae2a3d6d5cb9d12033.jpg
    │   │   └── BloodImage_00386_jpg.rf.30a456d560fcbb4662900a288de765c3.jpg
    │   ├── labels
    │   │   ├── BloodImage_00289_jpg.rf.e4e288486cea56079f9a7c913d4b450b.txt
    │   │   ├── BloodImage_00301_jpg.rf.105fe61dd7143cb22960ed9829c67727.txt
    │   │   ├── BloodImage_00302_jpg.rf.9a22d757ab1acdd7ac35da2a8a0a2586.txt
    │   │   ├── BloodImage_00325_jpg.rf.5aaa15454be96e1334cc7e22399bbee4.txt
    │   │   ├── BloodImage_00334_jpg.rf.5f7249b59c3c4a0043325619a929116f.txt
    │   │   ├── BloodImage_00336_jpg.rf.0c43658168a0602346cb8c597cde2cb4.txt
    │   │   ├── BloodImage_00337_jpg.rf.fe254f7319162b61c5fbf7ce0cb0f534.txt
    │   │   ├── BloodImage_00350_jpg.rf.db4df841149322ca365bda6df243a4c8.txt
    │   │   ├── BloodImage_00359_jpg.rf.56ffe9f91e22e7d456aa886ce6ca117d.txt
    │   │   ├── BloodImage_00369_jpg.rf.49cee9c3a29b86001acadc99962652d6.txt
    │   │   ├── BloodImage_00385_jpg.rf.3b93195bce5adeae2a3d6d5cb9d12033.txt
    │   │   └── BloodImage_00386_jpg.rf.30a456d560fcbb4662900a288de765c3.txt
    │   └── labels.cache
    ├── train
    │   ├── images
    │   │   ├── BloodImage_00268_jpg.rf.e1d8eccd1a77cadaeadebca867a24c62.jpg
    │   │   ├── BloodImage_00269_jpg.rf.8475d6c8dead19457a63ae00dd84c924.jpg
    │   │   ├── BloodImage_00271_jpg.rf.08085d620b04fabd323d2b1ed8a64f1d.jpg
    │   │   ├── BloodImage_00279_jpg.rf.5ade4f1f61924eb07fb0a8169278d395.jpg
    │   │   ├── BloodImage_00282_jpg.rf.d0067872fcfb4df4e5bb9cf783110825.jpg
    │   │   ├── BloodImage_00283_jpg.rf.6263895fbbc2628f18e1a0ed61c74bd6.jpg
    │   │   ├── BloodImage_00285_jpg.rf.05ee9fb43b10619cff770fa0c5d124f5.jpg
    │   │   ├── BloodImage_00287_jpg.rf.b03ee5db05e8750c2d71cc5444f3905c.jpg
    │   │   ├── BloodImage_00288_jpg.rf.3adee28ab6ed4a5420f851a6b3d275df.jpg
    │   │   ├── BloodImage_00290_jpg.rf.870ab7d987c50d829a5f28c1a972d917.jpg
    │   │   ├── BloodImage_00291_jpg.rf.92ed5a625cc17a671eae55fd48329f02.jpg
    │   │   ├── BloodImage_00292_jpg.rf.469f85f31734b15c32cbd1a96e274673.jpg
    │   │   ├── BloodImage_00293_jpg.rf.8d19f05e21730e57d74657b0ce9b950c.jpg
    │   │   ├── BloodImage_00294_jpg.rf.8ea49a5b1990a964de2ce838ab4a9525.jpg
    │   │   ├── BloodImage_00295_jpg.rf.b8ad2a56170ce42d667c25819b788c1d.jpg
    │   │   ├── BloodImage_00299_jpg.rf.e8fa71a67d5eae37ed66b6b7976772c1.jpg
    │   │   ├── BloodImage_00303_jpg.rf.96019c5afa192dbb914016f0243964fd.jpg
    │   │   ├── BloodImage_00304_jpg.rf.d065dccc029aa497e69aac485f5c7ea9.jpg
    │   │   ├── BloodImage_00305_jpg.rf.3be7c63f74a5b205817365ceececd3bc.jpg
    │   │   ├── BloodImage_00307_jpg.rf.f3a9b5a5c544183e2035095fb497fb4c.jpg
    │   │   ├── BloodImage_00308_jpg.rf.4598085d3090b176e16953ab1054aef8.jpg
    │   │   ├── BloodImage_00310_jpg.rf.52e3296f7a0d9e655e36ce1ac367faff.jpg
    │   │   ├── BloodImage_00311_jpg.rf.5b1c9c2ae20f4f304393360e4e72d7ac.jpg
    │   │   ├── BloodImage_00312_jpg.rf.644485683ce0a8eadc17594a2b970697.jpg
    │   │   ├── BloodImage_00313_jpg.rf.48fee7adf53e53903cefb3254bfc852e.jpg
    │   │   ├── BloodImage_00314_jpg.rf.722b30c67ca8aef37b0a8001c53c12e7.jpg
    │   │   ├── BloodImage_00317_jpg.rf.6397dd207ef8b48c5ece8f699bfec8e5.jpg
    │   │   ├── BloodImage_00318_jpg.rf.ae41d28eabbf137f0f7c3629398e73fa.jpg
    │   │   ├── BloodImage_00320_jpg.rf.b252995123ab52685fe98090382015f9.jpg
    │   │   ├── BloodImage_00322_jpg.rf.f3faf31c0fe23b4f042343e7e33ccc18.jpg
    │   │   ├── BloodImage_00323_jpg.rf.f4e0e3a72c74a8b196373bb12697e49e.jpg
    │   │   ├── BloodImage_00324_jpg.rf.d2fa0cee237fa4698401213fd2dd9190.jpg
    │   │   ├── BloodImage_00326_jpg.rf.115f475c35cfe69d12f7e7500b6f0504.jpg
    │   │   ├── BloodImage_00327_jpg.rf.421f505f875168babd109d8a26f2ec25.jpg
    │   │   ├── BloodImage_00330_jpg.rf.d899322812a2f5b1123bd333a034adc3.jpg
    │   │   ├── BloodImage_00332_jpg.rf.8b8041f67309058f264e1c3a10583eea.jpg
    │   │   ├── BloodImage_00333_jpg.rf.6afd664cfe1773c628ef63fdfd43f751.jpg
    │   │   ├── BloodImage_00338_jpg.rf.c0f0d2cd5a51f7fff518cef688e274af.jpg
    │   │   ├── BloodImage_00339_jpg.rf.f6a5fb7adfe24d938b07341f520c36e2.jpg
    │   │   ├── BloodImage_00340_jpg.rf.4bca1d6008a5698ffe8d12357e406e56.jpg
    │   │   ├── BloodImage_00341_jpg.rf.c6dccb482a4e3dc5c36260794df8771f.jpg
    │   │   ├── BloodImage_00342_jpg.rf.ecf9fcdc3df62dffe75a4b60198a30dd.jpg
    │   │   ├── BloodImage_00343_jpg.rf.307a02c2f64a99bb49154ce548c5d4a1.jpg
    │   │   ├── BloodImage_00345_jpg.rf.59971e759c69d5c761ebed6b19055d04.jpg
    │   │   ├── BloodImage_00346_jpg.rf.2f037f569f125bc4dc04070e0d71e592.jpg
    │   │   ├── BloodImage_00347_jpg.rf.b79a3ab2a3c8c62f9742e7798153047e.jpg
    │   │   ├── BloodImage_00349_jpg.rf.7a08496a67a69eeb3b6d260b3161c855.jpg
    │   │   ├── BloodImage_00351_jpg.rf.dacf4e0468f400e24030c912ba31bed2.jpg
    │   │   ├── BloodImage_00352_jpg.rf.51d52fe58f151051bd8d24007ab79338.jpg
    │   │   ├── BloodImage_00353_jpg.rf.cf5ed147e4f2675fbabbc9b0db750ecf.jpg
    │   │   ├── BloodImage_00354_jpg.rf.2339d6a97e76030873dea7459366967e.jpg
    │   │   ├── BloodImage_00356_jpg.rf.595525c2603ccc2bfc1e6e11e31f0c94.jpg
    │   │   ├── BloodImage_00357_jpg.rf.dfcf95336c225fdfa116fc9a13b8975b.jpg
    │   │   ├── BloodImage_00360_jpg.rf.f0983057278ee639d5194578d6a398c2.jpg
    │   │   ├── BloodImage_00361_jpg.rf.f0e1c1ab8c33d2be40b11d57b38d3ff1.jpg
    │   │   ├── BloodImage_00362_jpg.rf.7eb26575c314d8e51ff2a63f25c3213c.jpg
    │   │   ├── BloodImage_00365_jpg.rf.cd273f5fdd8232252180ddad4ff479e4.jpg
    │   │   ├── BloodImage_00366_jpg.rf.2c1d44c63137f3391eaa4f3cb2e10dcf.jpg
    │   │   ├── BloodImage_00367_jpg.rf.106032545117e1485f4b1dee535ab919.jpg
    │   │   ├── BloodImage_00368_jpg.rf.4e4a65ca533e139823601d539704a97f.jpg
    │   │   ├── BloodImage_00370_jpg.rf.fea04779dac40eda31c39d37bfd4a158.jpg
    │   │   ├── BloodImage_00372_jpg.rf.68038e2600568911d6fbfaac9614ca3c.jpg
    │   │   ├── BloodImage_00374_jpg.rf.efcd9616e49b6b6b2653fe6cb529d254.jpg
    │   │   ├── BloodImage_00375_jpg.rf.f0488f3844c9164dbe76ef782322d08d.jpg
    │   │   ├── BloodImage_00376_jpg.rf.81cf6993eebc05cadc7742bc508cdbe5.jpg
    │   │   ├── BloodImage_00378_jpg.rf.b660aa32201296c176d9e480d9fef52c.jpg
    │   │   ├── BloodImage_00379_jpg.rf.ffbd5df18884ad9c40fe678c9ee6cc6a.jpg
    │   │   ├── BloodImage_00381_jpg.rf.6e79077e93fc5bfbfddbb6f001f0542f.jpg
    │   │   ├── BloodImage_00382_jpg.rf.510c260a1701cb071eda23dcd73dd74a.jpg
    │   │   ├── BloodImage_00383_jpg.rf.49f36a689e33479923805330d18c27be.jpg
    │   │   ├── BloodImage_00387_jpg.rf.163ffb5cac6e736ad12f70ef0f929a4e.jpg
    │   │   ├── BloodImage_00388_jpg.rf.12add1a7f8073842ada61f56cbf15b31.jpg
    │   │   ├── BloodImage_00389_jpg.rf.8dbb82596b12d4884776e010b19e95d4.jpg
    │   │   ├── BloodImage_00390_jpg.rf.21fe48cc0ffd20fa6693b20aa235b2db.jpg
    │   │   ├── BloodImage_00391_jpg.rf.efb4e2fae1122eccfb96a89da9a59873.jpg
    │   │   ├── BloodImage_00393_jpg.rf.ab4b32eb995ec97db50c5301c2670ee5.jpg
    │   │   ├── BloodImage_00395_jpg.rf.f0014c982c64e4801bf7a8cd3e36d480.jpg
    │   │   ├── BloodImage_00396_jpg.rf.a95185dd1fa63430734448a6e9b69b50.jpg
    │   │   ├── BloodImage_00397_jpg.rf.b52311dfc95f514c1c6d841aab431403.jpg
    │   │   ├── BloodImage_00398_jpg.rf.46416c58102949fa8c92a23a2834e50a.jpg
    │   │   ├── BloodImage_00400_jpg.rf.f0a808bacaa11a4e5f21fad1c14edf2a.jpg
    │   │   ├── BloodImage_00405_jpg.rf.6e09ceadb7a1c87e63aa40a45a918d05.jpg
    │   │   ├── BloodImage_00407_jpg.rf.bfd06dc020ea6cf3df8ecc8809dd29b6.jpg
    │   │   ├── BloodImage_00408_jpg.rf.080ed677663fc4119c3618ad0a7a87ab.jpg
    │   │   └── BloodImage_00409_jpg.rf.07399e04e5384fb02b947ea025eb3cd0.jpg
    │   ├── labels
    │   │   ├── BloodImage_00268_jpg.rf.e1d8eccd1a77cadaeadebca867a24c62.txt
    │   │   ├── BloodImage_00269_jpg.rf.8475d6c8dead19457a63ae00dd84c924.txt
    │   │   ├── BloodImage_00271_jpg.rf.08085d620b04fabd323d2b1ed8a64f1d.txt
    │   │   ├── BloodImage_00279_jpg.rf.5ade4f1f61924eb07fb0a8169278d395.txt
    │   │   ├── BloodImage_00282_jpg.rf.d0067872fcfb4df4e5bb9cf783110825.txt
    │   │   ├── BloodImage_00283_jpg.rf.6263895fbbc2628f18e1a0ed61c74bd6.txt
    │   │   ├── BloodImage_00285_jpg.rf.05ee9fb43b10619cff770fa0c5d124f5.txt
    │   │   ├── BloodImage_00287_jpg.rf.b03ee5db05e8750c2d71cc5444f3905c.txt
    │   │   ├── BloodImage_00288_jpg.rf.3adee28ab6ed4a5420f851a6b3d275df.txt
    │   │   ├── BloodImage_00290_jpg.rf.870ab7d987c50d829a5f28c1a972d917.txt
    │   │   ├── BloodImage_00291_jpg.rf.92ed5a625cc17a671eae55fd48329f02.txt
    │   │   ├── BloodImage_00292_jpg.rf.469f85f31734b15c32cbd1a96e274673.txt
    │   │   ├── BloodImage_00293_jpg.rf.8d19f05e21730e57d74657b0ce9b950c.txt
    │   │   ├── BloodImage_00294_jpg.rf.8ea49a5b1990a964de2ce838ab4a9525.txt
    │   │   ├── BloodImage_00295_jpg.rf.b8ad2a56170ce42d667c25819b788c1d.txt
    │   │   ├── BloodImage_00299_jpg.rf.e8fa71a67d5eae37ed66b6b7976772c1.txt
    │   │   ├── BloodImage_00303_jpg.rf.96019c5afa192dbb914016f0243964fd.txt
    │   │   ├── BloodImage_00304_jpg.rf.d065dccc029aa497e69aac485f5c7ea9.txt
    │   │   ├── BloodImage_00305_jpg.rf.3be7c63f74a5b205817365ceececd3bc.txt
    │   │   ├── BloodImage_00307_jpg.rf.f3a9b5a5c544183e2035095fb497fb4c.txt
    │   │   ├── BloodImage_00308_jpg.rf.4598085d3090b176e16953ab1054aef8.txt
    │   │   ├── BloodImage_00310_jpg.rf.52e3296f7a0d9e655e36ce1ac367faff.txt
    │   │   ├── BloodImage_00311_jpg.rf.5b1c9c2ae20f4f304393360e4e72d7ac.txt
    │   │   ├── BloodImage_00312_jpg.rf.644485683ce0a8eadc17594a2b970697.txt
    │   │   ├── BloodImage_00313_jpg.rf.48fee7adf53e53903cefb3254bfc852e.txt
    │   │   ├── BloodImage_00314_jpg.rf.722b30c67ca8aef37b0a8001c53c12e7.txt
    │   │   ├── BloodImage_00317_jpg.rf.6397dd207ef8b48c5ece8f699bfec8e5.txt
    │   │   ├── BloodImage_00318_jpg.rf.ae41d28eabbf137f0f7c3629398e73fa.txt
    │   │   ├── BloodImage_00320_jpg.rf.b252995123ab52685fe98090382015f9.txt
    │   │   ├── BloodImage_00322_jpg.rf.f3faf31c0fe23b4f042343e7e33ccc18.txt
    │   │   ├── BloodImage_00323_jpg.rf.f4e0e3a72c74a8b196373bb12697e49e.txt
    │   │   ├── BloodImage_00324_jpg.rf.d2fa0cee237fa4698401213fd2dd9190.txt
    │   │   ├── BloodImage_00326_jpg.rf.115f475c35cfe69d12f7e7500b6f0504.txt
    │   │   ├── BloodImage_00327_jpg.rf.421f505f875168babd109d8a26f2ec25.txt
    │   │   ├── BloodImage_00330_jpg.rf.d899322812a2f5b1123bd333a034adc3.txt
    │   │   ├── BloodImage_00332_jpg.rf.8b8041f67309058f264e1c3a10583eea.txt
    │   │   ├── BloodImage_00333_jpg.rf.6afd664cfe1773c628ef63fdfd43f751.txt
    │   │   ├── BloodImage_00338_jpg.rf.c0f0d2cd5a51f7fff518cef688e274af.txt
    │   │   ├── BloodImage_00339_jpg.rf.f6a5fb7adfe24d938b07341f520c36e2.txt
    │   │   ├── BloodImage_00340_jpg.rf.4bca1d6008a5698ffe8d12357e406e56.txt
    │   │   ├── BloodImage_00341_jpg.rf.c6dccb482a4e3dc5c36260794df8771f.txt
    │   │   ├── BloodImage_00342_jpg.rf.ecf9fcdc3df62dffe75a4b60198a30dd.txt
    │   │   ├── BloodImage_00343_jpg.rf.307a02c2f64a99bb49154ce548c5d4a1.txt
    │   │   ├── BloodImage_00345_jpg.rf.59971e759c69d5c761ebed6b19055d04.txt
    │   │   ├── BloodImage_00346_jpg.rf.2f037f569f125bc4dc04070e0d71e592.txt
    │   │   ├── BloodImage_00347_jpg.rf.b79a3ab2a3c8c62f9742e7798153047e.txt
    │   │   ├── BloodImage_00349_jpg.rf.7a08496a67a69eeb3b6d260b3161c855.txt
    │   │   ├── BloodImage_00351_jpg.rf.dacf4e0468f400e24030c912ba31bed2.txt
    │   │   ├── BloodImage_00352_jpg.rf.51d52fe58f151051bd8d24007ab79338.txt
    │   │   ├── BloodImage_00353_jpg.rf.cf5ed147e4f2675fbabbc9b0db750ecf.txt
    │   │   ├── BloodImage_00354_jpg.rf.2339d6a97e76030873dea7459366967e.txt
    │   │   ├── BloodImage_00356_jpg.rf.595525c2603ccc2bfc1e6e11e31f0c94.txt
    │   │   ├── BloodImage_00357_jpg.rf.dfcf95336c225fdfa116fc9a13b8975b.txt
    │   │   ├── BloodImage_00360_jpg.rf.f0983057278ee639d5194578d6a398c2.txt
    │   │   ├── BloodImage_00361_jpg.rf.f0e1c1ab8c33d2be40b11d57b38d3ff1.txt
    │   │   ├── BloodImage_00362_jpg.rf.7eb26575c314d8e51ff2a63f25c3213c.txt
    │   │   ├── BloodImage_00365_jpg.rf.cd273f5fdd8232252180ddad4ff479e4.txt
    │   │   ├── BloodImage_00366_jpg.rf.2c1d44c63137f3391eaa4f3cb2e10dcf.txt
    │   │   ├── BloodImage_00367_jpg.rf.106032545117e1485f4b1dee535ab919.txt
    │   │   ├── BloodImage_00368_jpg.rf.4e4a65ca533e139823601d539704a97f.txt
    │   │   ├── BloodImage_00370_jpg.rf.fea04779dac40eda31c39d37bfd4a158.txt
    │   │   ├── BloodImage_00372_jpg.rf.68038e2600568911d6fbfaac9614ca3c.txt
    │   │   ├── BloodImage_00374_jpg.rf.efcd9616e49b6b6b2653fe6cb529d254.txt
    │   │   ├── BloodImage_00375_jpg.rf.f0488f3844c9164dbe76ef782322d08d.txt
    │   │   ├── BloodImage_00376_jpg.rf.81cf6993eebc05cadc7742bc508cdbe5.txt
    │   │   ├── BloodImage_00378_jpg.rf.b660aa32201296c176d9e480d9fef52c.txt
    │   │   ├── BloodImage_00379_jpg.rf.ffbd5df18884ad9c40fe678c9ee6cc6a.txt
    │   │   ├── BloodImage_00381_jpg.rf.6e79077e93fc5bfbfddbb6f001f0542f.txt
    │   │   ├── BloodImage_00382_jpg.rf.510c260a1701cb071eda23dcd73dd74a.txt
    │   │   ├── BloodImage_00383_jpg.rf.49f36a689e33479923805330d18c27be.txt
    │   │   ├── BloodImage_00387_jpg.rf.163ffb5cac6e736ad12f70ef0f929a4e.txt
    │   │   ├── BloodImage_00388_jpg.rf.12add1a7f8073842ada61f56cbf15b31.txt
    │   │   ├── BloodImage_00389_jpg.rf.8dbb82596b12d4884776e010b19e95d4.txt
    │   │   ├── BloodImage_00390_jpg.rf.21fe48cc0ffd20fa6693b20aa235b2db.txt
    │   │   ├── BloodImage_00391_jpg.rf.efb4e2fae1122eccfb96a89da9a59873.txt
    │   │   ├── BloodImage_00393_jpg.rf.ab4b32eb995ec97db50c5301c2670ee5.txt
    │   │   ├── BloodImage_00395_jpg.rf.f0014c982c64e4801bf7a8cd3e36d480.txt
    │   │   ├── BloodImage_00396_jpg.rf.a95185dd1fa63430734448a6e9b69b50.txt
    │   │   ├── BloodImage_00397_jpg.rf.b52311dfc95f514c1c6d841aab431403.txt
    │   │   ├── BloodImage_00398_jpg.rf.46416c58102949fa8c92a23a2834e50a.txt
    │   │   ├── BloodImage_00400_jpg.rf.f0a808bacaa11a4e5f21fad1c14edf2a.txt
    │   │   ├── BloodImage_00405_jpg.rf.6e09ceadb7a1c87e63aa40a45a918d05.txt
    │   │   ├── BloodImage_00407_jpg.rf.bfd06dc020ea6cf3df8ecc8809dd29b6.txt
    │   │   ├── BloodImage_00408_jpg.rf.080ed677663fc4119c3618ad0a7a87ab.txt
    │   │   └── BloodImage_00409_jpg.rf.07399e04e5384fb02b947ea025eb3cd0.txt
    │   └── labels.cache
    └── valid
        ├── images
        │   ├── BloodImage_00274_jpg.rf.c139abd1792e0ffe07cb82e02a9e031e.jpg
        │   ├── BloodImage_00276_jpg.rf.ff5ab3a615d8dc715314b4cf6a0ff6ce.jpg
        │   ├── BloodImage_00277_jpg.rf.e03e8b0cb7ba4887ccf226faffa039ea.jpg
        │   ├── BloodImage_00281_jpg.rf.7f42dc6a75ba1e4fbcd0b2a2f5301f13.jpg
        │   ├── BloodImage_00296_jpg.rf.629c7d5fba915ebdf8e3e087647e7934.jpg
        │   ├── BloodImage_00297_jpg.rf.a04aa31cd3db26908e42d426aef24c03.jpg
        │   ├── BloodImage_00298_jpg.rf.d0aa119c4f394aed0060f52d5a6c3904.jpg
        │   ├── BloodImage_00300_jpg.rf.9bda3f2dc304d4557d781294e12f2594.jpg
        │   ├── BloodImage_00309_jpg.rf.bb892b3140c84836cc095e49e79e0943.jpg
        │   ├── BloodImage_00315_jpg.rf.2474ef08d570f6ce69c5987c05fc83e0.jpg
        │   ├── BloodImage_00319_jpg.rf.cbe3e8f5785a021e9ef6464545ea7816.jpg
        │   ├── BloodImage_00331_jpg.rf.3090709d0964f61d92c48b94aff80084.jpg
        │   ├── BloodImage_00335_jpg.rf.814c79ad38295791ef5d915d0e5cffc4.jpg
        │   ├── BloodImage_00344_jpg.rf.34e5227c60620fad8ff23aaa5e5c69ae.jpg
        │   ├── BloodImage_00348_jpg.rf.97ec8c703be56a9f36c004e9178ccdfc.jpg
        │   ├── BloodImage_00355_jpg.rf.dced7a1203b71b721d22be4617fc96f9.jpg
        │   ├── BloodImage_00364_jpg.rf.0125c8f2f2754216d734618e33fda9b1.jpg
        │   ├── BloodImage_00371_jpg.rf.73ded480b7b0a0ac05f22e90f4f478ab.jpg
        │   ├── BloodImage_00377_jpg.rf.38d0fadf00d8b2355cc8228ff41e0faa.jpg
        │   ├── BloodImage_00384_jpg.rf.989105a4bc0c4b0710457d310d62221b.jpg
        │   ├── BloodImage_00392_jpg.rf.d6c7f1baf550a2d709a1f353cf0021f8.jpg
        │   ├── BloodImage_00402_jpg.rf.cd6309adfd39b106d88cbb1517820371.jpg
        │   ├── BloodImage_00403_jpg.rf.8e73a82ca74b00311385217c70b33b12.jpg
        │   ├── BloodImage_00404_jpg.rf.c25423833af21b0a1cd2cef70a12bce5.jpg
        │   └── BloodImage_00410_jpg.rf.ebc1cae591fa39c4465095853a64b1f7.jpg
        ├── labels
        │   ├── BloodImage_00274_jpg.rf.c139abd1792e0ffe07cb82e02a9e031e.txt
        │   ├── BloodImage_00276_jpg.rf.ff5ab3a615d8dc715314b4cf6a0ff6ce.txt
        │   ├── BloodImage_00277_jpg.rf.e03e8b0cb7ba4887ccf226faffa039ea.txt
        │   ├── BloodImage_00281_jpg.rf.7f42dc6a75ba1e4fbcd0b2a2f5301f13.txt
        │   ├── BloodImage_00296_jpg.rf.629c7d5fba915ebdf8e3e087647e7934.txt
        │   ├── BloodImage_00297_jpg.rf.a04aa31cd3db26908e42d426aef24c03.txt
        │   ├── BloodImage_00298_jpg.rf.d0aa119c4f394aed0060f52d5a6c3904.txt
        │   ├── BloodImage_00300_jpg.rf.9bda3f2dc304d4557d781294e12f2594.txt
        │   ├── BloodImage_00309_jpg.rf.bb892b3140c84836cc095e49e79e0943.txt
        │   ├── BloodImage_00315_jpg.rf.2474ef08d570f6ce69c5987c05fc83e0.txt
        │   ├── BloodImage_00319_jpg.rf.cbe3e8f5785a021e9ef6464545ea7816.txt
        │   ├── BloodImage_00331_jpg.rf.3090709d0964f61d92c48b94aff80084.txt
        │   ├── BloodImage_00335_jpg.rf.814c79ad38295791ef5d915d0e5cffc4.txt
        │   ├── BloodImage_00344_jpg.rf.34e5227c60620fad8ff23aaa5e5c69ae.txt
        │   ├── BloodImage_00348_jpg.rf.97ec8c703be56a9f36c004e9178ccdfc.txt
        │   ├── BloodImage_00355_jpg.rf.dced7a1203b71b721d22be4617fc96f9.txt
        │   ├── BloodImage_00364_jpg.rf.0125c8f2f2754216d734618e33fda9b1.txt
        │   ├── BloodImage_00371_jpg.rf.73ded480b7b0a0ac05f22e90f4f478ab.txt
        │   ├── BloodImage_00377_jpg.rf.38d0fadf00d8b2355cc8228ff41e0faa.txt
        │   ├── BloodImage_00384_jpg.rf.989105a4bc0c4b0710457d310d62221b.txt
        │   ├── BloodImage_00392_jpg.rf.d6c7f1baf550a2d709a1f353cf0021f8.txt
        │   ├── BloodImage_00402_jpg.rf.cd6309adfd39b106d88cbb1517820371.txt
        │   ├── BloodImage_00403_jpg.rf.8e73a82ca74b00311385217c70b33b12.txt
        │   ├── BloodImage_00404_jpg.rf.c25423833af21b0a1cd2cef70a12bce5.txt
        │   └── BloodImage_00410_jpg.rf.ebc1cae591fa39c4465095853a64b1f7.txt
        └── labels.cache

31 directories, 740 files
```
