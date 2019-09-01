# open-images-segmentation-2019
ICCV 2019 Open Images instance segmentation challenge.

## Installation and Setup
### Downloading Dataset Files
1. Download all files from the [Open Images challenge website](https://storage.googleapis.com/openimages/web/challenge2019_downloads.html) under the "Instance Segmentation track annotations" section.
I put all dataset files under `/data`. Note that the training and validation image-level labels are called "challenge-2019-...-segmentation-labels.csv". I renamed them to "challenge-2019-...-segmentation-imagelabels.csv" for clarity.

```
└───data
│   │   challenge-2019-classes-description-segmentable.csv
|   |   challenge-2019-label300-segmentable-hierarchy.json
|   |   challenge-2019-train-segmentation-bbox.csv
|   |   challenge-2019-train-segmentation-imagelabels.csv
|   |   challenge-2019-train-segmentation-masks.csv
|   |   challenge-2019-validation-segmentation-bbox.csv
|   |   challenge-2019-validation-segmentation-imagelabels.csv
|   |   challenge-2019-validation-segmentation-masks.csv
│   └───images
│       └───test
│       └───train
|       └───validation
│   └───masks
│       └───train
|           └───train-masks-0
|           └───...
|           └───train-masks-f
|       └───validation
|           └───validation-masks-0
|           └───...
|           └───validation-masks-f
```

2. Setup evaluation files.

Make sure you follow the instructions [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).
