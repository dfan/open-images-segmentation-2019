# open-images-segmentation-2019
ICCV 2019 Open Images instance segmentation challenge. I won 49th place (bronze) with a public AP of 0.0941 by training MaskRCNN ResNet50-FPN with batch size 4 for 1.3M iterations (roughly 6 epochs; 848,512 training images had annotations). This took roughly a week on my RTX 2080 Ti. After the competition, I let it train to 1.8M iterations and would have received 43rd place (public AP of 0.1496).

### Installation and Setup
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

2. Make a conda environment and install the dependencies using the provided `environment.yml` file.

3. Create the COCO-formatted instance segmentation annotations using `utils/convert_to_coco.py`. I followed [this](http://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch) as a guide when writing the script.

4. Download `e2e_mask_rcnn_R_50_FPN_1x.pth` from the [MaskRCNN model zoo](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/MODEL_ZOO.md) or whichever COCO-pretrained model you want. I opted for a smaller model so that I could train with a larger batch size. Using the ResNet101 backbone, I could only train with a batch size of 1. With ResNet50, I could train with a batch size of 4 which led to more stable loss values. Then trim the roi_head layers since OpenImages has a different number of object classes from COCO. Example is at `maskrcnn/tools/convert_coco_model_to_openimages.py`.

5. Follow the instructions from the [official MaskRCNN repo](https://github.com/facebookresearch/maskrcnn-benchmark) and build the code.

6. Make changes to code to accomodate OpenImages. Here is a non-exhaustive list of changes I made to the MaskRCNN code base. Related thread that I found useful is [here](https://github.com/facebookresearch/maskrcnn-benchmark/issues/521).
- Make Dataset class for OpenImages
- Add dataset paths to `paths_catalog.py`
- Add logic to the evaluation file for OpenImages
- Write a new yaml config file with the desired hyperparameters (my setup is at `maskrcnn/configs/e2e_faster_rcnn_R_50_FPN_1x_openimages.yaml`). Make sure ROI_HEAD.NUM_CLASSES parameter is overwritten with the number of classes in OpenImages (300 + 1 for background = 301). Also make sure that you've specified the path to your pretrained weights.

7. Setup evaluation files for calculating validation loss.
Make sure you follow the instructions [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md). You will have to make modifications depending on whether you do evaluation on the Kaggle formatted predicions CSV or follow Tensorflow's format. Relevant thread [here](https://www.kaggle.com/c/open-images-2019-instance-segmentation/discussion/105478) (which I contributed to!).

8. Train, validate, evaluate, iterate.
`python tools/train_net.py --config-file "configs/e2e_faster_rcnn_R_50_FPN_1x_openimages.yaml" SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.01 SOLVER.MAX_ITER 1800000 SOLVER.STEPS "(1200000, 1600000)" TEST.IMS_PER_BATCH 1 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 4000`
`python tools/test_net.py --config configs/e2e_faster_rcnn_R_50_FPN_1x_openimages.yaml TEST.IMS_PER_BATCH 1`
  - This runs a different evaluation function from the Kaggle competition which uses the Tensorflow API (step 7). But in practice I found these numbers to be similar so you can decide what to do. Setting up Tensorflow evaluation was not the easiest.
`python tools/make_predictions.py --config-file configs/e2e_faster_rcnn_R_50_FPN_1x_openimages.yaml --weights-file model_final.pth`

### Challenges Encountered and Lessons Learned
I could have gotten further in the competition if I got the code and dataset setup earlier, but I kept making mistakes and running into issues. Almost all of the mistakes I made were dataset related. It took about 3 weeks of part-time work for me to completely get my pipeline working properly, which left less than 2 weeks to train.

- Dataset-related
  - For a while I failed to realize that not all images in `challenge-2019-train-segmentation-imagelabels.csv` have annotations. So then I tried generating annotations using the ImageIds in `challenge-2019-train-segmentation-masks.csv`. But I forgot that many images have multiple annotations, so my ImageIds weren't correct.
    - In spite of this, I was in a hurry to train and started some experiments. Of course, my validation accuracy was terrible. I attributed this to not training for very long, but should have taken more time to visualize my annotations and ensure they were actually correct.
  - I initially decided to write my own functions for formatting the mask annotations in COCO format as a learning exercise, but something was wrong with my code. After using [pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools), my issues went away. While I learned a lot in the process, reinventing the wheel is not a good idea for delivering results under time pressure.
  - My annotation file was huge because I initially decided to store the entire binary mask instead of just the polygon of the mask boundary. The latter is standard. This made loading the annotations extremely slow. After switching to polygon format, I could load the training annotation file in less than a minute (compared to 30+ minutes)
- Model/code-related
  - The MaskRCNN repository from Facebook is not the easiest to work with. I felt that it was somewhat overengineered and highly tailored to the COCO dataset. It took me a long time to read through the code and understand it enough to modify for OpenImages, and there was not any documentation on how to support additional datasets. I figured out most on my own and supplemented the holes by digging through the GitHub issues section.
    - In retrospect, I could have saved a lot of time by working with a PyTorch port of Matterport's MaskRCNN implementation. I originally opted for the official Facebook repository since Matterport reports a lower final trained accuracy, which suggested that their implementation is not entirely faithful to the paper. But the time I would have saved by getting Matterport's code to work might have outweighed the small drop in AP points, since I would have been able to train for longer.
  - My evaluation script was wrong for a while, because I wasn't resizing and fitting the output mask into its bounding box. I had assumed that MaskRCNN would output predictions for the entire image, but it actually just outputs predictions for the relevant image region. I would have realized this earlier if I took the time to properly visualize my results.
- Miscellaneous
  - I could not figure out what PyTorch and CUDA versions were most compatible with the latest version of the repo. After struggling a bit and getting weird warnings during training, I finally found a combination that worked. Torch 1.1.0 and CUDA 10.1 worked for me.
  - Kaggle kept timing out on my predictions file, and I realized that it was because I wasn't limiting the number of detections per image. It seems like something on the order of 7-8 detections per image is small enough.
  - I submitted my final predictions file using a confidence threshold of 30%, but afterward found that 20% was even better. Maybe 10% is even better. My AP is much lower for higher thresholds since many images will have no output predictions. I suspect lower confidence thresholds work better since many instances of classes in OpenImages are small and have low representation.

### Final Reflections
My validation AP at the time of submission was about 0.17 while my leaderboard score was about 0.09. After training to 1.8M iterations post-competition, my validation AP was 0.21 and leaderboard AP was 0.14. One thing that I should have done is balance the classes, since OpenImages is highly imbalanced. An easy way to do this is to modify the dataloader to weight the less-represented classes more, and the more-represented classes less.

In general, I felt this competition was more a challenge of time and hardware, rather than algorithms or domain knowledge. The fact that only 200 people entered the competition reflects how hardware intensive it was. The dataset post-processing takes almost 600 GB on my machine. A lot of Kaggle users probably don't have that much free disk space. Computation-wise, I had a difficult time training with an appropriate batch size even on a 2080 Ti, which is a top-of-the-line consumer GPU. With the ResNet101-FPN backbone, I could train with a whopping batch size of 1. After downsizing to the ResNet50-FPN backbone, I could then train with a batch size of 4. But this is far smaller than the batch size that Facebook used to get its state-of-art results on COCO; it used 8 Nvidia V100 GPUs ($50,000???) in parallel for a total batch size of 16. It's often the case that published results from state-of-art models are difficult to replicate for the average user, due to differences in available compute.

