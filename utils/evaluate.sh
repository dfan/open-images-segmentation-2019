BOUNDING_BOXES=/home/dfan/datasets/open_images_segmentation/challenge-2019-validation-segmentation-bbox
IMAGE_LABELS=/home/dfan/datasets/open_images_segmentation/challenge-2019-validation-segmentation-imagelabels
INPUT_PREDICTIONS=/home/dfan/Projects/kaggle/open-images-segmentation-2019/sample_validation_submission.csv
INSTANCE_SEGMENTATIONS=/home/dfan/datasets/open_images_segmentation/processed-validation-segmentation-masks
OUTPUT_METRICS=/home/dfan/Projects/kaggle/open-images-segmentation-2019/output/evaluation_output.txt

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

python object_detection/metrics/oid_challenge_evaluation.py \
    --input_annotations_boxes=${BOUNDING_BOXES}_expanded.csv \
    --input_annotations_labels=${IMAGE_LABELS}_expanded.csv \
    --input_class_labelmap=object_detection/data/oid_object_detection_challenge_500_label_map.pbtxt \
    --input_predictions=${INPUT_PREDICTIONS} \
    --input_annotations_segm=${INSTANCE_SEGMENTATIONS}_expanded.csv
    --output_metrics=${OUTPUT_METRICS} \
