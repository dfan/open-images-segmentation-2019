HIERARCHY_FILE=/home/dfan/Projects/kaggle/open-images-segmentation-2019/data/challenge-2019-label300-segmentable-hierarchy
BOUNDING_BOXES=/home/dfan/Projects/kaggle/open-images-segmentation-2019/data/challenge-2019-validation-segmentation-bbox
IMAGE_LABELS=/home/dfan/Projects/kaggle/open-images-segmentation-2019/data/challenge-2019-validation-segmentation-imagelabels
INSTANCE_SEGMENTATIONS=/home/dfan/Projects/kaggle/open-images-segmentation-2019/data/processed-validation-segmentation-masks

python object_detection/dataset_tools/oid_hierarchical_labels_expansion.py \
    --json_hierarchy_file=${HIERARCHY_FILE}.json \
    --input_annotations=${BOUNDING_BOXES}.csv \
    --output_annotations=${BOUNDING_BOXES}_expanded.csv \
    --annotation_type=1

python object_detection/dataset_tools/oid_hierarchical_labels_expansion.py \
    --json_hierarchy_file=${HIERARCHY_FILE}.json \
    --input_annotations=${IMAGE_LABELS}.csv \
    --output_annotations=${IMAGE_LABELS}_expanded.csv \
    --annotation_type=2

python object_detection/dataset_tools/oid_hierarchical_labels_expansion.py \
    --json_hierarchy_file=${HIERARCHY_FILE}.json \
    --input_annotations=${INSTANCE_SEGMENTATIONS}.csv \
    --output_annotations=${INSTANCE_SEGMENTATIONS}_expanded.csv \
    --annotation_type=1
