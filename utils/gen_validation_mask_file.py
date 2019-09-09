# Download the segmentations and segmentation file from https://storage.googleapis.com/openimages/web/challenge2019_downloads.html
# Segmentation file is called "challenge-2019-validation-segmentation-masks.csv".
import argparse
import cv2
import numpy as np
import os
from pycocotools import _mask as coco_mask

# Inputs:
#   seg_file: name of a CSV file containing MaskPath, ImageID, LabelName, BoxID, BoxXMin, BoxXMax, BoxYMin, BoxYMax,
#               PredictedIoU, Clicks.\
#   bbox_file: name of a CSV file containing ImageID, LabelName, XMin, XMax, YMin, YMax, IsGroupOf
#   output_file: name of the new CSV file. Contains ImageID,LabelName,ImageWidth,ImageHeight,XMin,YMin,XMax,YMax,IsGroupOf,Mask
#               Where Mask is MS COCO RLE encoding of a binary mask stored in .png file.
#
# Return:
#   N/A. Produces a new CSV file named "output_file" according to above specifications.
def gen_new_validation_csv(seg_file, bbox_file, output_file):
    # First get "isGroupOf" information. Key is ImageID, LabelName, XMin (rounded to 2 decimals) concatenated together
    bbox_group_dict = {}
    with open(bbox_file, 'r') as f:
        next(f) # Skip header line
        for line in f:
            image_id, label_name, x_min, _, _, _, is_group_of = line.strip().split(',')
            # Rounding is necessary because the decimal precision of the bounding box coordinates differs between files...
            key = image_id + '_' + label_name + '_' + str(round(float(x_min), 2))
            bbox_group_dict[key] = is_group_of
    print('Generated IsGroupOf dictionary.')

    f_out = open(output_file, 'w')
    header = 'ImageID,LabelName,ImageWidth,ImageHeight,XMin,YMin,XMax,YMax,Score,Mask'
    f_out.write(header + '\n')

    counter = 0
    with open(seg_file, 'r') as f:
        next(f) # Skip header line
        for line in f:
            mask_name, image_id, label_name, _, x_min, y_min, x_max, y_max, _, _ = line.strip().split(',')
            # Form full mask and image path. Assuming data is structured per the readme.
            mask_folder_prefix = mask_name[0]
            mask_path = '/home/dfan/datasets/open_images_segmentation/masks/validation/validation-masks-{}/{}'.format(mask_folder_prefix, mask_name) # e.g. validation-masks-a/blahblah.png
            image_path = '/home/dfan/datasets/open_images_segmentation/images/validation/{}.jpg'.format(image_id)

            # Get image width and height
            im = cv2.imread(image_path)
            height = im.shape[0]
            width = im.shape[1]

            # Figure out IsGroupOf value
            # Rounding is necessary because the decimal precision of the bounding box coordinates differs between files...
            dict_key = image_id + '_' + label_name + '_' + str(round(float(x_min), 2))
            is_group_of = bbox_group_dict[dict_key]

            # convert input mask to expected COCO API input --
            mask = cv2.imread(mask_path, 0)
            mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
            mask = mask.astype(np.uint8)
            mask = np.asfortranarray(mask)

            # RLE encode mask
            encoded_mask = str(coco_mask.encode(mask)[0]["counts"], 'utf-8')

            # Write ImageID,LabelName,ImageWidth,ImageHeight,XMin,YMin,XMax,YMax,IsGroupOf,Mask
            f_out.write(image_id + ',' + label_name + ',' + str(width) + ',' + str(height) + ',' + x_min + ',' + y_min + ',' + x_max + ',' + y_max + ',' + is_group_of + ',' + encoded_mask + '\n')

            counter += 1
            if counter % 1000 == 0:
                print('Processed {} masks.'.format(counter))

    f_out.close()
    print('Successfully combined validation mask information into CSV format!')

# E.g: python gen_validation_mask_file.py -s challenge-2019-validation-segmentation-masks.csv -b challenge-2019-validation-segmentation-bbox.csv -o processed-validation-segmentation-masks.csv
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--segmentation', type=str, required=True, help='Specify the segmentation mask file from website. Called "challenge-2019-validation-segmentation-masks.csv" by default.')
    parser.add_argument('-b', '--bbox', type=str, required=True, help='Specify the bounding box file from website. Sole purpose is to get the "IsGroupOf" field. Called "challenge-2019-validation-segmentation-bbox.csv" by default.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Specify the output filename for new csv.')
    args = vars(parser.parse_args())

    root_dir = '/home/dfan/datasets/open_images_segmentation/'
    seg_file = os.path.join(root_dir, args['segmentation'])
    bbox_file = os.path.join(root_dir, args['bbox'])
    output_file = os.path.join(root_dir, args['output'])
    gen_new_validation_csv(seg_file, bbox_file, output_file)
