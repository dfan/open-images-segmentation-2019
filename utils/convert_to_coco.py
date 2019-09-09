# Produce JSON files for train/val/test splits in COCO format.

import argparse
import cv2
import numpy as np
import os
from pycocotools import _mask as coco_mask

def get_images_section(image_sourcefile, image_dir):
  images = []
  # Three fields: ImageID,LabelName,Confidence
  with open(image_sourcefile, 'r') as f:
    next(f) # skip header
    for line in f:
      im_id, label_name, _ = line.strip().split(',')
      im_filename = '{}.jpg'.format(im_id)
      im = cv2.imread(os.path.join(image_dir, im_filename))
      height = im.shape[0]
      width = im.shape[1]

      images.append(
        {
          'file_name': im_filename,
          'height': height,
          'width': width,
          'id': im_id
        }
      )

  print('Generated images section.')
  return images

def get_categories_section(category_sourcefile):
  categories = []
  category_dict = {}
  # Two fields: CategoryId,CategoryName (but no header in file)
  with open(category_sourcefile, 'r') as f:
    counter = 1
    for line in f:
      class_id, class_name = line.strip().split(',')
      categories.append(
        {
          'id': counter,
          'name': class_name,     # e.g. "doughnut"
          'original_id': class_id # e.g. "/m/0jy4k"
        }
      )
      category_dict[class_id] = counter
      counter += 1

  print('Generated categories section.')
  return categories, category_dict

def get_bbox_dict(bbox_sourcefile):
  # First get "isGroupOf" information. Key is ImageID, LabelName, XMin (rounded to 2 decimals) concatenated together
  bbox_group_dict = {}
  with open(bbox_sourcefile, 'r') as f:
    next(f) # Skip header line
    for line in f:
      image_id, label_name, x_min, _, _, _, is_group_of = line.strip().split(',')
      # Rounding is necessary because the decimal precision of the bounding box coordinates differs between files...
      key = image_id + '_' + label_name + '_' + str(round(float(x_min), 2))
      bbox_group_dict[key] = int(is_group_of)
  print('Generated IsGroupOf dictionary.')
  return bbox_group_dict

def get_annotations_section(bbox_sourcefile, mask_sourcefile, mask_dir, category_dict):
  # Key is ImageID, LabelName, XMin (rounded to 2 decimals) concatenated together
  bbox_group_dict = get_bbox_dict(bbox_sourcefile)
  annotations = []

  # Ten fields: MaskPath,ImageID,LabelName,BoxID,BoxXMin,BoxXMax,BoxYMin,BoxYMax,PredictedIoU,Clicks
  counter = 1
  with open(mask_sourcefile, 'r') as f:
    next(f) # Skip header line
    for line in f:
      mask_path, image_id, label_name, _, x_min, x_max, y_min, y_max, _, _ = line.strip().split(',')
      subset = mask_dir.split('/')[-1]
      mask_sub_folder = '{}-masks-{}'.format(subset, mask_path[0]) # e.g. validation-masks-a
      mask_path = os.path.join(mask_dir, mask_sub_folder, mask_path)
      mask = cv2.imread(mask_path, 0)
      height = mask.shape[0]
      width = mask.shape[1]
      category_id = category_dict[label_name] # Maps class/label name to integer

      # convert input mask to expected COCO API input --
      mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
      mask = mask.astype(np.uint8)
      mask = np.asfortranarray(mask)
      # RLE encode mask
      encoded_mask = coco_mask.encode(mask)
      counts = encoded_mask[0]['counts']

      bbox = coco_mask.toBbox(encoded_mask)[0] # [top left x position, top left y position, width, height]
      area = bbox[2] * bbox[3]

      # Figure out IsGroupOf value
      # Rounding is necessary because the decimal precision of the bounding box coordinates differs between files...
      dict_key = image_id + '_' + label_name + '_' + str(round(float(x_min), 2))
      is_group_of = bbox_group_dict[dict_key]

      annotations.append(
        {
          'segmentation': {
            'counts': counts,
            'size': [width, height]
          },
          'area': area,
          'iscrowd': is_group_of,
          'image_id': image_id,
          'bbox': bbox,
          'category_id': category_id,
          'original_category_id': label_name,
          'id': counter
        }
      )
      counter += 1

    print('Generated annotations section.')
    return annotations

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Convert OpenImages annotations into COCO format')
  parser.add_argument('-p', '--path', help='Path to OpenImages dataset', type=str, required=True)
  args = vars(parser.parse_args())

  root_dir = args['path']
  for subset in ['train', 'validation']:
    annotation_dir = os.path.join(root_dir, 'annotations')
    image_dir = os.path.join(root_dir, 'images', subset)
    mask_dir = os.path.join(root_dir, 'masks', subset)
    category_sourcefile = os.path.join(annotation_dir, 'challenge-2019-classes-description-segmentable.csv')
    image_sourcefile = os.path.join(annotation_dir, 'challenge-2019-{}-segmentation-imagelabels.csv'.format(subset))
    bbox_sourcefile = os.path.join(annotation_dir, 'challenge-2019-{}-segmentation-bbox.csv'.format(subset))
    mask_sourcefile = os.path.join(annotation_dir, 'challenge-2019-{}-segmentation-masks.csv'.format(subset))

    dataset = {}
    dataset['info'] = {}
    dataset['licenses'] = {}
    dataset['images'] = get_images_section(image_sourcefile, image_dir)
    categories, category_dict = get_categories_section(category_sourcefile)
    dataset['categories'] = categories
    dataset['annotations'] = get_annotations_section(bbox_sourcefile, mask_sourcefile, mask_dir, category_dict)

    output_file = os.path.join(annotation_dir, '{}_coco.json'.format(subset))
    with open(output_file, 'w') as f:
      json.dump(dataset, f)
    print('Finished generating {} COCO file.'.format(subset))
