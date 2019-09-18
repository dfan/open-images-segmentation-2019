import argparse
import json
import os

def fix_image_paths(json_filename, mask_filename):
  with open(json_filename, 'r') as f:
    dataset = json.load(f)

  # Track old image IDs
  old_im_ids = {} # int to filename
  for im_dict in dataset['images']:
    old_im_ids[im_dict['id']] = im_dict['file_name']
  
  # Create proper image IDs
  new_im_ids = {} # filename to int
  with open(mask_filename, 'r') as f:
    counter = 1
    next(f) # Skip header
    for line in f:
      # MaskPath,ImageID,LabelName,BoxID,BoxXMin,BoxXMax,BoxYMin,BoxYMax,PredictedIoU,Clicks
      _, image_id, _, _, _, _, _, _, _, _ = line.strip().split(',')
      image_filename = image_id + '.jpg'
      if image_filename not in new_im_ids:
        new_im_ids[image_filename] = counter
        counter += 1
  
  # Keep entries with annotations and fix ids
  new_images = []
  for im_dict in dataset['images']:
    # Make sure image exists in mask annotation file
    if im_dict['file_name'] in new_im_ids:
      im_dict['id'] = new_im_ids[im_dict['file_name']]
      new_images.append(im_dict)
          
  dataset['images'] = new_images

  # Fix Annotations
  for anno_dict in dataset['annotations']:
    file_name = old_im_ids[anno_dict['image_id']]
    new_id = new_im_ids[file_name]
    anno_dict['image_id'] = new_id

  with open(json_filename, 'w') as f:
    json.dump(dataset, f)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Add directory to OpenImage image paths if not there.')
  parser.add_argument('-f', '--file', help='Name of JSON file', type=str, required=True)
  parser.add_argument('-s', '--subset', help='train or validation', type=str, required=True)
  args = vars(parser.parse_args())
  filename = args['file']
  subset = args['subset']

  assert(subset == 'train' or subset == 'validation')
  mask_file = '/home/dfan/datasets/open_images_segmentation/annotations/challenge-2019-{}-segmentation-masks.csv'.format(subset)
  root_dir = '/home/dfan/datasets/open_images_segmentation'
  annotation_dir = os.path.join(root_dir, 'annotations')
  json_file = os.path.join(annotation_dir, filename)
  fix_image_paths(json_file, mask_file)
