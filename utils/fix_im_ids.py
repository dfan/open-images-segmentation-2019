import argparse
import json
import os

def fix_image_paths(json_filename):
  with open(json_filename, 'r') as f:
    dataset = json.load(f)

  # Build dictionary of image IDs
  counter = 1
  im_to_id = {}
  for im_dict in dataset['images']:
    im_to_id[im_dict['id']] = counter
    counter += 1

  for im_dict in dataset['images']:
    old_id = im_dict['id']
    im_dict['id'] = im_to_id[old_id]

  for anno_dict in dataset['annotations']:
    old_id = anno_dict['image_id']
    anno_dict['image_id'] = im_to_id[old_id]

  with open(json_filename, 'w') as f:
    json.dump(dataset, f)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Add directory to OpenImage image paths if not there.')
  parser.add_argument('-f', '--file', help='Name of JSON file', type=str, required=True)
  args = vars(parser.parse_args())
  filename = args['file']

  root_dir = '/home/dfan/datasets/open_images_segmentation'
  annotation_dir = os.path.join(root_dir, 'annotations')
  filename = os.path.join(annotation_dir, filename)
  fix_image_paths(filename)
