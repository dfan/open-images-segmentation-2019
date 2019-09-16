import argparse
import cv2
from glob import glob
from tqdm import tqdm
import mmap
import os

def get_num_lines(file_path):
  fp = open(file_path, "r+")
  buf = mmap.mmap(fp.fileno(), 0)
  lines = 0
  while buf.readline():
    lines += 1
  return lines

def get_all_im_ids(image_dir='/home/dfan/datasets/open_images_segmentation/images/test'):
  file_descriptor = os.path.join(image_dir, '*.jpg')
  image_ids = []
  for filename in glob(file_descriptor):
    im_id = filename.split('/')[-1].replace('.jpg', '')
    image_ids.append(im_id)
  return image_ids

def fix_csv(input_file, output_file):
  lines_to_write = {} # map ID to prediction string
  # Merge individual lines per image into one line
  with open(input_file, 'r') as f:
    next(f) # Skip header
    for line in tqdm(f, total=get_num_lines(input_file)):
      image_id, width, height, prediction_string = line.strip().split(',')
      if image_id not in lines_to_write:
        lines_to_write[image_id] = '{},{},{},{}'.format(image_id, width, height, prediction_string)
      else:
        old_string = lines_to_write[image_id]
        if len(old_string.split(' ')) < 12:
          lines_to_write[image_id] = '{} {}'.format(old_string, prediction_string)
  
  # Take care of images with no predictions
  all_im_ids = get_all_im_ids()
  remaining_ids = list(set(all_im_ids) - set(lines_to_write.keys()))

  print('{} images without predictions'.format(len(remaining_ids)))
  for im_id in remaining_ids:
    image = cv2.imread(os.path.join('/home/dfan/datasets/open_images_segmentation/images/test', im_id + '.jpg'))
    lines_to_write[im_id] = '{},{},{},'.format(im_id, image.shape[1], image.shape[0])

  with open(output_file, 'w') as f:
    f.write('ImageID,ImageWidth,ImageHeight,PredictionString\n')
    for image_id in lines_to_write:
      f.write(lines_to_write[image_id] + '\n')

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--input-file', type=str, required=True, help='The input CSV to fix')
  parser.add_argument('--output-file', type=str, required=True, help='The destination CSV')
  args = vars(parser.parse_args())

  input_file = args['input_file']
  output_file = args['output_file']
  assert(input_file != output_file)

  fix_csv(input_file, output_file)
