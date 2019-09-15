import argparse
import os
import torch
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transforms

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from open_images_test_dataset import OpenImagesTestDataset

from torchvision.transforms import functional as F
import random

import cv2
import base64
import numpy as np
from pycocotools import _mask as coco_mask
import typing as t
import zlib

class Resize(object):
    def __call__(self, image):
        image = F.resize(image, (1024, 1024))
        return image

class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target

def is_data_parallel(model):
    for key in model.state_dict():
        if 'module.' in key:
            return True
    return False

def convert_mask_to_format(mask):
  mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
  mask_to_encode = mask_to_encode.astype(np.uint8)
  mask_to_encode = np.asfortranarray(mask_to_encode)

  # RLE encode mask --
  encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

  # compress and base64 encoding --
  binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
  base64_str = base64.b64encode(binary_str)
  return base64_str

def get_categories(category_sourcefile='/home/dfan/datasets/open_images_segmentation/annotations/challenge-2019-classes-description-segmentable.csv'):
  category_dict = {}
  # Two fields: CategoryId,CategoryName (but no header in file)
  with open(category_sourcefile, 'r') as f:
    counter = 1
    for line in f:
      class_id, class_name = line.strip().split(',')
      category_dict[counter] = class_id
      counter += 1

  return category_dict

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--config-file", help="path to config file", type=str, required=True)
  parser.add_argument("--weights-file", help="path to trained weights file", type=str, required=True)
  parser.add_argument("--output-file", help="path to output file", type=str, required=True)
  args = vars(parser.parse_args())
  config_file = args['config_file']
  weights_file = args['weights_file']
  output_file = args['output_file']

  cfg.merge_from_file(config_file)
  cfg.freeze()
  device = cfg.MODEL.DEVICE
  
  model = build_detection_model(cfg)
  is_data_parallel = is_data_parallel(model)
  if is_data_parallel:
      model = nn.DataParallel(model)
  
  model_dict = model.state_dict()
  pretrained_dict = torch.load(weights_file)
  for key in pretrained_dict['model']:
     model_dict[key] = pretrained_dict['model'][key] 
  model.load_state_dict(model_dict)
  model.to(device)
  model.eval()

  test_process_steps = transforms.Compose([
    Resize(),
    transforms.ToTensor(),
    Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255
    )
  ])
  dataset = OpenImagesTestDataset(test_process_steps)
  test_params = {'batch_size': 1, 'num_workers': 3}
  dataloader = data.DataLoader(dataset, **test_params)

  category_dict = get_categories()

  f_out = open(output_file, 'w')
  header = 'ImageID,ImageWidth,ImageHeight,PredictionString'
  f_out.write(header + '\n')

  with torch.no_grad():
    for counter, (image, filename, shape) in enumerate(dataloader):
      image = image.to(device)
      image = image[0]
      output = model(image)[0]
      orig_height, orig_width = shape

      filename = filename[0]
      for i in range(len(output)):
          image_id = filename.split('/')[-1].replace('.jpg', '')
          sigmoid_mask = output.get_field('mask')[i] # 1 x 28 x 28 (MaskRCNN produces 28x28 which is then resized to ROI)
          sigmoid_mask = sigmoid_mask.permute(1,2,0).cpu().numpy()
          sigmoid_mask = cv2.resize(sigmoid_mask, (orig_height, orig_width), interpolation=cv2.INTER_LINEAR)
          mask = sigmoid_mask > 0.5
          mask = cv2.resize(np.float32(mask), (orig_width, orig_height))

          formatted_mask = convert_mask_to_format(mask)
          
          label = output.get_field('labels')[i].item()
          pred_class = category_dict[label]
          score = output.get_field('scores')[i].item()
          
          prediction_string = '{} {} {}'.format(pred_class, score, str(formatted_mask, 'utf-8'))
          f_out.write('{},{},{},{}\n'.format(image_id, orig_width.item(), orig_height.item(), prediction_string))
        
      if (counter + 1)  % 1000 == 0:
         print('Processed {} test images.'.format(counter + 1))
  f_out.close()

