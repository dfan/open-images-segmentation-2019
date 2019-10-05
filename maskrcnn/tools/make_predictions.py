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
from PIL import ImageOps

import cv2
import base64
import numpy as np
from pycocotools import _mask as coco_mask
from tqdm import tqdm
import zlib

class Resize(object):
    def __call__(self, image):
        w,h = image.size
       
        # Resize longer side to 1024, preserve aspect ratio
        if w < h:
          image = F.resize(image, (1024, int(round(w * 1024 / h))))
        else:
          image = F.resize(image, (int(round(h * 1024 / w)), 1024))
        
        top_pad = (1024 - image.size[1]) // 2 # PIL image size is width x height
        bottom_pad = 1024 - image.size[1] - top_pad
        left_pad = (1024 - image.size[0]) // 2
        right_pad = 1024 - image.size[0] - left_pad
        padding = (left_pad, top_pad, right_pad, bottom_pad)
        
        image = ImageOps.expand(image, padding)
        return image, padding

class ToTensor(object):
    # Wrapper so image padding can get passed
    def __call__(self, tuple_input):
        image, padding = tuple_input
        if padding is None:
            return transforms.ToTensor()(image)
        return transforms.ToTensor()(image), padding

class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, tuple_input):
        image, padding = tuple_input
        if self.to_bgr255:
            image = image[[2, 1, 0], :, :] * 255.0
        image = F.normalize(image, mean=self.mean, std=self.std)
        if padding is None:
            return image
        return image, padding

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
  args = vars(parser.parse_args())
  config_file = args['config_file']
  weights_file = args['weights_file']

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
    ToTensor(),
     Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255
    )
  ])
  dataset = OpenImagesTestDataset(test_process_steps)
  test_params = {'batch_size': 1, 'num_workers': 3, 'pin_memory': False}
  dataloader = data.DataLoader(dataset, **test_params)

  category_dict = get_categories()
  header = 'ImageID,ImageWidth,ImageHeight,PredictionString'

  f_out_p20 = open('predictions_p20.csv', 'w')
  f_out_p30 = open('predictions_p30.csv', 'w')
  f_out_p40 = open('predictions_p40.csv', 'w')
  f_out_p50 = open('predictions_p50.csv', 'w')
  f_out_p60 = open('predictions_p60.csv', 'w')
  f_out_p70 = open('predictions_p70.csv', 'w')

  f_out_p20.write(header + '\n')
  f_out_p30.write(header + '\n')
  f_out_p40.write(header + '\n')
  f_out_p50.write(header + '\n')
  f_out_p60.write(header + '\n')
  f_out_p70.write(header + '\n')

  with torch.no_grad():
    for counter, (image, filename, shape, padding) in tqdm(enumerate(dataloader), total=len(dataloader)):
      image = image.to(device)
      image = image[0]
      output = model(image)[0]
      orig_height, orig_width = shape

      filename = filename[0]
      image_id = filename.split('/')[-1].replace('.jpg', '')
      base_fields = '{},{},{},'.format(image_id, orig_width.item(), orig_height.item())
      f_out_p20.write(base_fields)
      f_out_p30.write(base_fields)
      f_out_p40.write(base_fields)
      f_out_p50.write(base_fields)
      f_out_p60.write(base_fields)
      f_out_p70.write(base_fields)

      all_predictions_p20 = []
      all_predictions_p30 = []
      all_predictions_p40 = []
      all_predictions_p50 = []
      all_predictions_p60 = []
      all_predictions_p70 = []

      boxes = output.bbox
      masks = output.get_field('mask')
      labels = output.get_field('labels').tolist()
      scores = output.get_field('scores').tolist()      

      # Sort scores and get top 7
      sorted_scores = np.array(scores)
      top_indexes = np.argsort(-sorted_scores)
      end = min(7, len(sorted_scores))
      top_indexes = top_indexes[:end]

      for i in top_indexes:
        if scores[i] > 0.2:
          x_min, y_min, x_max, y_max = [int(round(x.item())) for x in boxes[i]]
          roi_mask = masks[i] # 1 x 28 x 28 (MaskRCNN produces 28x28 which is then resized to ROI)
          roi_mask = roi_mask.permute(1,2,0).cpu().numpy()
          roi_mask = cv2.resize(roi_mask, (x_max-x_min+1, y_max-y_min+1), interpolation=cv2.INTER_LINEAR)
          sigmoid_mask = np.zeros((1024, 1024))
          sigmoid_mask[y_min:y_max+1, x_min:x_max+1] = roi_mask
          
          # Recover from padding
          left_pad, top_pad, right_pad, bottom_pad = [x.item() for x in padding]
          # cv2.imwrite('output/{}_old.png'.format(image_id), image.permute(1,2,0).cpu().numpy())
          # cv2.imwrite('output/{}_mask_old_{}.png'.format(image_id, round(scores[i], 5)), (sigmoid_mask > 0.5) * 255)
          sigmoid_mask = sigmoid_mask[top_pad:1024-bottom_pad, left_pad:1024-right_pad]
          new_im = image.permute(1,2,0).cpu().numpy()[top_pad:1024-bottom_pad, left_pad:1024-right_pad]
          # cv2.imwrite('output/{}_new.png'.format(image_id), new_im)
          # cv2.imwrite('output/{}_mask_new_{}.png'.format(image_id, round(scores[i], 5)), (sigmoid_mask > 0.5)*255)

          sigmoid_mask = cv2.resize(sigmoid_mask, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
          assert(sigmoid_mask.shape == (orig_height, orig_width))
          mask = sigmoid_mask > 0.5

          formatted_mask = convert_mask_to_format(mask).decode()
          
          label = labels[i]
          pred_class = category_dict[label]
          score = round(scores[i], 5)
          
          all_predictions_p20.extend([pred_class, str(score), formatted_mask])
          if scores[i] > 0.3:
            all_predictions_p30.extend([pred_class, str(score), formatted_mask])
          if scores[i] > 0.4:
            all_predictions_p40.extend([pred_class, str(score), formatted_mask])
          if scores[i] > 0.5:
            all_predictions_p50.extend([pred_class, str(score), formatted_mask])
          if scores[i] > 0.6:
            all_predictions_p60.extend([pred_class, str(score), formatted_mask])
          if scores[i] > 0.7:
            all_predictions_p70.extend([pred_class, str(score), formatted_mask])

      prediction_string_p20 = ' '.join(all_predictions_p20)
      prediction_string_p30 = ' '.join(all_predictions_p30)
      prediction_string_p40 = ' '.join(all_predictions_p40)
      prediction_string_p50 = ' '.join(all_predictions_p50)
      prediction_string_p60 = ' '.join(all_predictions_p60)
      prediction_string_p70 = ' '.join(all_predictions_p70)
      #print(image_id, orig_width, orig_height, prediction_string)
      f_out_p20.write(prediction_string_p20 + '\n')
      f_out_p30.write(prediction_string_p30 + '\n')
      f_out_p40.write(prediction_string_p40 + '\n')
      f_out_p50.write(prediction_string_p50 + '\n')
      f_out_p60.write(prediction_string_p60 + '\n')
      f_out_p70.write(prediction_string_p70 + '\n')

  f_out_p20.close()
  f_out_p30.close()
  f_out_p40.close()
  f_out_p50.close()
  f_out_p60.close()
  f_out_p70.close()

