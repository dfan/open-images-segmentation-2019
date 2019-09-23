import numpy as np
import os
from glob import glob
from PIL import Image
import torch
from torch.utils import data
import cv2

class OpenImagesTestDataset(data.dataset.Dataset):
  def __init__(self, transform, test_dir='/home/dfan/datasets/open_images_segmentation/images/test'):
    file_descriptor = os.path.join(test_dir, '*.jpg')
    files = glob(file_descriptor)

    img_names = []
    for image in files:
      img_names.append(image)

    self.img_names = img_names
    self.transform = transform

  def __getitem__(self, index):
    im = Image.open(self.img_names[index])
    orig_width, orig_height = im.size
    
    if im.getbands()[0] == 'L':
        im = im.convert('RGB')

    if self.transform:
        im, padding = self.transform(im)

    return im, self.img_names[index], (orig_height, orig_width), padding

  def __len__(self):
    return len(self.img_names)

if __name__ == '__main__':
  dataset = OpenImagesTestDataset()
