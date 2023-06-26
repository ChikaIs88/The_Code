import cv2
import numpy as np

import os
from os.path import join as pjoin, splitext as spt
import os
import torch
import numpy as np
import PIL
from PIL import Image

from os.path import join as pjoin, splitext as spt

from dataset import CDDataset, get_transforms
import transforms as T

# import path_config as Data_path
  
# # reading the image data from desired directory
# img = cv2.imread("chess5.png")
# cv2.imshow('Image',img)
  
# # counting the number of pixels
# number_of_white_pix = np.sum(img == 255)

# # find total number of pixels
# tot_pix = img.size

# white_pix_percentage = number_of_white_pix / tot_pix * 100;

# # number_of_black_pix = np.sum(img == 0)

# print('The Amount of change is:', white_pix_percentage)
# # print('Number of black pixels:', number_of_black_pix)



class PCD_CV(CDDataset):
    # all images are 256x256
    # object: white(255)  ->  True
    #                  toTensor 
    def __init__(self, root, rotation=True, transforms=None):
        super(PCD_CV, self).__init__(root, transforms)
        self.root = root
        self.rotation = rotation
        # self.gt, self.t0, self.t1 = self._init_data_list()
        self.gt = self._init_data_list()
        self._transforms = transforms

    def _init_data_list(self):
        gt = []
        # t0 = []
        # t1 = []
        for file in os.listdir(os.path.join(self.root, 'mask')):
            if self._check_validness(file):
                idx = int(file.split('.')[0])
                if self.rotation or idx % 4 == 0:
                    gt.append(pjoin(self.root, 'mask', file))
                    # t0.append(pjoin(self.root, 't0', file.replace('png', 'jpg')))
                    # t1.append(pjoin(self.root, 't1', file.replace('png', 'jpg')))
        return gt, #t0, t1


