import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import os
import glob

import cv2
import numpy as np


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_paths = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))
        self.transform = transform

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_paths)


class ChangeDetectionModel(nn.Module):
    def __init__(self):
        super(ChangeDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.sigmoid(self.conv3(x))
        return x


    
dataset1 = ImageDataset(image_folder='pcd/set0/train/t0', transform=Compose([ToTensor()]))
dataset2 = ImageDataset(image_folder='pcd/set0/train/t1', transform=Compose([ToTensor()]))
gt_dataset = ImageDataset(image_folder='pcd/set0/train/mask', transform=Compose([ToTensor()]))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# dataset = ChangeDetectionDataset(dataset_path, transform=transform)

data_loader = DataLoader(dataset1, batch_size=1, shuffle=True)

model = ChangeDetectionModel().to(device)

model.eval()

# # Generate the change mask
# with torch.no_grad():
#     change_mask = model(test_image1, test_image2).squeeze().cpu().numpy()

# # Convert the change mask to black and white
# change_mask_bw = (change_mask > 0.5).astype("uint8") * 255
# change_mask_bw_inverted = 255 - change_mask_bw

# change_mask_image = Image.fromarray(change_mask_bw_inverted, mode="L")

# change_mask_image.save("pcd/change_mask.jpg")
# image1 = dataset1[0].unsqueeze(0).convert("RGB").unsqueeze(0).to(device)  # Add batch dimension
# image2 = dataset2[0].unsqueeze(0).convert("RGB").unsqueeze(0).to(device)  # Add batch dimension

image1 = dataset1[0].unsqueeze(0).to(device)  # Add batch dimension
image2 = dataset2[0].unsqueeze(0).to(device)  # Add batch dimension
gt = gt_dataset[0].unsqueeze(0).to(device) 


if torch.equal(image1, image2) is False:


    with torch.no_grad():
        change_mask = model(image1, image2).squeeze().cpu().numpy()  

        #  we calculate the change quantification by taking the mean of all the values in the change mask tensor.


    # change_mask = change_mask / np.max(change_mask)

    # Clamp the values between 0 and 1 (optional)
    # change_mask = np.clip(change_mask, 0, 1)

    amount_of_change =   change_mask.mean()

    gt_mask =   torch.mean(gt)

    change_q =   round(((amount_of_change / 255.0 ) * 100), 3 )  # round(change_mask.mean(), 2)
    gt_value = torch.round(((gt_mask / 255.0 ) * 100), decimals=3) 

    # change_detected = torch.clamp(gt_value, 0, 1)


    print("The Amount of change is", change_q, "and the gt is", gt_value )
else:
    print("These are the same images")