import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import os

import cv2
import numpy as np


class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_list) // 2

    def __getitem__(self, idx):
        idx *= 2
        image_path1 = os.path.join(self.root_dir,  'set0', 'train', 't0')
        image_path2 = os.path.join(self.root_dir, 'set0', 'train', 't1')
        image1 = Image.open(image_path1).convert("RGB")
        image2 = Image.open(image_path2).convert("RGB")

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2


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


dataset_path = "pcd/set0/train/t0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

dataset = ChangeDetectionDataset(dataset_path, transform=transform)

data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

model = ChangeDetectionModel().to(device)

test_image1 = transform(Image.open("pcd/set0/train/t0/00000098.jpg").convert("RGB")).unsqueeze(0).to(device)
test_image2 = transform(Image.open("pcd/set0/train/t1/00000098.jpg").convert("RGB")).unsqueeze(0).to(device)

# model.load_state_dict(torch.load("pcd/trained_model_checkpoint.pth"))

model.eval()

# # Generate the change mask
# with torch.no_grad():
#     change_mask = model(test_image1, test_image2).squeeze().cpu().numpy()

# # Convert the change mask to black and white
# change_mask_bw = (change_mask > 0.5).astype("uint8") * 255
# change_mask_bw_inverted = 255 - change_mask_bw

# change_mask_image = Image.fromarray(change_mask_bw_inverted, mode="L")

# change_mask_image.save("pcd/change_mask.jpg")

with torch.no_grad():
    change_mask = model(test_image1, test_image2).squeeze().cpu().numpy()  

    #  we calculate the change quantification by taking the mean of all the values in the change mask tensor.

change_quantification = round(change_mask.mean(), 2)

print("The Amount of change is", change_quantification)
