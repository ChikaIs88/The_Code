import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from pytorch_lightning import LightningModule, Trainer
from PIL import Image
import os

import cv2
import numpy as np
import glob


class ChangeDetectionModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32*32*3, 1)  # Assuming input image size is 32x32
        
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)  # Concatenate the two images along the channel dimension
        x = self.conv(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x1, x2 = batch
        prediction = self(x1, x2)
        loss = nn.MSELoss()(prediction, torch.zeros_like(prediction))
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

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

# Assuming you have two folders 'image_folder1' and 'image_folder2' containing the respective images
dataset1 = ImageDataset(image_folder='pcd/set0/train/t0', transform=Compose([ToTensor()]))
dataset2 = ImageDataset(image_folder='pcd/set0/train/t1', transform=Compose([ToTensor()]))

data_loader = DataLoader(dataset1, batch_size=1)

model = ChangeDetectionModel()

trainer = Trainer(max_epochs=10)
trainer.fit(model, data_loader)

# Assuming you want to compare the first image from dataset1 with the first image from dataset2
image1 = dataset1[0].unsqueeze(0)  # Add batch dimension
image2 = dataset2[0].unsqueeze(0)  # Add batch dimension

model.eval()
with torch.no_grad():
    change_amount = model(image1, image2).item()
    print(f"Amount of change: {change_amount}")
