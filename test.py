import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import os


class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_list) // 2

    def __getitem__(self, idx):
        idx *= 2
        image_path1 = os.path.join(self.root_dir, self.image_list[idx])
        image_path2 = os.path.join(self.root_dir, self.image_list[idx + 1])
        image1 = Image.open(image_path1).convert("RGB")
        image2 = Image.open(image_path2).convert("RGB")

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2


class ChangeDetectionModel(nn.Module):
    def __init__(self):
        super(ChangeDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
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


def train_model(model, train_loader, num_epochs, device):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images1, images2 in train_loader:
            images1 = images1.to(device)
            images2 = images2.to(device)

            optimizer.zero_grad()

            outputs = model(images1, images2)
            loss = criterion(outputs, torch.ones_like(outputs))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")


def generate_change_mask(model, image1, image2, device):
    model.to(device)
    model.eval()

    image1 = image1.unsqueeze(0).to(device)
    image2 = image2.unsqueeze(0).to(device)

    with torch.no_grad():
        change_mask = model(image1, image2)

    return change_mask.squeeze().cpu().numpy()


# Set the path to your dataset
dataset_path = "pcd/set0/train"

# Set the device (cuda or cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformations
transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Create the dataset
dataset = ChangeDetectionDataset(dataset_path, transform=transform)

# Create the data loader
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Create the model
model = ChangeDetectionModel()

# Train the model
train_model(model, data_loader, num_epochs=10, device=device)

# Example usage:
# Load two test images
test_image1 = transform(Image.open("pcd/set0/train/t0/00000033.jpg").convert("RGB"))
test_image2 = transform(Image.open("pcd/set0/train/t1/00000033.jpg").convert("RGB"))

# Generate the change mask
change_mask = generate_change_mask(model, test_image1, test_image2, device)

# Save the change mask as an image
change_mask_image = Image.fromarray((change_mask * 255).astype("uint8"), mode="L")
change_mask_image.save("pcd/change_mask.jpg")
