# questions to ask 
# 1. How do I place all these classes in the mylightingmodule class?
# 2. How do I set the set the directory to the dataset
# 3. How do I run?


from typing import Any, Callable

from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from torch.nn import Module
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from functools import partial
from typing import Any, List, Optional
from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision.transforms import Compose, ToTensor, ToPILImage

import os
from os.path import join as pjoin, splitext as spt
from pathlib import Path

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)

class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)
    
class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )



# class Model(Module):
#     # Here you should put your model class
#         def __init__(self, in_channels: int, num_classes: int) -> None:
#             super().__init__(
#                 ASPP(in_channels, [12, 24, 36]),
#                 nn.Conv2d(256, 256, 3, padding=1, bias=False),
#                 nn.BatchNorm2d(256),
#                 nn.ReLU(),
#                 nn.Conv2d(256, num_classes, 1),
#             )
#     pass


class MyLightningModule(LightningModule):

    def __init__(
            self,
            model,
            loss: Callable = mse_loss,
            lr: float = 1e-3,
            *args: Any,
            **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.loss = loss
        self.lr = lr

    def training_step(self, batch, batch_idx):
        before, after, target = batch
        prediction = self.model(before, after)
        loss = self.loss(prediction, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        before, after, target = batch
        prediction = self.model(before, after)
        loss = self.loss(prediction, target)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        before, after, target = batch
        prediction = self.model(before, after)
        loss = self.loss(prediction, target)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        return optimizer


class MyDataset(Dataset):
    # Here you should put your dataset class
    def __init__(self, root, transforms=None):
        super(MyDataset, self).__init__()
        self.root = root
        self.gt, self.t0, self.t1 = self._init_data_list()
        if transforms is None:
            transforms = Compose([ToTensor()])
        self._transforms = transforms
        self._revert_transforms = Compose([ToPILImage()])
        self.name = ''
        self.num_classes = 2
        
    def _init_data_list(self):
        gt = []
        t0 = []
        t1 = []
        for file in os.listdir(os.path.join(self.root, 'mask')):
            if self._check_validness(file):
                idx = int(file.split('.')[0].split('_')[-1])
                # if self.rotation or idx == 0:
                gt.append(pjoin(self.root, 'mask', file))
                t0.append(pjoin(self.root, 't0', file))
                t1.append(pjoin(self.root, 't1', file))
        return gt, t0, t1

    def _check_validness(self, f):
        return any([i in spt(f)[1] for i in ['jpg']])

    def _pil_loader(self, path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def get_raw(self, index):
        fn_t0 = self.t0[index]
        fn_t1 = self.t1[index]
        fn_mask = self.gt[index]

        img_t0 = self._pil_loader(fn_t0)
        img_t1 = self._pil_loader(fn_t1)
        imgs = [img_t0, img_t1]

        mask = self._pil_loader(fn_mask).convert("L")
        return imgs, mask

    def __getitem__(self, index):
        imgs, mask = self.get_raw(index)
        if self._transforms is not None:
            mask = self._transforms(mask)
            imgs, = self._transforms(imgs)
        return imgs, mask

    def __len__(self):
        return len(self.gt)

    def get_mask_ratio(self):
        all_count = 0
        mask_count = 0
        for i in range(len(self.gt)):
            _, mask = self.get_raw(i)
            target = (F.to_tensor(mask) != 0).long()
            mask_count += target.sum()
            all_count += target.numel()
        mask_ratio = mask_count / float(all_count)
        background_ratio = (all_count - mask_count) / float(all_count)
        return [mask_ratio, background_ratio]
    



    def get_pil(self, imgs, mask, pred=None):
        assert self._revert_transforms is not None
        t0, t1 = self._revert_transforms(imgs.cpu())
        w, h = t0.size
        output = Image.new('RGB', (w * 2, h * 2))
        output.paste(t0)
        output.paste(t1, (w, 0))
        mask = F.to_pil_image(mask.cpu().float())
        output.paste(mask, (0, h))
        pred = F.to_pil_image(pred.cpu().float())
        output.paste(pred, (w, h))
        return output
    
    
    pass


class MyDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        # Here you should create the various splits depending on the ones you need
        # ideally you can have them already in separate folders
        self.train_data = MyDataset(
            os.path.join(self.data_dir,"train")
        ) 
        self.val_data = MyDataset(
            os.path.join(self.data_dir,"validation")
        ) 
        self.test_data = MyDataset(
            os.path.join(self.data_dir,"test")
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)


if __name__ == "__main__":
    model = MyLightningModule(
        DeepLabHead(in_channels=1, num_classes=1),  # This should be your actual model class. Check what these mean
    )
    datamodule = MyDataModule(
        'pcd/set0',  # This should be the path to your dataset folder
        batch_size=8, 
    ) 
    # Trainer has many parameters that may be useful for advanced training procedures
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints" , save_top_k=2, monitor="val_loss"
        )
    
    # trainer = Trainer(
    #     accelerator="cpu", callbacks=[checkpoint_callback]
    #     )
    
    trainer = Trainer(
        min_epochs=1, 
        max_epochs=1000,
        accelerator="gpu" # Use 'gpu' to use the GPU acceleration if available. Here we go.
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test()