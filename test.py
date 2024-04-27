import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from torchvision.io import read_image
import torch.nn as nn
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        image = torch.zeros([3, 224, 224], dtype=torch.float32)
        prompt = "Image ne"
        label = 0
        return (image, prompt), label
    

# dataset = CustomImageDataset()
# data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# for (image, prompt), label in data_loader:
#     print(type(image), type(prompt), type(label))
#     print(image.shape, prompt, label)

# c_dim = 3
# text_features = torch.randint(0, 100, size=(2, c_dim))
# image_features = torch.randint(0, 100, size=(2, c_dim, 5, 5))
# imshape = image_features.shape
# image_features = image_features.permute(0, 2, 3, 1).reshape(imshape[0], -1, c_dim)
# print(image_features.shape)
# print(image_features)


# logit = image_features @ text_features.mT
# print(logit.shape)
# logit = logit.view(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)
# print(logit.shape)

class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x

layer = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

x = torch.randn(3, 1, 5, 5)
print(x.shape)
print(x)
x = layer(x)
print(x.shape)
print(x)