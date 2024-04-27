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

import clip
text_features = clip.tokenize(["a photo of a dog", "a photo of a cat"])
print(text_features.shape)
print(text_features)

text_features = text_features.to('cuda')
clip_pretrained, _ = clip.load("ViT-B/32", device='cuda', jit=False)
text_features = clip_pretrained.encode_text(text_features)
text_features = text_features.unsqueeze(1)
print(text_features.shape)
print(text_features)