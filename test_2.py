import torch
import torch.nn as nn

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

x = torch.randn(1, 1, 3, 3)
out = conv2d(x)
print(conv2d.weight)
print(x)
print(out)