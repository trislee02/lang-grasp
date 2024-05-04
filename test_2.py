import torch
import torch.nn as nn

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

x = torch.randint(0, 9, (2, 1, 3, 3))
print(x)
y = torch.randint(0, 9, (2, 1, 3, 3))
print(y)
print(x * y)