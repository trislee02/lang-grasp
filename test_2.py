import torch
import torch.nn as nn

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

y = torch.tensor([[0, 1, 0], 
                  [2.1, 0, 3], 
                  [0, 4, 0]]).float()
print(y)

# Create a mask with the same shape as `y` with 1 at element greater than 0 and 0 otherwise
mask = (y != 0).float()
print(mask)