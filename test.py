# An example of using SmoothL1Loss of Pytorch

import torch
import torch.nn as nn

# Define the input and target
input = torch.randn(2, 3, requires_grad=True)
target = torch.randn(2, 3)

# Define the loss function
loss = nn.SmoothL1Loss(reduction='sum')

# Compute the loss
output = loss(input, target)
output.backward()

# Print the output
print(input)
print("====================================")
print(target)
print("====================================")
print((input - target).abs())
print("====================================")
print((input - target)**2)
print("====================================")
print(output)
print(input.grad)

