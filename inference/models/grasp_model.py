import torch
import torch.nn as nn
import torch.nn.functional as F


class GraspModel(nn.Module):
    """
    An abstract model for grasp network in a common format.
    """

    def __init__(self):
        super(GraspModel, self).__init__()

    def forward(self, x_in):
        raise NotImplementedError()

    def compute_loss(self, xc, yc):
        # print(f"Input shape: {xc.shape}") # [1, 3, 224, 224]
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        # print(f"Prediction shape: {pos_pred.shape}, {cos_pred.shape}, {sin_pred.shape}, {width_pred.shape}") # [1, 1, 224, 224], [1, 1, 224, 224], [1, 1, 224, 224], [1, 1, 224, 224]
        # print(f"Ground truth shape: {y_pos.shape}, {y_cos.shape}, {y_sin.shape}, {y_width.shape}") # [1, 1, 224, 224], [1, 1, 224, 224], [1, 1, 224, 224], [1, 1, 224, 224]

        pos_pred = pos_pred * (y_pos + 1e-3)
        cos_pred = cos_pred * (y_cos + 1e-3)
        sin_pred = sin_pred * (y_sin + 1e-3)
        width_pred = width_pred * (y_width + 1e-3)

        p_loss = F.smooth_l1_loss(pos_pred, y_pos * (y_pos + 1e-3), reduction='mean')
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos * (y_cos + 1e-3), reduction='mean')
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin * (y_sin + 1e-3), reduction='mean')
        width_loss = F.smooth_l1_loss(width_pred, y_width * (y_width + 1e-3), reduction='mean') 

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            },
            'gt': {
                'pos': y_pos,
                'cos': y_cos,
                'sin': y_sin,
                'width': y_width
            }
        }

    def predict(self, xc):
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)
        return {
            'pos': pos_pred,
            'cos': cos_pred,
            'sin': sin_pred,
            'width': width_pred
        }


class ResidualBlock(nn.Module):
    """
    A residual block with dropout option
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in
