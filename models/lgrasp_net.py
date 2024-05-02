import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lgrasp_blocks import FeatureFusionBlock, Interpolate, _make_encoder, FeatureFusionBlock_custom, forward_vit, _make_grcnn
import clip
import numpy as np
import pandas as pd
import os

from inference.models.grasp_model import GraspModel

class depthwise_clipseg_conv(nn.Module):
    def __init__(self):
        super(depthwise_clipseg_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    
    def depthwise_clipseg(self, x, channels):
        x = torch.cat([self.depthwise(x[:, i].unsqueeze(1)) for i in range(channels)], dim=1)
        return x

    def forward(self, x):
        channels = x.shape[1]
        out = self.depthwise_clipseg(x, channels)
        return out


class depthwise_conv(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # support for 4D tensor with NCHW
        C, H, W = x.shape[1:]
        x = x.reshape(-1, 1, H, W)
        x = self.depthwise(x)
        x = x.view(-1, C, H, W)
        return x


class depthwise_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(depthwise_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x, act=True):
        x = self.depthwise(x)
        if act:
            x = self.activation(x)
        return x


class bottleneck_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(bottleneck_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()


    def forward(self, x, act=True):
        sum_layer = x.max(dim=1, keepdim=True)[0]
        x = self.depthwise(x)
        x = x + sum_layer
        if act:
            x = self.activation(x)
        return x

def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        activation=nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

class LGrasp(GraspModel): # Origin: LSeg(BaseModel)
    def __init__(
        self,
        features=256,
        backbone="clip_vitl16_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs,
    ):
        super(LGrasp, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "clip_vitl16_384": [5, 11, 17, 23],
            "clipRN50x16_vitl16_384": [5, 11, 17, 23],
            "clip_vitb32_384": [2, 5, 8, 11],
        }

        # Instantiate backbone and reassemble blocks
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        if backbone in ["clipRN50x16_vitl16_384"]:
            self.out_c = 768
        else:
            self.out_c = 512
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        self.arch_option = kwargs["arch_option"]
        self.block_depth = kwargs['block_depth']
        if self.block_depth > 0:
            self.scratch.head_block_pos_1 = depthwise_block(activation=kwargs["activation"])
            self.scratch.head_block_cos_1 = depthwise_block(activation=kwargs["activation"])
            self.scratch.head_block_sin_1 = depthwise_block(activation=kwargs["activation"])
            self.scratch.head_block_width_1 = depthwise_block(activation=kwargs["activation"])

            self.scratch.head_block_pos_2 = depthwise_block(activation=kwargs["activation"])
            self.scratch.head_block_cos_2 = depthwise_block(activation=kwargs["activation"])
            self.scratch.head_block_sin_2 = depthwise_block(activation=kwargs["activation"])
            self.scratch.head_block_width_2 = depthwise_block(activation=kwargs["activation"])

            self.scratch.head_block_pos_3 = depthwise_block(activation=kwargs["activation"])
            self.scratch.head_block_cos_3 = depthwise_block(activation=kwargs["activation"])
            self.scratch.head_block_sin_3 = depthwise_block(activation=kwargs["activation"])
            self.scratch.head_block_width_3 = depthwise_block(activation=kwargs["activation"])

        self.scratch.output_conv_pos = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )
        self.scratch.output_conv_cos = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )
        self.scratch.output_conv_sin = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )
        self.scratch.output_conv_width = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

    def forward(self, x_in, prompt=''):
        # Check if x is of type tuple, i.e. from a dataloader
        # x[0] is a Tensor [batch_size, c, h, w], x[1] is a Tuple with `batch_size` prompts
        if isinstance(x_in, tuple):
            x = x_in[0]
            prompt = list(x_in[1])

        if prompt == '':
            text = clip.tokenize([''])   
        else:
            text = clip.tokenize(prompt)  

        # print(f"Text (after tokenize) length: {len(text)}") # = batch_size
        # print(f"Image shape: {x.shape}") # [batch_size, 3, H, W] # HxW is the input size

        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        text = text.to(x.device)
        self.logit_scale = self.logit_scale.to(x.device)
        # Encode text features
        text_features = self.clip_pretrained.encode_text(text)
        text_features = text_features.unsqueeze(1)
        # print(f"Text features shape: {text_features.shape}") # [batch_size, 1, out_c] 

        # Get image features
        image_features = self.scratch.head1(path_1)
        # print(f"Image features shape: {image_features.shape}") # [batch_size, out_c, H/2, W/2]

        out_image_features = image_features.detach().clone()

        imshape = image_features.shape
        image_features = image_features.permute(0,2,3,1).reshape(imshape[0], -1, self.out_c)
        # print(f"Image features shape (after reshaped and permute): {image_features.shape}") # [batch_size, H/2 * W/2, out_c] 

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # print(f"Logit scale shape: {self.logit_scale.shape}") # []

        logits_per_image = self.logit_scale * image_features.half() @ text_features.mT

        # print(f"Logits per image shape: {logits_per_image.shape}") # [batch_size, H/2 * W/2, 1]

        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2)

        out_logits_per_image = out.detach().clone()

        # print(f"Out (before headblock) shape: {out.shape}") # [batch_size, 1, H/2, W/2]



        out_pos = self.scratch.head_block_pos_1(out)
        out_cos = self.scratch.head_block_cos_1(out)
        out_sin = self.scratch.head_block_sin_1(out)
        out_width = self.scratch.head_block_width_1(out)

        if self.block_depth > 1:
            out_pos = self.scratch.head_block_pos_2(out_pos)
            out_cos = self.scratch.head_block_cos_2(out_cos)
            out_sin = self.scratch.head_block_sin_2(out_sin)
            out_width = self.scratch.head_block_width_2(out_width)

        if self.block_depth > 2:
            out_pos = self.scratch.head_block_pos_3(out_pos)
            out_cos = self.scratch.head_block_cos_3(out_cos)
            out_sin = self.scratch.head_block_sin_3(out_sin)
            out_width = self.scratch.head_block_width_3(out_width)

        # print(f"Out (after headblock) shape: {out.shape}") # [batch_size, 1, H/2, W/2]

        # Bilinear interpolation
        pos_output = self.scratch.output_conv_pos(out_pos) # [batch_size, 1, H, W]
        cos_output = self.scratch.output_conv_cos(out_cos)
        sin_output = self.scratch.output_conv_sin(out_sin)
        width_output = self.scratch.output_conv_width(out_width)

        # print(f"Out (after output_conv_pos) shape: {out.shape}") # [batch_size, 1, H, W]

        return pos_output, cos_output, sin_output, width_output, out_image_features, out_logits_per_image

class LGraspNet(LGrasp):
    """Network for semantic segmentation."""
    def __init__(self, path=None, scale_factor=0.5, crop_size=480, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256
        kwargs["use_bn"] = True

        self.crop_size = crop_size
        self.scale_factor = scale_factor

        super().__init__(**kwargs)

        if path is not None:
            self.load(path)
    
        
    