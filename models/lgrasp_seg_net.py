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
from .lseg_module import LSegModule

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

def _make_srb_block(activation='relu'):
    srb = nn.Module()
    srb.head_block_pos_1 = depthwise_block(activation=activation)
    srb.head_block_cos_1 = depthwise_block(activation=activation)
    srb.head_block_sin_1 = depthwise_block(activation=activation)
    srb.head_block_width_1 = depthwise_block(activation=activation)

    srb.head_block_pos_2 = depthwise_block(activation=activation)
    srb.head_block_cos_2 = depthwise_block(activation=activation)
    srb.head_block_sin_2 = depthwise_block(activation=activation)
    srb.head_block_width_2 = depthwise_block(activation=activation)

    srb.head_block_pos_3 = depthwise_block(activation=activation)
    srb.head_block_cos_3 = depthwise_block(activation=activation)
    srb.head_block_sin_3 = depthwise_block(activation=activation)
    srb.head_block_width_3 = depthwise_block(activation=activation)

    srb.output_conv_pos = nn.Sequential(
        Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
    )
    srb.output_conv_cos = nn.Sequential(
        Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
    )
    srb.output_conv_sin = nn.Sequential(
        Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
    )
    srb.output_conv_width = nn.Sequential(
        Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
    )

    print(f"Weight: {srb.head_block_pos_1.depthwise.depthwise.weight}")

    return srb.cuda()

lseg_out = {}

def get_image_feature():
    def hook(model, input, output):
        lseg_out['image_features'] = output
    return hook

def make_lseg():
    module = LSegModule.load_from_checkpoint(
        checkpoint_path='checkpoints/demo_e200.ckpt',
        data_path="../datasets/",
        dataset='ade20k',
        backbone='clip_vitl16_384',
        aux=False,
        num_features=256,
        aux_weight=0,
        se_loss=False,
        se_weight=0,
        base_lr=0,
        batch_size=1,
        max_epochs=0,
        ignore_index=255,
        dropout=0.0,
        scale_inv=False,
        augment=False,
        no_batchnorm=False,
        widehead=True,
        widehead_hr=False,
        map_locatin="cpu",
        arch_option=0,
        block_depth=0,
        activation='lrelu',
    )

    # model
    lseg_net = module.net
    lseg_net = lseg_net.cuda()

    lseg_net.scratch.head1.register_forward_hook(get_image_feature())

    return lseg_net

class LGrasp(GraspModel): # Origin: LSeg(BaseModel)
    def __init__(
        self,
        features=256,
        backbone="clip_vitl16_384",
        readout="project",
        use_bn=False,
        **kwargs,
    ):
        super(LGrasp, self).__init__()

        if backbone in ["clipRN50x16_vitl16_384"]:
            self.out_c = 768
        else:
            self.out_c = 512

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()

        self.lseg_net = make_lseg()  

        self.srb = _make_srb_block(activation=kwargs["activation"])

        self.conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1).cuda()
        self.activation = nn.ReLU().cuda()
        

    def forward(self, x_in, prompt=''):
        # x[0] is a Tensor [batch_size, c_channels, h, w]
        # x[1] is a Tuple with `batch_size` prompts
        if isinstance(x_in, tuple):
            x = x_in[0]
            prompt = list(x_in[1])

        text_features, image_features = self.lseg_net(x, prompt, features_only=True)
        # return text_features, image_features

        text_features = text_features.unsqueeze(1)
        # print(f"Text features shape: {text_features.shape}") # [batch_size, 1, out_c] 

        imshape = image_features.shape
        image_features = image_features.permute(0,2,3,1).reshape(imshape[0], -1, self.out_c)
        # print(f"Image features shape (after reshaped and permute): {image_features.shape}") # [batch_size, H/2 * W/2, out_c] 

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logits_per_image = self.lseg_net.logit_scale * image_features.half() @ text_features.mT

        # print(f"Logits per image shape: {logits_per_image.shape}") # [batch_size, H/2 * W/2, 1]

        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2)
        print(out)
        # print(f"Out (before headblock) shape: {out.shape}") # [batch_size, 1, H/2, W/2]

        # Test
        out = self.conv2d(out)
        print(out)
        out = self.activation(out)

        return out, out
    
        out_pos = self.srb.head_block_pos_1(out)
        out_cos = self.srb.head_block_cos_1(out)
        out_sin = self.srb.head_block_sin_1(out)
        out_width = self.srb.head_block_width_1(out)

    
        out_pos = self.srb.head_block_pos_2(out_pos)
        out_cos = self.srb.head_block_cos_2(out_cos)
        out_sin = self.srb.head_block_sin_2(out_sin)
        out_width = self.srb.head_block_width_2(out_width)

        out_pos = self.srb.head_block_pos_3(out_pos)
        out_cos = self.srb.head_block_cos_3(out_cos)
        out_sin = self.srb.head_block_sin_3(out_sin)
        out_width = self.srb.head_block_width_3(out_width)

        # print(f"Out (after headblock) shape: {out.shape}") # [batch_size, 1, H/2, W/2]


        # Bilinear interpolation
        pos_output = self.srb.output_conv_pos(out_pos) # [batch_size, 1, H, W]
        cos_output = self.srb.output_conv_cos(out_cos)
        sin_output = self.srb.output_conv_sin(out_sin)
        width_output = self.srb.output_conv_width(out_width)

        # print(f"Out (after output_conv_pos) shape: {out.shape}") # [batch_size, 1, H, W]
    
        return pos_output, cos_output, sin_output, width_output

class LGraspNet(LGrasp):
    """Network for semantic segmentation."""
    def __init__(self, scale_factor=0.5, crop_size=480, **kwargs):

        kwargs["use_bn"] = True

        self.crop_size = crop_size
        self.scale_factor = scale_factor

        super().__init__(**kwargs)

    
        
    