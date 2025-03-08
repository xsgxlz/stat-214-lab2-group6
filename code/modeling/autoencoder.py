from dataclasses import dataclass
from typing import List

import torch
from torch import nn
from torchvision.models.efficientnet import EfficientNet, MBConvConfig, FusedMBConvConfig, Conv2dNormActivation

class EfficientNetEncoder(EfficientNet):
    def __init__(
        self,
        inverted_residual_setting,
        dropout,
        input_channels=8,
        stochastic_depth_prob=0.1,
        norm_layer=None,
        last_channel=None,
    ):
        super().__init__(
            inverted_residual_setting=inverted_residual_setting,
            dropout=dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            num_classes=1,  # Dummy value, ignored
            norm_layer=norm_layer,
            last_channel=last_channel,
        )
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        self.features[0] = Conv2dNormActivation(
            input_channels,
            firstconv_output_channels,
            kernel_size=3,
            stride=1,
            norm_layer=norm_layer,
            activation_layer=nn.SiLU,
        )
        del self.avgpool
        del self.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

class EfficientNetDecoder(nn.Module):
    def __init__(self, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Stage 1: Upsample from 96x96x64 to 192x192x32
        self.stage1 = nn.Sequential(
            # Transposed conv for upsampling
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1, output_padding=0
            ),  # 96x96 -> 192x192
            # Depthwise conv (EfficientNet-style)
            Conv2dNormActivation(
                32, 32, kernel_size=3, stride=1, groups=32,  # Depthwise
                norm_layer=norm_layer, activation_layer=nn.SiLU
            ),
            # Projection (no expansion, just refine)
            Conv2dNormActivation(
                32, 32, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            ),
        )

        # Stage 2: Upsample from 192x192x32 to 384x384x8
        self.stage2 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 16, kernel_size=4, stride=2, padding=1, output_padding=0
            ),  # 192x192 -> 384x384
            Conv2dNormActivation(
                16, 16, kernel_size=3, stride=1, groups=16,
                norm_layer=norm_layer, activation_layer=nn.SiLU
            ),
            Conv2dNormActivation(
                16, 8, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            ),  # Final output: 8 channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)  # 96x96x64 -> 192x192x32
        x = self.stage2(x)  # 192x192x32 -> 384x384x8
        return x

def masked_mse(images, masks, reconstructions):
    flattened_images = images.transpose(0, 1).flatten(start_dim=1)
    flattened_reconstructions = reconstructions.transpose(0, 1).flatten(start_dim=1)
    flattened_masks = masks.flatten().bool()

    image_content = flattened_images[:, flattened_masks]
    reconstruction_content = flattened_reconstructions[:, flattened_masks]

    return nn.functional.mse_loss(image_content, reconstruction_content)

@dataclass
class AutoencoderConfig:
    num_layers_block: List[int]
    augmentation_flip: bool
    augmentation_rotate: bool

    def __str__(self):
        return f"AutoencoderConfig({self.num_layers_block}, flip={self.augmentation_flip}, rotate={self.augmentation_rotate})"