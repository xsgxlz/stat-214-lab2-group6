from dataclasses import dataclass
from typing import List

import torch
from torch import nn
from torchvision.models.efficientnet import EfficientNet, MBConvConfig, FusedMBConvConfig, Conv2dNormActivation

class EfficientNetEncoder(EfficientNet):
    """Custom EfficientNet-based encoder for feature extraction from multi-channel images."""
    def __init__(
        self,
        inverted_residual_setting,           # List[MBConvConfig]: Configs for inverted residual blocks
        dropout,                            # float: Dropout probability for training
        input_channels=8,                   # int: Number of input channels (default: 8 for MISR data)
        stochastic_depth_prob=0.1,          # float: Probability for stochastic depth regularization
        norm_layer=None,                    # Callable: Normalization layer (default: BatchNorm2d)
        last_channel=None,                  # int: Number of channels in the last layer (optional)
    ):
        """Initialize the encoder, overriding the first conv layer and removing pooling/classifier."""
        super().__init__(
            inverted_residual_setting=inverted_residual_setting,
            dropout=dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            num_classes=1,  # Dummy value, ignored since classifier is removed
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
        del self.avgpool  # Remove pooling layer
        del self.classifier  # Remove classification head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, input_channels, height, width)

        Returns:
            torch.Tensor: Feature maps of shape (batch, channels, height/4, width/4)
        """
        return self.features(x)

class EfficientNetDecoder(nn.Module):
    """Decoder to upsample encoder features back to original image resolution."""
    def __init__(self, norm_layer=None):
        """Initialize the decoder with two upsampling stages.

        Args:
            norm_layer (Callable, optional): Normalization layer (default: BatchNorm2d)
        """
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
            # Projection
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
        """Upsample encoder features to reconstruct the input image.

        Args:
            x (torch.Tensor): Encoder output of shape (batch, 64, 96, 96)

        Returns:
            torch.Tensor: Reconstructed image of shape (batch, 8, 384, 384)
        """
        x = self.stage1(x)  # 96x96x64 -> 192x192x32
        x = self.stage2(x)  # 192x192x32 -> 384x384x8
        return x

def masked_mse(images, masks, reconstructions):
    """Compute Mean Squared Error (MSE) only on masked regions.

    Args:
        images (torch.Tensor): Ground truth images (batch, channels, height, width)
        masks (torch.Tensor): Binary masks (batch, height, width), 1 for valid regions
        reconstructions (torch.Tensor): Predicted images (batch, channels, height, width)

    Returns:
        torch.Tensor: Scalar MSE loss for masked regions
    """
    flattened_images = images.transpose(0, 1).flatten(start_dim=1)
    flattened_reconstructions = reconstructions.transpose(0, 1).flatten(start_dim=1)
    flattened_masks = masks.flatten().bool()

    image_content = flattened_images[:, flattened_masks]
    reconstruction_content = flattened_reconstructions[:, flattened_masks]

    return nn.functional.mse_loss(image_content, reconstruction_content)

@dataclass
class AutoencoderConfig:
    """Configuration dataclass for autoencoder settings."""
    num_layers_block: List[int]          # List[int]: Number of layers per block in the encoder
    augmentation_flip: bool              # bool: Whether to apply horizontal flip augmentation
    augmentation_rotate: bool            # bool: Whether to apply rotation augmentation

    def __str__(self):
        """Return a string representation of the config.

        Returns:
            str: Formatted config description
        """
        return f"AutoencoderConfig({self.num_layers_block}, flip={self.augmentation_flip}, rotate={self.augmentation_rotate})"