# %%
import time
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
from torchvision.models.efficientnet import MBConvConfig, FusedMBConvConfig

sys.path.append("/jet/home/azhang19/stat 214/stat-214-lab2-group6/code/modeling")
from preprocessing import to_NCHW, pad_to_384x384, standardize_images
from autoencoder import EfficientNetEncoder, EfficientNetDecoder, AutoencoderConfig, masked_mse

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

use_amp = True

# %%
# Load and preprocess data
data = np.load("/jet/home/azhang19/stat 214/stat-214-lab2-group6/data/array_data.npz")
unlabeled_images, unlabeled_masks, labeled_images, labeled_masks, labels = data["unlabeled_images"], data["unlabeled_masks"], data["labeled_images"], data["labeled_masks"], data["labels"]

unlabeled_images = pad_to_384x384(to_NCHW(unlabeled_images))
unlabeled_masks = pad_to_384x384(unlabeled_masks)

labeled_images = pad_to_384x384(to_NCHW(labeled_images))
labeled_masks = pad_to_384x384(labeled_masks)
labels = pad_to_384x384(labels)

# Convert to tensors and move to GPU
unlabeled_images = torch.tensor(unlabeled_images, dtype=torch.float32).to(device)  # [161, 8, 384, 384]
unlabeled_masks = torch.tensor(unlabeled_masks, dtype=torch.bool).to(device)    # [161, 384, 384]

labeled_images = torch.tensor(labeled_images, dtype=torch.float32).to(device)      # [3, 8, 384, 384]
labeled_masks = torch.tensor(labeled_masks, dtype=torch.bool).to(device)        # [3, 384, 384]
labels = torch.tensor(labels, dtype=torch.long).to(device)                      # [3, 384, 384]


# Standardize images
unlabeled_images, std_channel, mean_channel = standardize_images(unlabeled_images, unlabeled_masks)
labeled_images, _, _ = standardize_images(labeled_images, labeled_masks, std_channel, mean_channel)

# %%
def parse_args():
    parser = argparse.ArgumentParser(description="Configure the Autoencoder model")

    # Argument for num_layers_block (list of integers)
    parser.add_argument(
        '--num-layers-block',
        type=int,
        nargs='+',  # Accepts multiple values as a list
        default=[1, 1, 1],  # Default value
        help="Number of layers in each block (e.g., --num-layers-block 1 1 1)"
    )

    # Argument for augmentation_flip (boolean)
    parser.add_argument(
        '--augmentation-flip',
        action='store_true',  # Sets to True if flag is present, False otherwise
        default=False,  # Default value
        help="Enable random flip augmentation"
    )

    # Argument for augmentation_rotate (boolean)
    parser.add_argument(
        '--augmentation-rotate',
        action='store_true',  # Sets to True if flag is present, False otherwise
        default=False,  # Default value
        help="Enable random rotation augmentation"
    )

# Parse arguments
    args = parser.parse_args()
    return args

args = parse_args()
config = AutoencoderConfig(
        num_layers_block=args.num_layers_block,
        augmentation_flip=args.augmentation_flip,
        augmentation_rotate=args.augmentation_rotate
    )

#config = AutoencoderConfig(num_layers_block=[1, 1, 1], augmentation_flip=True, augmentation_rotate=True)
print(config)

# %%
augmentation = []
if config.augmentation_flip:
    augmentation.append(torchvision.transforms.RandomHorizontalFlip(p=0.5))
    augmentation.append(torchvision.transforms.RandomVerticalFlip(p=0.5))
if config.augmentation_rotate:
    augmentation.append(torchvision.transforms.RandomRotation(degrees=180, expand=True,
                        interpolation=torchvision.transforms.InterpolationMode.BILINEAR))
    augmentation.append(torchvision.transforms.RandomCrop(size=384))
augmentation = torchvision.transforms.Compose(augmentation)

def apply_augment(images, masks, augmentation):
    images_masks = torch.cat([masks.unsqueeze(1).float(), images], dim=1)
    images_masks = [augmentation(image_mask) for image_mask in images_masks]
    images_masks = torch.stack(images_masks)
    return images_masks[:, 1:], images_masks[:, 0] > 0.5

if config.augmentation_flip or config.augmentation_rotate:
    augment = lambda images, masks: apply_augment(images, masks, augmentation)
else:
    augment = lambda images, masks: (images, masks)    

# %%
encoder_config = [
    FusedMBConvConfig(1, 3, 1, 16, 16, config.num_layers_block[0]),  # 384x384x8 -> 384x384x16
    FusedMBConvConfig(4, 3, 2, 16, 32, config.num_layers_block[1]),  # 384x384x16 -> 192x192x32
    MBConvConfig(4, 3, 2, 32, 64, config.num_layers_block[2]),       # 192x192x32 -> 96x96x64
]

# Build encoder and decoder
encoder = EfficientNetEncoder(
    inverted_residual_setting=encoder_config,
    dropout=0.1,
    input_channels=8,
    last_channel=64,
)

decoder = EfficientNetDecoder()

autoencoder = nn.Sequential(encoder, decoder).train().to(device)
#compiled_autoencoder = torch.compile(autoencoder)

# %%
num_epochs = 40000
ckpt = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 15000, 20000, 25000, 30000, 35000, 40000]
initial_lr = 1e-3  # Moderate starting LR for AdamW
weight_decay = 1e-2  # Regularization for small dataset

# Optimizer and scheduler
optimizer = optim.AdamW(autoencoder.parameters(), lr=initial_lr, weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)  # Decay to near-zero
scaler = torch.amp.GradScaler(device, enabled=use_amp)

losses = np.zeros(num_epochs)

# %%
@torch.compile
def trainer(images, masks, model, augment, optimizer, scheduler, scaler, loss_fn):
    with torch.inference_mode():
        images, masks = augment(images, masks)
    images, masks = images.clone(), masks.clone()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(device, enabled=use_amp):
        reconstructions = model(images)
        loss = loss_fn(images, masks, reconstructions)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    scheduler.step()

    return loss

# %%
ckpt_path = "/jet/home/azhang19/stat 214/stat-214-lab2-group6/code/modeling/ckpt"
os.makedirs(f"{ckpt_path}/{str(config)}", exist_ok=True)

# %%
for epoch in range(num_epochs):
    t = time.perf_counter()
    loss = trainer(unlabeled_images, unlabeled_masks, autoencoder, augment, optimizer, scheduler, scaler, masked_mse)
    losses[epoch] = loss.item()
    if epoch + 1 in ckpt:
        torch.save(autoencoder.state_dict(), f"{ckpt_path}/{str(config)}/autoencoder_{epoch + 1}.pth")
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss:.4f} - Time: {time.perf_counter() - t:.2f}s")

# %%



