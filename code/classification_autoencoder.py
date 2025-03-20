# %%
import time
import sys
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision.models.efficientnet import MBConvConfig, FusedMBConvConfig

import optuna

sys.path.append("/jet/home/azhang19/stat 214/stat-214-lab2-group6/code/modeling")
from preprocessing import to_NCHW, pad_to_384x384, standardize_images
from autoencoder import EfficientNetEncoder, EfficientNetDecoder, AutoencoderConfig
from classification import masked_bce_loss, masked_hinge_loss, l1_reg, ConvHead

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
ckpt_path = "/jet/home/azhang19/stat 214/stat-214-lab2-group6/code/modeling/ckpt"
saved_epoch = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 15000, 20000, 25000, 30000, 35000, 40000]

all_features = torch.zeros((2, 2, 2, 2, 2, len(saved_epoch), 3, 64, 384, 384)).to(device)

num_layers_per_block = [1, 2]
augementation = [True, False]
for block1, block2, block3, flip, rotate in itertools.product(*[num_layers_per_block] * 3,*[augementation] * 2):
    config = AutoencoderConfig(
        num_layers_block=[block1, block2, block3],
        augmentation_flip=flip,
        augmentation_rotate=rotate
    )
    encoder_config = [
        FusedMBConvConfig(1, 3, 1, 16, 16, config.num_layers_block[0]),  # 384x384x8 -> 384x384x16
        FusedMBConvConfig(4, 3, 2, 16, 32, config.num_layers_block[1]),  # 384x384x16 -> 192x192x32
        MBConvConfig(4, 3, 2, 32, 64, config.num_layers_block[2]),       # 192x192x32 -> 96x96x64
    ]

    encoder = EfficientNetEncoder(
        inverted_residual_setting=encoder_config,
        dropout=0.1,
        input_channels=8,
        last_channel=64,
    )

    decoder = EfficientNetDecoder()
    autoencoder = nn.Sequential(encoder, decoder).train().to(device)

    folder_path = os.path.join(ckpt_path, str(config))
    
    for i, epoch in enumerate(saved_epoch):
        autoencoder.load_state_dict(torch.load(os.path.join(folder_path, f"autoencoder_{epoch}.pth")))
        autoencoder.eval()
        with torch.inference_mode():
            encoder = autoencoder[0]
            features = encoder(labeled_images)
            features = nn.functional.interpolate(features, size=384, mode="bicubic", antialias=True)
            all_features[block1-1, block2-1, block3-1, int(flip), int(rotate), i] = features

# %%
@torch.compile
def train_and_validate(
    train_data, train_labels, val_data, val_labels,
    in_channels, num_layers, kernel_size, hidden_channels,
    epochs, lr, weight_decay, optimizer_class, loss_mix_ratio, l1, class_weight, device
):
    # Create the classifier
    #classifier = nn.Conv2d(in_channels, 1, kernel_size=kernel_size, 
    #                      padding="same", padding_mode="replicate").to(device)
    classifier = ConvHead(in_channels, 1, num_layers, kernel_size, hidden_channels).to(device)
    classifier.train()

    # Instantiate the optimizer
    optimizer = optimizer_class(classifier.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    for epoch in range(epochs):
        classifier.train()
        optimizer.zero_grad(set_to_none=True)
        pred = classifier(train_data)

        bce_loss = masked_bce_loss(pred, train_labels, class_weight=class_weight)
        hinge_loss = masked_hinge_loss(pred, train_labels, class_weight=class_weight)

        loss = loss_mix_ratio * bce_loss + (1 - loss_mix_ratio) * hinge_loss
        # Add L1 regularization
        loss = loss + l1 * l1_reg(classifier)
        loss.backward()
        optimizer.step()

    # Validation
    classifier.eval()
    with torch.inference_mode():
        val_pred = classifier(val_data)
        val_loss, val_acc, val_f1 = masked_hinge_loss(val_pred, val_labels, acc=True, f1=True)

    return val_f1, val_acc

def objective(trial):
    # Suggest hyperparameters with updated API calls
    epochs = trial.suggest_int("epochs", 0, 800)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "AdamW"])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    hidden_channels = trial.suggest_int("hidden_channels", 1, 64)
    kernel_size = trial.suggest_int("kernel_size", 1, 10)
    loss_mix_ratio = trial.suggest_float("loss_mix_ratio", 0, 1)
    l1 = trial.suggest_float("l1", 1e-5, 5e-1, log=True)

    block1 = trial.suggest_int("block1", 1, 2)
    block2 = trial.suggest_int("block2", 1, 2)
    block3 = trial.suggest_int("block3", 1, 2)
    flip = trial.suggest_categorical("flip", [True, False])
    rotate = trial.suggest_categorical("rotate", [True, False])
    autoencoder_epoch = trial.suggest_int("autoencoder_epoch", 0, len(saved_epoch)-1)

    with_orginal = trial.suggest_categorical("with_orginal", [True, False])
    class_weight = trial.suggest_categorical("class_weight", ["balanced", None])

    feature = all_features[block1-1, block2-1, block3-1, int(flip), int(rotate), autoencoder_epoch]
    feature, _, _ = standardize_images(feature, labeled_masks)

    if with_orginal:
        feature = torch.cat([feature, labeled_images], dim=1)
    
    # Map string to actual optimizer class
    optimizer_class = torch.optim.SGD if optimizer_name == "SGD" else torch.optim.AdamW

    # Cross-validation indices (modify as needed)
    train_val_idx = [0, 1]
    
    # Container for metrics from each fold
    fold_records_f1 = torch.zeros(len(train_val_idx))  # For F1 (objective)
    fold_records_acc = torch.zeros(len(train_val_idx))  # For accuracy (logging)

    # Assuming feature and labels are defined globally (e.g., torch tensors)
    for i in train_val_idx:
        # Leave-one-out style split
        train_idx = [j for j in train_val_idx if j != i]
        val_idx = [i]

        # Get training and validation data
        train_data = feature[train_idx]
        train_labels = labels[train_idx]
        val_data = feature[val_idx]
        val_labels = labels[val_idx]

        # Train and validate, get both F1 and accuracy
        val_f1, val_acc = train_and_validate(
            train_data=train_data,
            train_labels=train_labels,
            val_data=val_data,
            val_labels=val_labels,
            in_channels=feature.shape[1],
            num_layers=num_layers,
            kernel_size=kernel_size,
            hidden_channels=hidden_channels,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            optimizer_class=optimizer_class,
            loss_mix_ratio=loss_mix_ratio,
            l1=l1,
            class_weight=class_weight,
            device=device
        )

        fold_records_f1[i] = val_f1
        fold_records_acc[i] = val_acc
    
    avg_f1 = fold_records_f1.mean().item()
    avg_acc = fold_records_acc.mean().item()

    trial.set_user_attr("val_acc", avg_acc)
    #print(f"Trial {trial.number}: F1 = {avg_f1}, Acc = {avg_acc}")
    
    # Return average F1 score across folds
    return avg_f1

# %%
study = optuna.create_study(direction="maximize")

# %%
# Optimize the study by running a number of trials (e.g., 100 trials).
study.optimize(objective, n_trials=9000)

# %%
import pickle
with open("optuna_study_autoencoder.pkl", "wb") as f:
    pickle.dump(study, f)
