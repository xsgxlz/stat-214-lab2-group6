import torch
import torch.nn as nn


def l1_reg(model):
    l1 = 0
    for name, param in model.named_parameters():
        if "bias" not in name:
            l1 += torch.norm(param, p=1)
    return l1

def accuracy(pred_class, target):
    return (pred_class == target).float().mean()

def macro_f1_score(pred_class, target):
    # F1 for positive class (1)
    tp_pos = ((pred_class == 1) & (target == 1)).sum()
    fp_pos = ((pred_class == 1) & (target == 0)).sum()
    fn_pos = ((pred_class == 0) & (target == 1)).sum()
    f1_pos = 2 * tp_pos / (2 * tp_pos + fp_pos + fn_pos + 1e-8)

    # F1 for negative class (-1, mapped to 0)
    tp_neg = ((pred_class == 0) & (target == 0)).sum()
    fp_neg = ((pred_class == 0) & (target == 1)).sum()
    fn_neg = ((pred_class == 1) & (target == 0)).sum()
    f1_neg = 2 * tp_neg / (2 * tp_neg + fp_neg + fn_neg + 1e-8)

    macro_f1 = (f1_pos + f1_neg) / 2
    return macro_f1

def masked_bce_loss(pred, label, acc=False, f1=False, class_weight=None):
    # pred: (N, 1, H, W); label: (N, H, W), with label values -1 (negative), 1 (positive), 0 (masked)
    pred = pred.flatten()
    label = label.flatten()
    mask = (label != 0)  # valid indices

    pred_valid = pred[mask]
    label_valid = label[mask]

    # -1/1 -> 0/1
    target_valid = (label_valid + 1) // 2  # [0, 1]

    # Compute class weights if "balanced"
    if class_weight == "balanced":
        n_samples = target_valid.numel()  # Number of valid samples
        n_classes = 2
        counts = torch.bincount(target_valid.long(), minlength=2)  # Counts of 0 and 1
        weights = n_samples / (n_classes * counts.float().clamp(min=1))  # Avoid div by 0
        weight_tensor = weights[target_valid.long()]  # Per-sample weights
    else:
        weight_tensor = None  # Uniform weights (None for BCE)

    # Compute binary cross entropy loss with logits
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred_valid, target_valid.float(), weight=weight_tensor
    )

    with torch.inference_mode():
        pred_class = (pred_valid > 0)
    
    result = [loss]
    if acc:
        with torch.inference_mode():
            acc_value = accuracy(pred_class, target_valid)
        result.append(acc_value)
    if f1:
        with torch.inference_mode():
            f1_value = macro_f1_score(pred_class, target_valid)
        result.append(f1_value)
    
    return loss if len(result) == 1 else tuple(result)

def masked_hinge_loss(pred, label, acc=False, f1=False, class_weight=None):
    """
    Compute masked hinge loss (SVM-style) with optional accuracy, F1 metrics, and class weights.
    
    Args:
        pred: Tensor of shape [N, 1, H, W], logits (unnormalized predictions).
        label: Tensor of shape [N, H, W], with values -1 (negative), 1 (positive), 0 (masked).
        acc: Boolean, whether to compute accuracy.
        f1: Boolean, whether to compute F1 score.
        class_weight: "balanced" or None. If "balanced", weights are n_samples / (n_classes * counts).
    
    Returns:
        loss: Hinge loss on valid pixels (scalar tensor).
        or (loss, acc, f1): Tuple if acc and/or f1 are True.
    """
    pred = pred.flatten()
    label = label.flatten()
    mask = (label != 0)

    pred_valid = pred[mask]
    label_valid = label[mask]

    # Compute class weights if "balanced"
    if class_weight == "balanced":
        n_samples = label_valid.numel()
        n_classes = 2
        # Convert -1/1 to 0/1 for bincount
        target_valid = (label_valid + 1) // 2  # [0, 1]
        counts = torch.bincount(target_valid.long(), minlength=2)
        weights = n_samples / (n_classes * counts.float().clamp(min=1))
        weight_tensor = weights[target_valid.long()]
    else:
        weight_tensor = torch.ones_like(label_valid, dtype=torch.float32)

    # Compute hinge loss: max(0, 1 - y * pred)
    hinge = torch.clamp(1 - label_valid * pred_valid, min=0)
    weighted_hinge = hinge * weight_tensor  # Apply per-sample weights
    loss = weighted_hinge.mean()

    with torch.inference_mode():
        pred_class = (pred_valid > 0)
        label_valid_binary = label_valid > 0  # -1/1 -> 0/1 for metrics
    
    result = [loss]
    if acc:
        with torch.inference_mode():
            acc_value = accuracy(pred_class, label_valid_binary)
        result.append(acc_value)
    if f1:
        with torch.inference_mode():
            f1_value = macro_f1_score(pred_class, label_valid_binary)
        result.append(f1_value)
    
    return loss if len(result) == 1 else tuple(result)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding="same")
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()
        self.residue = (in_channels == out_channels)
    def forward(self, x):
        out = self.activation(self.bn(self.conv(x)))
        if self.residue:
            out += x
        return out

class ConvHead(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, kernel_size, hidden_channels):
        super().__init__()
        if num_layers == 1:
            self.layers = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding="same")
        else:
            input_layer = ConvBlock(in_channels, hidden_channels, kernel_size)
            hidden_layers = [ConvBlock(hidden_channels, hidden_channels, kernel_size) for _ in range(num_layers - 2)]
            output_layer = nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, padding="same")
            self.layers = nn.Sequential(input_layer, *hidden_layers, output_layer)
    def forward(self, x):
        return self.layers(x)