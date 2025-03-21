import torch
import torch.nn as nn

def l1_reg(model):
    """Compute L1 regularization term for a model's weights (excluding biases).
    
    Args:
        model (nn.Module): PyTorch model to compute L1 norm over.
    
    Returns:
        torch.Tensor: Scalar L1 norm of weights.
    """
    l1 = 0
    for name, param in model.named_parameters():
        if "bias" not in name:
            l1 += torch.norm(param, p=1)
    return l1

def accuracy(pred_class, target):
    """Calculate accuracy between predicted and target binary classes.
    
    Args:
        pred_class (torch.Tensor): Predicted class labels (0 or 1).
        target (torch.Tensor): True class labels (0 or 1).
    
    Returns:
        torch.Tensor: Scalar accuracy (fraction of correct predictions).
    """
    return (pred_class == target).float().mean()

def macro_f1_score(pred_class, target):
    """Compute Macro F1 score for binary classification.
    
    Args:
        pred_class (torch.Tensor): Predicted class labels (0 or 1).
        target (torch.Tensor): True class labels (0 or 1).
    
    Returns:
        torch.Tensor: Scalar Macro F1 score, averaging F1 for both classes.
    """
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
    """Compute masked binary cross-entropy loss with optional metrics.
    
    Args:
        pred (torch.Tensor): Logits of shape [N, 1, H, W].
        label (torch.Tensor): Labels of shape [N, H, W] (-1, 1, 0 for masked).
        acc (bool): If True, return accuracy.
        f1 (bool): If True, return Macro F1 score.
        class_weight (str or None): "balanced" for inverse class frequency weights, else None.
    
    Returns:
        torch.Tensor: Scalar BCE loss, or tuple (loss, acc, f1) if acc/f1 requested.
    """
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
    """Compute masked hinge loss with optional metrics and class weights.
    
    Args:
        pred (torch.Tensor): Logits of shape [N, 1, H, W].
        label (torch.Tensor): Labels of shape [N, H, W] (-1, 1, 0 for masked).
        acc (bool): If True, return accuracy.
        f1 (bool): If True, return Macro F1 score.
        class_weight (str or None): "balanced" for inverse class frequency weights, else None.
    
    Returns:
        torch.Tensor: Scalar hinge loss, or tuple (loss, acc, f1) if acc/f1 requested.
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
    """Convolutional block with batch norm, SiLU activation, and optional residual connection.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
    """
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
    """Lightweight CNN head for pixel-wise classification.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (1 for binary classification).
        num_layers (int): Number of convolutional layers.
        kernel_size (int): Size of the convolutional kernel.
        hidden_channels (int): Number of channels in hidden layers.
    """
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

@torch.compile
def train_and_validate(
    train_data, train_labels, val_data, val_labels,
    in_channels, num_layers, kernel_size, hidden_channels,
    epochs, lr, weight_decay, optimizer_class, loss_mix_ratio, l1, class_weight,
    device, return_classifier=False
):
    """Train and validate a CNN classifier with a mixed loss function.
    
    Args:
        train_data (torch.Tensor): Training features [N, C, H, W].
        train_labels (torch.Tensor): Training labels [N, H, W] (-1, 1, 0).
        val_data (torch.Tensor): Validation features [N, C, H, W].
        val_labels (torch.Tensor): Validation labels [N, H, W] (-1, 1, 0).
        in_channels (int): Input feature channels.
        num_layers (int): Number of CNN layers.
        kernel_size (int): Convolutional kernel size.
        hidden_channels (int): Hidden layer channels.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        weight_decay (float): Weight decay for optimizer.
        optimizer_class (torch.optim.Optimizer): Optimizer class (e.g., SGD, AdamW).
        loss_mix_ratio (float): Weight for BCE vs. hinge loss (0 to 1).
        l1 (float): L1 regularization strength.
        class_weight (str or None): "balanced" or None for loss weighting.
        device (torch.device): Device to run on (e.g., "cuda").
        return_classifier (bool): If True, return trained model with metrics.
    
    Returns:
        tuple: (val_f1, val_acc) or (classifier, val_f1, val_acc) if return_classifier=True.
    """
    # Create the classifier
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

    if return_classifier:
        return classifier, val_f1, val_acc
    else:
        return val_f1, val_acc