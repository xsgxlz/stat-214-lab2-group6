import torch


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

def masked_bce_loss(pred, label, acc=False, f1=False):
    # pred: (N, 1, H, W); label: (N, H, W), with label values -1 (negative), 1 (positive), 0 (masked)
    pred = pred.flatten()
    label = label.flatten()
    mask = (label != 0)     # valid indices

    pred_valid = pred[mask]
    label_valid = label[mask]

    # -1/1 -> 0/1
    target_valid = (label_valid + 1) // 2

    # Compute binary cross entropy loss with logits
    loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_valid, target_valid.float())

    with torch.inference_mode():
        pred_class = (pred_valid > 0)
    
    result = [loss]
    if acc:
        with torch.inference_mode():
            acc = accuracy(pred_class, target_valid)
        result.append(acc)
    if f1:
        with torch.inference_mode():
            f1 = macro_f1_score(pred_class, target_valid)
        result.append(f1)
    
    return loss if len(result) == 1 else tuple(result)

def masked_hinge_loss(pred, label, acc=False, f1=False):
    """
    Compute masked hinge loss (SVM-style) with optional accuracy and F1 metrics.
    
    Args:
        pred: Tensor of shape [N, 1, H, W], logits (unnormalized predictions).
        label: Tensor of shape [N, H, W], with values -1 (negative), 1 (positive), 0 (masked).
        acc: Boolean, whether to compute accuracy.
        f1: Boolean, whether to compute F1 score.
    
    Returns:
        loss: Hinge loss on valid pixels (scalar tensor).
        or (loss, acc, f1): Tuple if acc and/or f1 are True.
    """
    # Flatten inputs
    pred = pred.flatten()  # [N * H * W]
    label = label.flatten()  # [N * H * W]
    
    # Mask for valid pixels (non-zero labels)
    mask = (label != 0)  # [N * H * W], True for -1 or 1
    
    # Filter to valid pixels
    pred_valid = pred[mask]
    label_valid = label[mask]
    
    # Compute hinge loss: max(0, 1 - y * pred)
    # Labels are already -1/1, pred is logits
    hinge = torch.clamp(1 - label_valid * pred_valid, min=0)
    loss = hinge.mean()  # Average over valid pixels
    
    # Compute predictions for metrics (threshold at 0, like BCE)
    with torch.inference_mode():
        pred_class = (pred_valid > 0)  # True (1) if logit > 0, False (0) otherwise
        label_valid = label_valid > 0  # True (1) if label == 1, False (0) if label == -1
    
    # Prepare result
    result = [loss]
    if acc:
        with torch.inference_mode():
            acc_value = accuracy(pred_class, label_valid)  # Assuming -1/1 labels work
        result.append(acc_value)
    if f1:
        with torch.inference_mode():
            f1_value = macro_f1_score(pred_class, label_valid)  # Assuming -1/1 compatible
        result.append(f1_value)
    
    # Return loss alone or tuple based on flags
    return loss if len(result) == 1 else tuple(result)
