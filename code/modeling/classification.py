import torch


def l1_reg(model):
    l1 = 0
    for name, param in model.named_parameters():
        if "bias" not in name:
            l1 += torch.norm(param, p=1)
    return l1

def accuracy(pred_class, target):
    return (pred_class == target).float().mean()

def marco_f1_score(pred_class, target):
    # F1 for positive class (1)
    tp_pos = ((pred_class == 1) & (target == 1)).sum().float()
    fp_pos = ((pred_class == 1) & (target == 0)).sum().float()
    fn_pos = ((pred_class == 0) & (target == 1)).sum().float()
    f1_pos = 2 * tp_pos / (2 * tp_pos + fp_pos + fn_pos + 1e-8)

    # F1 for negative class (-1, mapped to 0)
    tp_neg = ((pred_class == 0) & (target == 0)).sum().float()
    fp_neg = ((pred_class == 0) & (target == 1)).sum().float()
    fn_neg = ((pred_class == 1) & (target == 0)).sum().float()
    f1_neg = 2 * tp_neg / (2 * tp_neg + fp_neg + fn_neg + 1e-8)

    macro_f1 = (f1_pos + f1_neg) / 2
    return macro_f1

def masked_bce_loss_acc(pred, label):
    # pred: (N, 1, H, W); label: (N, H, W), with label values -1 (negative), 1 (positive), 0 (masked)
    pred = pred.flatten()
    label = label.flatten()
    mask = (label != 0)     # valid indices

    pred_valid = pred[mask]
    label_valid = label[mask]

    # -1/1 -> 0/1
    target_valid = (label_valid + 1) / 2

    # Compute binary cross entropy loss with logits
    loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_valid, target_valid)

    pred_class = (pred_valid > 0).float()

    return loss, accuracy(pred_class, target_valid), marco_f1_score(pred_class, target_valid)