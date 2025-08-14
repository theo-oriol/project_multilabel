import torch
import torch.nn as nn
import torch.nn.functional as F

def get_loss_function(name):
    name = name.lower()
    if name == "bcewithlogitsloss":
        return nn.BCEWithLogitsLoss(reduction='mean')
    elif name == "crossentropyloss":
        return nn.CrossEntropyLoss()
    elif name == "mse":
        return nn.MSELoss()
    elif name == "focalbce":
        return Focal_binary_cross_entropy()
    else:
        raise ValueError(f"Unknown loss function: {name}")
    



class Focal_binary_cross_entropy(nn.Module):
    def __init__(self, gamma=2):
        super(Focal_binary_cross_entropy, self).__init__()
        self.gamma = gamma
    def forward(self, logits, targets):   
        targets = targets.to(dtype=logits.dtype)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce) 
        focal_weight = (1 - pt).pow(self.gamma)

        loss = focal_weight * bce
        loss = loss.mean()
        return loss