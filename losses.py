import torch
import torch.nn as nn

# Focal Loss as an improvement to Cross Entropy Loss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        
    def forward(self, logits, targets):
        ce_loss = self.ce_loss(logits, targets)  # Compute CE loss first
        pt = torch.exp(-ce_loss)  # Probabilities of true classes
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # Apply Focal Loss formula
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
