import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# CCC Loss (for VA task)
# ----------------------
class CCCLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho = torch.sum(vx * vy) / ((torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))) + self.eps)
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = 2 * rho * x_s * y_s / ((x_s ** 2 + y_s ** 2 + (x_m - y_m) ** 2) + self.eps)
        return 1 - ccc


class VALoss(nn.Module):
    """
    Combined VA loss: CCC + MAE + MSE (optional)
    """
    def __init__(self, ccc=True, mae=True, mse=False, alpha=0.5, beta=0.5, eps=1e-8):
        super().__init__()
        self.cccloss = CCCLoss(eps=eps)
        self.alpha = alpha
        self.beta = beta
        self.maeloss = nn.L1Loss()
        self.mseloss = nn.MSELoss()
        self.ccc = ccc
        self.mae = mae
        self.mse = mse

    def forward(self, x, y):
        loss = 0
        if self.ccc:
            loss += self.alpha * self.cccloss(x[:, 0], y[:, 0]) + self.beta * self.cccloss(x[:, 1], y[:, 1])
        if self.mae:
            loss += self.maeloss(x, y)
        if self.mse:
            loss += self.mseloss(x, y)
        return loss


# ----------------------
# EXPR Loss
# ----------------------
class ExprLoss(nn.Module):
    def __init__(self, weight, device):
        super().__init__()
        w = torch.FloatTensor(weight).to(device)
        self.ce = nn.CrossEntropyLoss(weight=w)

    def forward(self, x, y):
        return self.ce(x, y)


# ----------------------
# AU Focal Loss
# ----------------------
class AUFocalLoss(nn.Module):
    def __init__(self, device='cuda', weights=None, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.weights = torch.tensor(weights).to(device) if weights is not None else None
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits before softmax
        bce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weights) \
            if self.weights is not None else F.cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ----------------------
# EXPR Focal Loss (multi-class)
# ----------------------
class ExprFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')  # [B]
        pt = torch.exp(-ce_loss)  # pt = softmax prob. of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_loss(task: str, device: torch.device):
    if task == "expr":
        # 예시용 weight (8 classes), 필요 시 수정 가능
        weights = [1.0] * 8
        return ExprFocalLoss(weight=torch.tensor(weights).to(device))
        # return ExprLoss(weight=weights, device=device)

    elif task == "va":
        return VALoss(ccc=True, mae=True, mse=False)

    elif task == "au":
        # 예시용 weight (12 classes), 필요 시 수정 가능
        weights = [1.0] * 12
        return  AUFocalLoss(weights=weights, device=device)

    else:
        raise ValueError(f"Unknown task: {task}")
