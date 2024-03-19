import torch
import torch.nn as nn
import torch.nn.functional as F

class CCCLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho = torch.sum(vx * vy) / ((torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))) + 1e-10)
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = 2 * rho * x_s * y_s / ((x_s ** 2 + y_s ** 2 + (x_m - y_m) ** 2) + 1e-10)
        return 1 - ccc
    # def forward(self, x, y):
    #     y = y.contiguous().view(-1)
    #     x = x.contiguous().view(-1)
    #     vx = x - torch.mean(x)
    #     vy = y - torch.mean(y)
    #     rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + self.eps)
    #     x_m = torch.mean(x)
    #     y_m = torch.mean(y)
    #     x_s = torch.std(x)
    #     y_s = torch.std(y)
    #     ccc = 2 * rho * x_s * y_s / (torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
    #     return 1 - ccc

class VALoss(nn.Module):
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

class ExprLoss(nn.Module):
    def __init__(self, weight, device):
        super().__init__()
        w = torch.FloatTensor(weight).to(device)  # Move weights to the specified device
        self.ce = nn.CrossEntropyLoss(weight=w)

    def forward(self, x, y):
        return self.ce(x, y)


class FocalLoss(nn.Module):
    def __init__(self, device='cuda', weights=None, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weights = torch.tensor(weights).to(device) if weights is not None else None
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 입력된 inputs는 클래스 별 예측 확률의 logit(즉, softmax 이전의 값들)이어야 합니다.
        if self.weights is not None:
            BCE_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weights)
        else:
            BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        pt = torch.exp(-BCE_loss)  # pt는 모델이 정확히 예측한 클래스에 대한 확률
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
