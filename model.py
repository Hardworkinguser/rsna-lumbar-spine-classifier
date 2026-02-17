# model.py
import torch
import torch.nn as nn
import torchvision.models as models

class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.reduction = 8
        self.query_conv = nn.Conv2d(in_channels, in_channels // self.reduction, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // self.reduction, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, w, h = x.size()
        q = self.query_conv(x).view(b, -1, w * h).permute(0, 2, 1)
        k = self.key_conv(x).view(b, -1, w * h)
        v = self.value_conv(x).view(b, -1, w * h)
        energy = torch.bmm(q, k)
        attn = self.softmax(energy)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(b, c, w, h)
        return self.gamma * out + x

class CustomConvNeXtWithAttention(nn.Module):
    def __init__(self, num_classes=3, pretrained_weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1):
        super().__init__()
        self.model = models.convnext_small(weights=pretrained_weights)
        self.attention = Attention(in_channels=768)
        num_ftrs = self.model.classifier[2].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x, return_features=False):
        x = self.model.features(x)
        x = self.attention(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        output = self.model.classifier(x)
        if return_features:
            return output, x
        return output

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha if alpha is not None else 1.0
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if isinstance(self.alpha, torch.Tensor):
            focal_loss = self.alpha[targets] * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()
