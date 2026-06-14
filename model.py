"""Multi-task model: shared CNN backbone with two heads.

- Keypoint head: regresses normalized (x, y) in [0, 1] via a sigmoid.
- Classification head: 3-way softmax over shape classes.
"""

import torch
import torch.nn as nn
import torchvision


class GCPMultiTaskModel(nn.Module):
    def __init__(self, backbone_name="resnet34", num_classes=3, pretrained=True):
        super().__init__()

        if backbone_name == "resnet34":
            backbone = torchvision.models.resnet34(
                weights=torchvision.models.ResNet34_Weights.DEFAULT if pretrained else None
            )
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif backbone_name == "resnet18":
            backbone = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif backbone_name == "efficientnet_b0":
            backbone = torchvision.models.efficientnet_b0(
                weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            feat_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone = backbone
        self.feat_dim = feat_dim

        self.keypoint_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

        self.class_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        if feats.dim() > 2:
            feats = torch.flatten(feats, 1)

        kp = torch.sigmoid(self.keypoint_head(feats))  # (B, 2) in [0, 1]
        logits = self.class_head(feats)  # (B, num_classes)

        return kp, logits
