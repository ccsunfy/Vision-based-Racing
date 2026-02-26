import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional, Type, Union, Dict
from torchvision import models
from torch.utils.data import Dataset, DataLoader

# 定义学生策略
import torch
import torch.nn as nn
from torchvision import models

class StudentPolicy(nn.Module):
    backbone_alias = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "efficientnet_l": models.efficientnet_v2_l,
        "mobilenet_l": models.mobilenet_v3_large,
    }

    def __init__(
        self,
        input_channels: int = 3,      
        backbone_name: str = "resnet18",
        pretrained: bool = True,        
        freeze_backbone: bool = True,  
        hidden_dim: int = 256,         
    ):
        super().__init__()
        
        # choose and intialize backbone
        self.backbone = self.backbone_alias[backbone_name](pretrained=pretrained)
        
        # freeze backbone parameters 
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # exchange the last layer
        if "resnet" in backbone_name:
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # 移除原始分类层
        elif "efficientnet" in backbone_name:
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif "mobilenet" in backbone_name:
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier = nn.Identity()

        self.action_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4) 
        )

    def forward(self, image):
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        features = self.backbone(image)
        actions = self.action_head(features)
        return actions