# shoplifting_detection/models/video_focalnet.py

import torch
import torch.nn as nn
from timm import create_model

class VideoFocalNet(nn.Module):
    def __init__(self, num_classes, num_frames):
        super().__init__()
        self.backbone = create_model('focalnet_tiny_srf', pretrained=True, num_classes=0)
        self.num_frames = num_frames
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        # x shape: (batch_size, num_frames, 3, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)

        features = self.backbone(x)
        features = features.view(b, t, -1)

        # Add temporal dimension
        features = features.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (b, C, t, 1, 1)
        features = self.temporal_pool(features).squeeze(-1).squeeze(-1).squeeze(-1)

        return self.fc(features)