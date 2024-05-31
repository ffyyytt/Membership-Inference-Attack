import torch
import torchvision

class ModelFromBackbone(torch.nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = getattr(torchvision.models.models, backbone)(pretrained=True)
        self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)