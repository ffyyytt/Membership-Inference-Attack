import torch
import torchvision

class ModelFromBackbone(torch.nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = getattr(torchvision.models, backbone)(pretrained=True, weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)