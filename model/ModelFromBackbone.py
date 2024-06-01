import torch
import torchvision

class ModelFromBackbone(torch.nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = getattr(torchvision.models, backbone)(weights=None)

        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, num_classes)
        elif hasattr(self.backbone, 'classifier') and isinstance(self.backbone.classifier, torch.nn.Sequential):
            self.backbone.classifier = torch.nn.Linear(self.backbone.last_channel, num_classes)
        else:
            raise NotImplementedError(f"Backbone {backbone} is not supported yet.")

    def forward(self, x):
        return self.backbone(x)