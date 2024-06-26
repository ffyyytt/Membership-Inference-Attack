import torch
import torchvision
import tensorflow as tf

class ModelFromBackbone(torch.nn.Module):
    def __init__(self, backbone, num_classes, device):
        super().__init__()
        self.device = device
        self.backbone = getattr(torchvision.models, backbone)(weights=None)

        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, num_classes)
        elif hasattr(self.backbone, 'classifier') and isinstance(self.backbone.classifier, torch.nn.Sequential):
            self.backbone.classifier = torch.nn.Linear(self.backbone.last_channel, num_classes)
        else:
            raise NotImplementedError(f"Backbone {backbone} is not supported yet.")

    def forward(self, x):
        x.to(self.device)
        return self.backbone(x)
    
def modelTF(backbone: str = "resnet50", n_classes: int = 10):
    inputImage = tf.keras.layers.Input(shape = (None, None, 3), dtype=tf.uint8, name = 'image')
    image = tf.keras.layers.Lambda(lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="tf"))(inputImage)
    feature = tf.keras.layers.GlobalAveragePooling2D()(getattr(tf.keras.applications, backbone)(weights = None, include_top = False)(image))
    output = tf.keras.layers.Dense(n_classes, activation='softmax')(feature)

    model = tf.keras.models.Model(inputs = [inputImage], outputs = [output])
    return model