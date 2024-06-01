import torch
import torchvision

class ImageDatasetFromImagePathsAndLabel(torch.utils.data.Dataset):
    def __init__(self, imagePaths, labels, transform=None, target_transform=None):
        self.imagePaths = imagePaths
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.imagePaths)
    
    def __getitem__(self, idx):
        image = torchvision.io.read_image(self.imagePaths[idx])
        print(image)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label