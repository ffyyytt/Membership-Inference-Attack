import torch
import torchvision

class ImageDatasetFromImagePathsAndLabel(torch.utils.data.Dataset):
    def __init__(self, imagePaths, labels, device, transform=None, target_transform=None):
        self.device = device
        self.imagePaths = imagePaths
        self.labels = torch.Tensor(labels).to(self.device)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.imagePaths)
    
    def __getitem__(self, idx):
        image = torchvision.io.read_image(self.imagePaths[idx]).to(torch.float32).to(self.device)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label