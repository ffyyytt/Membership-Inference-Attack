import torch
import torchvision

class ImageDatasetFromImagePathsAndLabel(torch.utils.data.Dataset):
    def __init__(self, imagePaths, labels, device, transform=None, target_transform=None, cache=False):
        self.device = device
        self.imagePaths = imagePaths
        self.labels = torch.Tensor(labels).to(self.device)
        self.transform = transform
        self.target_transform = target_transform
        self.cache = cache
        if self.cache:
            self.boolcached = [0]*len(self.imagePaths)
            self.images = [0]*len(self.imagePaths)
    
    def __len__(self):
        return len(self.imagePaths)
    
    def __getitem__(self, idx):
        if self.cache:
            if self.boolcached[idx]:
                image = self.images[idx]
            else:
                image = torchvision.io.read_image(self.imagePaths[idx]).to(torch.float32).to(self.device)
                self.images[idx] = image
                self.boolcached[idx] = 1
        else:
            image = torchvision.io.read_image(self.imagePaths[idx]).to(torch.float32).to(self.device)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label