import os
import glob
import torch
import torchvision

import numpy as np

from torchvision.transforms import v2

from data.ImageDatasetFromImagePathsAndLabel import *
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

__RANDOM__SEED__ = 1312

# 0: Train, Member
# 1: Train
# 2: Non-mem
# 3-7: Shadow

__CIFAR10_N_CLASSES__ = 10
__CIFAR10_N_SPLIT__ = 2
__CIFAR10_N_SHADOW__ = 256
__CIFAR10_TRAIN_SET__ = [0]
__CIFAR10_MEMBER_SET__ = [0]
__CIFAR10_NON_MEM_SET__ = [1]
__CIFAR10_BATCH_SIZE__ = 128
__CIFAR10_TRANSFORMS__ = torchvision.transforms.v2.Compose([
    # torchvision.transforms.v2.RandomResizedCrop(size=(224, 224), antialias=True),
    torchvision.transforms.v2.ToTensor(),
    torchvision.transforms.v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
])

def _loadCIFAR10():
    labels = []
    labelset = {}
    
    # Glob
    imagePaths = glob.glob(os.path.expanduser("~")+"/data/CIFAR-10-images-master/train/*/*")
    for file in imagePaths:
        if file.split("/")[-2] not in labelset:
            labelset[file.split("/")[-2]] = len(labelset)
        labels.append(file.split("/")[-2])

    # One-hot
    for k, v in labelset.items():
        labelset[k] = [0.]*v + [1.] + [0.]*(len(labelset)-v-1)

    # map
    imagePaths = np.array(imagePaths)
    labels = np.array([labelset[label] for label in labels])

    return imagePaths, labels

def loadCenTrainCIFAR10(device):
    imagePaths, labels = [], []
    X, Y = _loadCIFAR10()
    skf = StratifiedKFold(n_splits=__CIFAR10_N_SPLIT__, shuffle=True, random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(skf.split(X, np.argmax(Y, axis=1))):
        if i in __CIFAR10_TRAIN_SET__:
            imagePaths += X[test_index].tolist()
            labels += Y[test_index].tolist()
    return torch.utils.data.DataLoader(ImageDatasetFromImagePathsAndLabel(imagePaths, labels, device, __CIFAR10_TRANSFORMS__), batch_size=__CIFAR10_BATCH_SIZE__, shuffle=False)

def loadCenShadowTrainCIFAR10(idx, device):
    imagePaths, labels = [], []
    X, Y = _loadCIFAR10()
    sss = StratifiedShuffleSplit(n_splits=__CIFAR10_N_SHADOW__, test_size=len(__CIFAR10_TRAIN_SET__)/__CIFAR10_N_SPLIT__, random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(sss.split(X, np.argmax(Y, axis=1))):
        if i == idx:
            imagePaths += X[test_index].tolist()
            labels += Y[test_index].tolist()
    return torch.utils.data.DataLoader(ImageDatasetFromImagePathsAndLabel(imagePaths, labels, device, __CIFAR10_TRANSFORMS__), batch_size=__CIFAR10_BATCH_SIZE__, shuffle=False)

def loadMIADataCIFAR10(device):
    imagePaths, labels, memberLabels = [], [], []
    X, Y = _loadCIFAR10()
    inOutLabels = np.zeros([len(X), __CIFAR10_N_SHADOW__])
    skf = StratifiedKFold(n_splits=__CIFAR10_N_SPLIT__, shuffle=True, random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(skf.split(X, np.argmax(Y, axis=1))):
        if i in __CIFAR10_MEMBER_SET__+__CIFAR10_NON_MEM_SET__:
            imagePaths += X[test_index].tolist()
            labels += Y[test_index].tolist()
            memberLabels += [int(i in __CIFAR10_MEMBER_SET__)]*len(test_index)

    sss = StratifiedShuffleSplit(n_splits=__CIFAR10_N_SHADOW__, test_size=len(__CIFAR10_TRAIN_SET__)/__CIFAR10_N_SPLIT__, random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(sss.split(X, np.argmax(Y, axis=1))):
        inOutLabels[test_index, i] = 1
    return torch.utils.data.DataLoader(ImageDatasetFromImagePathsAndLabel(imagePaths, labels, device, __CIFAR10_TRANSFORMS__), batch_size=__CIFAR10_BATCH_SIZE__, shuffle=False), memberLabels, inOutLabels