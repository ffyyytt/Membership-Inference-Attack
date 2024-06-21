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

__AID_N_CLASSES__ = 30
__AID_N_SPLIT__ = 8
__AID_N_SHADOW__ = 32
__AID_TRAIN_SET__ = [0, 1]
__AID_MEMBER_SET__ = [0]
__AID_NON_MEM_SET__ = [2]
__AID_SHADOW_SET__ = [3, 4, 5, 6, 7]
__AID_BATCH_SIZE__ = 4096
__AID_TRANSFORMS__ = torchvision.transforms.v2.Compose([
    torchvision.transforms.v2.RandomResizedCrop(size=(224, 224), antialias=True),
    torchvision.transforms.v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
])


def _loadAID():
    labels = []
    labelset = {}
    
    # Glob
    imagePaths = glob.glob(os.path.expanduser("~")+"/datasets/AID/*/*")
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

def loadCenTrainAID(device):
    imagePaths, labels = [], []
    X, Y = _loadAID()
    skf = StratifiedKFold(n_splits=__AID_N_SPLIT__, shuffle=True, random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(skf.split(X, np.argmax(Y, axis=1))):
        if i in __AID_TRAIN_SET__:
            imagePaths += X[test_index].tolist()
            labels += Y[test_index].tolist()
    return torch.utils.data.DataLoader(ImageDatasetFromImagePathsAndLabel(imagePaths, labels, device, __AID_TRANSFORMS__, None, False), batch_size=__AID_BATCH_SIZE__, shuffle=False)

def loadClientsTrainAID(device, nClients):
    imagePaths, labels = [], []
    X, Y = _loadAID()
    skf = StratifiedKFold(n_splits=__AID_N_SPLIT__, shuffle=True, random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(skf.split(X, np.argmax(Y, axis=1))):
        if i in __AID_TRAIN_SET__:
            imagePaths += X[test_index].tolist()
            labels += Y[test_index].tolist()
    
    imagePaths = np.array(imagePaths)
    labels = np.array(labels)
    
    trainloaders = []
    skf = StratifiedKFold(n_splits=nClients, shuffle=True, random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(skf.split(imagePaths, np.argmax(labels, axis=1))):
        trainloaders.append(torch.utils.data.DataLoader(ImageDatasetFromImagePathsAndLabel(imagePaths[test_index].tolist(), labels[test_index].tolist(), device, __AID_TRANSFORMS__, None, True), batch_size=__AID_BATCH_SIZE__, shuffle=False))
    return trainloaders

def loadCenShadowAID():
    imagePaths, labels = [], []
    X, Y = _loadAID()
    skf = StratifiedKFold(n_splits=__AID_N_SPLIT__, shuffle=True, random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(skf.split(X, np.argmax(Y, axis=1))):
        if i in __AID_SHADOW_SET__:
            imagePaths += X[test_index].tolist()
            labels += Y[test_index].tolist()
    return np.array(imagePaths), np.array(labels)

def loadClientsShadowTrainAID(idx, device, nClients):
    imagePaths, labels = [], []
    X, Y = loadCenShadowAID()
    sss = StratifiedShuffleSplit(n_splits=128, test_size=len(__AID_TRAIN_SET__)/len(__AID_SHADOW_SET__), random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(sss.split(X, np.argmax(Y, axis=1))):
        if i == idx:
            imagePaths += X[test_index].tolist()
            labels += Y[test_index].tolist()
    
    imagePaths = np.array(imagePaths)
    labels = np.array(labels)
    
    trainloaders = []
    skf = StratifiedKFold(n_splits=nClients, shuffle=True, random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(skf.split(imagePaths, np.argmax(labels, axis=1))):
        trainloaders.append(torch.utils.data.DataLoader(ImageDatasetFromImagePathsAndLabel(imagePaths[test_index].tolist(), labels[test_index].tolist(), device, __AID_TRANSFORMS__, None, False), batch_size=__AID_BATCH_SIZE__, shuffle=False))
    return trainloaders

def loadCenShadowTrainAID(idx, device):
    imagePaths, labels = [], []
    X, Y = loadCenShadowAID()
    sss = StratifiedShuffleSplit(n_splits=__AID_N_SHADOW__, test_size=len(__AID_TRAIN_SET__)/len(__AID_SHADOW_SET__), random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(sss.split(X, np.argmax(Y, axis=1))):
        if i == idx:
            imagePaths += X[test_index].tolist()
            labels += Y[test_index].tolist()
    return torch.utils.data.DataLoader(ImageDatasetFromImagePathsAndLabel(imagePaths, labels, device, __AID_TRANSFORMS__, None, True), batch_size=__AID_BATCH_SIZE__, shuffle=False, num_workers=8)

def loadMIADataAID(device):
    imagePaths, labels, memberLabels = [], [], []
    X, Y = _loadAID()
    inOutLabels = np.zeros([len(X), __AID_N_SHADOW__])
    skf = StratifiedKFold(n_splits=__AID_N_SPLIT__, shuffle=True, random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(skf.split(X, np.argmax(Y, axis=1))):
        if i in __AID_MEMBER_SET__+__AID_NON_MEM_SET__:
            imagePaths += X[test_index].tolist()
            labels += Y[test_index].tolist()
            memberLabels += [int(i in __AID_MEMBER_SET__)]*len(test_index)
    return torch.utils.data.DataLoader(ImageDatasetFromImagePathsAndLabel(imagePaths, labels, device, __AID_TRANSFORMS__, None, False), batch_size=__AID_BATCH_SIZE__, shuffle=False, num_workers=8), memberLabels, inOutLabels