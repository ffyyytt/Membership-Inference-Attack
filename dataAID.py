import os
import glob
import torch
import torchvision

import numpy as np

from data.ImageDatasetFromImagePathsAndLabel import *
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

__RANDOM__SEED__ = 1312

# 0, 1: Train
# 2: non-mem
# 3-7: Shadow

__AID_N_CLASSES__ = 30
__AID_N_SPLIT__ = 8
__AID_TRAIN_SET__ = [0, 1]
__AID_NON_MEM_SET__ = [2]
__AID_SHADOW__SET = [3, 4, 5, 6, 7]
__AID_BATCH_SIZE__ = 64
__AID_TRANSFORMS__ = torchvision.transforms.v2([
    torchvision.transforms.v2.RandomResizedCrop(size=(224, 224), antialias=True),
    torchvision.transforms.v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        labelset[k] = [0]*v + [1] + [0]*(len(labelset)-v-1)

    # map
    imagePaths = np.array(imagePaths)
    labels = np.array([labelset[label] for label in labels])

    return imagePaths, labels

def loadCenTrainAID():
    imagePaths, labels = [], []
    X, Y = _loadAID()
    skf = StratifiedKFold(n_splits=8, random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
        if i in __AID_SHADOW__SET:
            imagePaths += X[test_index]
            labels += Y[test_index]
    return torch.utils.data.DataLoader(ImageDatasetFromImagePathsAndLabel(imagePaths, labels, __AID_TRANSFORMS__), batch_size=__AID_BATCH_SIZE__)

def loadCenShadowAID():
    imagePaths, labels = [], []
    X, Y = _loadAID()
    skf = StratifiedKFold(n_splits=8, random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
        if i in __AID_TRAIN_SET__:
            imagePaths += X[test_index]
            labels += Y[test_index]
    return imagePaths, labels

def loadCenShadowTrainAID(idx):
    imagePaths, labels = [], []
    X, Y = loadCenShadowAID()
    sss = StratifiedShuffleSplit(n_splits=128, test_size=len(__AID_TRAIN_SET__)/len(__AID_SHADOW__SET), random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(sss.split(X, Y)):
        if i == idx:
            imagePaths += X[test_index]
            labels += Y[test_index]
    return torch.utils.data.DataLoader(ImageDatasetFromImagePathsAndLabel(imagePaths, labels, __AID_TRANSFORMS__), batch_size=__AID_BATCH_SIZE__)