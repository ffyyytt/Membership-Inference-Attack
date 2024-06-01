import torch

import dataAID
from utils import *

cenTrainDataLoader = dataAID.loadCenTrainAID()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainModel(cenTrainDataLoader, device, dataAID.__AID_N_CLASSES__)

shadowModels = []
for i in range(128):
    shadowDataLoader = dataAID.loadCenShadowTrainAID(i)
    shadowModels.append(trainModel(shadowDataLoader, device, dataAID.__AID_N_CLASSES__))