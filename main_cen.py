import torch

import dataAID
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cenTrainDataLoader = dataAID.loadCenTrainAID(device)

cenModel = trainModel(cenTrainDataLoader, device, dataAID.__AID_N_CLASSES__)
modelPredict(cenModel, cenTrainDataLoader, device)

shadowModels = []
for i in range(128):
    shadowDataLoader = dataAID.loadCenShadowTrainAID(i, device)
    shadowModels.append(trainModel(shadowDataLoader, device, dataAID.__AID_N_CLASSES__))