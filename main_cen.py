import torch
import pickle
import dataAID
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cenTrainDataLoader = dataAID.loadCenTrainAID(device)

cenModel = trainModel(cenTrainDataLoader, device, dataAID.__AID_N_CLASSES__)
print(modelPredict(cenModel, cenTrainDataLoader, device))
with open('pred.pickle', 'wb') as handle:
    pickle.dump(modelPredict(cenModel, cenTrainDataLoader, device), handle, protocol=pickle.HIGHEST_PROTOCOL)

shadowModels = []
for i in range(128):
    shadowDataLoader = dataAID.loadCenShadowTrainAID(i, device)
    shadowModels.append(trainModel(shadowDataLoader, device, dataAID.__AID_N_CLASSES__))