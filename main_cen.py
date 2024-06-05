import torch
import pickle
import dataAID
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cenTrainDataLoader = dataAID.loadCenTrainAID(device)
miaDataLoader = dataAID.loadMIADataAID(device)

cenModel = trainModel(cenTrainDataLoader, device, dataAID.__AID_N_CLASSES__)
yPred = modelPredict(cenModel, miaDataLoader, device)

shadowPreds = []
shadowModels = []
for i in range(5):
    shadowDataLoader = dataAID.loadCenShadowTrainAID(i, device)
    shadowModels.append(trainModel(shadowDataLoader, device, dataAID.__AID_N_CLASSES__))
    shadowPreds.append(modelPredict(shadowModels[-1], miaDataLoader, device))

scores = computeMIAScore(yPred, shadowPreds)
print(np.mean((scores > 0.5) == np.array([0]*(len(scores)//2)+[1]*(len(scores)//2))))