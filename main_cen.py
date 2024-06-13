import torch
import dataCIFARonline
from utils import *
from sklearn.metrics import roc_auc_score

backbone = "resnet18"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cenTrainDataLoader = dataCIFARonline.loadCenTrainCIFAR10(device)
miaDataLoader, memberLabels, inOutLabels = dataCIFARonline.loadMIADataCIFAR10(device)

cenModel = trainModel(cenTrainDataLoader, device, dataCIFARonline.__CIFAR10_N_CLASSES__, backbone)
yPred = modelPredict(cenModel, miaDataLoader, device)

shadowPreds = []
shadowModels = []
for i in trange(dataCIFARonline.__CIFAR10_N_SHADOW__):
    shadowDataLoader = dataCIFARonline.loadCenShadowTrainCIFAR10(i, device)
    shadowModels.append(trainModel(shadowDataLoader, device, dataCIFARonline.__CIFAR10_N_CLASSES__, verbose=0))
    shadowPreds.append(modelPredict(shadowModels[-1], miaDataLoader, device))

scores = computeMIAScore(yPred, shadowPreds, inOutLabels)
print(f"\n\nAttack: {roc_auc_score(memberLabels, scores)}\n\n")
for thr in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
    print(f"\n\nTPR at {thr} FPR: {TPRatFPR(memberLabels, scores, thr)}\n\n")