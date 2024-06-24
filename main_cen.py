import gc
import torch
import dataCIFARonline
from utils import *
from sklearn.metrics import roc_auc_score
import torchvision.transforms as transforms

backbone = "resnet18"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cenTrainDataLoader = dataCIFARonline.loadCenTrainCIFAR10(device)

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# batch_size = 32

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# cenTrainDataLoader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True, num_workers=2)

miaDataLoader, memberLabels, inOutLabels = dataCIFARonline.loadMIADataCIFAR10(device)

cenModel = trainModel(cenTrainDataLoader, device, dataCIFARonline.__CIFAR10_N_CLASSES__, backbone)
yPred = modelPredict(cenModel, miaDataLoader, device)

print(yPred[0].shape, yPred[1].shape)
print(np.argmax(yPred[0], axis=1))
print(np.argmax(yPred[1], axis=1))

print(np.mean(np.argmax(yPred[0], axis=1) == np.argmax(yPred[1], axis=1)))

shadowPreds = []
shadowModels = []
for i in range(dataCIFARonline.__CIFAR10_N_SHADOW__):
    shadowDataLoader = dataCIFARonline.loadCenShadowTrainCIFAR10(i, device)
    shadowModels.append(trainModel(shadowDataLoader, device, dataCIFARonline.__CIFAR10_N_CLASSES__, verbose=0))
    shadowPreds.append(modelPredict(shadowModels[-1], miaDataLoader, device, verbose=False))
    gc.collect()
    torch.cuda.empty_cache()
    if (i > 10):
        scores = computeMIAScore(yPred, shadowPreds, inOutLabels)
        print(f"Attack: {roc_auc_score(memberLabels, scores)}")
        print(f"TPR at {0.001} FPR: {TPRatFPR(memberLabels, scores, 0.001)}")
    

print(inOutLabels)
scores = computeMIAScore(yPred, shadowPreds, inOutLabels)
print(f"Attack: {roc_auc_score(memberLabels, scores)}")
for thr in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
    print(f"TPR at {thr} FPR: {TPRatFPR(memberLabels, scores, thr)}")