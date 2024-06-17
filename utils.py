import os
import scipy
import flwr as fl

from sklearn.metrics import roc_curve

from model.FLStrategies import *
from model.unit import *
from model.FLClient import *
from model.FLFeatureClient import *
from model.ModelFromBackbone import *

def trainModel(dataLoader, device, n_classes, backbone = "resnet18", epochs = 50, verbose=2):
    model = ModelFromBackbone(backbone, n_classes, device)
    if os.path.isfile(backbone+".weight"):
        model.load_state_dict(torch.load(backbone+".weight"))
    else:
        torch.save(model.state_dict(), backbone+".weight")
    trainModelWithModel(dataLoader, device, model, epochs, verbose)
    return model

def trainModelWithModel(dataLoader, device, model, epochs, verbose=2):
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    loss_fn=torch.nn.CrossEntropyLoss().to(device)

    train_unit = MyTrainUnit(module=model, optimizer=optimizer, lr_scheduler=scheduler, loss_fn=loss_fn, totalSteps=len(dataLoader), totalEpochs=epochs, verbose=verbose)
    torchtnt.framework.train(train_unit, dataLoader, max_epochs=epochs)
    return model

def FLModelPredict(parameters, n_classes, backbone, dataLoader, device):
    model = ModelFromBackbone(backbone, n_classes, device)
    FLset_parameters(model, parameters_to_ndarrays(parameters))
    model = model.to(device)
    predUnit = MyPredictUnit(module=model)
    torchtnt.framework.predict(predUnit, dataLoader)
    return predUnit.outputs, predUnit.labels

def modelPredict(model, dataLoader, device):
    model = model.to(device)
    predUnit = MyPredictUnit(module=model)
    torchtnt.framework.predict(predUnit, dataLoader)
    return predUnit.outputs, predUnit.labels

def probabilityNormalDistribution(data, p, eps=1e-6):
    mean = np.mean(data)
    std = max(np.std(data), eps)
    return scipy.stats.norm.cdf((p - mean) / std)

def LiRAcalculation(p, data, inOutLabel):
    truthIdxs = np.index(inOutLabel==1)[0]
    falseIdxs = np.index(inOutLabel==0)[0]
    if len(truthIdxs) == 0:
        return 1-probabilityNormalDistribution(data, p)
    else:
        return probabilityNormalDistribution(data[truthIdxs], p)/probabilityNormalDistribution(data[falseIdxs], p)

def minMaxScale(data):
    data = np.array(data)
    return (data-min(data))/(max(data)-min(data))

def computeMIAScore(yPred, shadowPreds, inOutLabels):
    trueYPred = np.max(yPred[0]*yPred[1], axis=1)
    trueShadowPred = np.array([np.max(shadowPreds[i][0]*shadowPreds[i][1], axis=1) for i in range(len(shadowPreds))])
    scores = [LiRAcalculation(trueShadowPred[:, i], trueYPred[i], inOutLabels[i]) for i in range(len(trueYPred))]
    return scores

def TPRatFPR(y_true, y_score, target_fpr = 0.01):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    tpr_at_target_fpr = tpr[np.where(fpr >= target_fpr)[0][0]]

    return tpr_at_target_fpr


def FLSetup(n_classes, device, backbone = "resnet18", nClients=10):
    params = FLget_parameters(ModelFromBackbone(backbone, n_classes, device))
    strategy = MyFedAVG(
        fraction_fit=1.,
        fraction_evaluate=1.,
        min_fit_clients=nClients,
        min_evaluate_clients=nClients,
        min_available_clients=nClients,
        initial_parameters=fl.common.ndarrays_to_parameters(params),
    )
    return strategy