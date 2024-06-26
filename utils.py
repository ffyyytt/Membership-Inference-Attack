import os
import scipy
import random
import flwr as fl

from sklearn.metrics import roc_curve

from model.FLStrategies import *
from model.unit import *
from model.FLClient import *
from model.FLFeatureClient import *
from model.ModelFromBackbone import *

__LR__ = 1e-2
__MOMENTUM__ = 0.9

def seedBasic(seed=1312):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def trainTFModel(dataLoader, strategy, n_classes, backbone="resnet18", epochs=50, verbose=2):
    with strategy.scope():
        model = modelTF(backbone, n_classes)
        optimizer = tf.keras.optimizers.SGD(learning_rate = __LR__, momentum=__MOMENTUM__)

        model.compile(optimizer = optimizer,
                    loss = {'output': tf.keras.losses.CategoricalCrossentropy()},
                    metrics = {"output": [tf.keras.metrics.CategoricalAccuracy()]})
        
    H = model.fit(dataLoader,
                  verbose = verbose,
                  epochs = epochs)
    return model

def trainModel(dataLoader, device, n_classes, backbone = "resnet18", epochs = 50, verbose=2):
    model = ModelFromBackbone(backbone, n_classes, device)
    if os.path.isfile(backbone+".weight"):
        model.load_state_dict(torch.load(backbone+".weight"))
    else:
        torch.save(model.state_dict(), backbone+".weight")
    model = torch.nn.DataParallel(model)
    trainModelWithModel(dataLoader, device, model, epochs, verbose)
    return model

def trainModelWithModel(dataLoader, device, model, epochs, verbose=2):
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=__LR__, momentum=0.9, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    loss_fn=torch.nn.CrossEntropyLoss().to(device)
    model.train()
    train_unit = MyTrainUnit(module=model, optimizer=optimizer, lr_scheduler=scheduler, loss_fn=loss_fn, totalSteps=len(dataLoader), totalEpochs=epochs, verbose=verbose, device=device)
    torchtnt.framework.train(train_unit, dataLoader, max_epochs=epochs)
    return model

def FLModelPredict(parameters, n_classes, backbone, dataLoader, device):
    model = ModelFromBackbone(backbone, n_classes, device)
    FLset_parameters(model, parameters_to_ndarrays(parameters))
    model = model.to(device)
    predUnit = MyPredictUnit(module=model)
    torchtnt.framework.predict(predUnit, dataLoader)
    return predUnit.outputs, predUnit.labels

def modelPredict(model, dataLoader, device, verbose=True):
    model = model.to(device)
    model.eval()
    predUnit = MyPredictUnit(module=model, verbose=verbose)
    torchtnt.framework.predict(predUnit, dataLoader)
    return predUnit.outputs, predUnit.labels

def probabilityNormalDistribution(data, p, eps=1e-6):
    if len(data) == 0:
        return 0.0
    mean = np.mean(data)
    std = max(np.std(data), eps)
    return scipy.stats.norm.cdf((p - mean) / std)

def LiRAcalculation(p, data, inOutLabel, eps=1e-6):
    truthIdxs = np.where(inOutLabel[:len(data)]==1)[0]
    falseIdxs = np.where(inOutLabel[:len(data)]==0)[0]
    if len(truthIdxs) == 0:
        return 1-probabilityNormalDistribution(data, p)
    else:
        return probabilityNormalDistribution(data[truthIdxs], p)/max(probabilityNormalDistribution(data[falseIdxs], p), eps)

def minMaxScale(data):
    data = np.array(data)
    return (data-min(data))/(max(data)-min(data))

def computeMIAScore(yPred, shadowPreds, inOutLabels):
    trueYPred = np.max(yPred[0]*yPred[1], axis=1)
    trueShadowPred = np.array([np.max(shadowPreds[i][0]*shadowPreds[i][1], axis=1) for i in range(len(shadowPreds))])
    scores = [LiRAcalculation(trueYPred[i], trueShadowPred[:, i], inOutLabels[i]) for i in range(len(trueYPred))]
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