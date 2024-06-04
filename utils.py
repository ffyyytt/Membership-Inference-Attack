import scipy
import flwr as fl

from model.ModelFromBackbone import *
from model.unit import *

def trainModel(dataLoader, device, n_classes, backbone = "mobilenet_v2", epochs = 5):
    model = ModelFromBackbone(backbone, n_classes)
    trainModelWithModel(dataLoader, device, model, epochs)
    return model

def trainModelWithModel(dataLoader, device, model, epochs):
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    loss_fn=torch.nn.CrossEntropyLoss().to(device)

    train_unit = MyTrainUnit(module=model, optimizer=optimizer, lr_scheduler=scheduler, loss_fn=loss_fn, totalSteps=len(dataLoader))
    torchtnt.framework.train(train_unit, dataLoader, max_epochs=epochs)
    return model

def modelPredict(model, dataLoader, device):
    model = model.to(device)
    predUnit = MyPredictUnit(module=model)
    torchtnt.framework.predict(predUnit, dataLoader)
    return predUnit.outputs, predUnit.labels

def probabilityNormalDistribution(data, p):
    mean = np.mean(data)
    std = np.std(data)    
    return scipy.stats.norm.cdf((p - mean) / std)

def computeMIAScore(yPred, shadowPreds):
    trueYPred = yPred[0][np.argmax(yPred[1], axis=1)]
    trueShadowPred = np.array([shadowPreds[i][0][np.argmax(shadowPreds[i][1], axis=1)] for i in range(len(shadowPreds))])
    scores = [1-probabilityNormalDistribution(yPred[i], trueShadowPred[:, i]) for i in range(len(shadowPreds))]
    return scores