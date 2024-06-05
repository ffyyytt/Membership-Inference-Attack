import torch

import flwr as fl
import numpy as np

from sklearn.metrics import mean_squared_error

from typing import *
from model.unit import *

def FLget_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def FLset_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict if len(v.shape)>0})
    net.load_state_dict(state_dict, strict=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, device, trainloader, valloader, localEpochs):
        self.cid = cid
        self.net = net
        self.device = device
        self.trainloader = trainloader
        self.valloader = valloader
        self.localEpochs = localEpochs

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return FLget_parameters(self.net)

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        FLset_parameters(self.net, parameters)
        self.__FLtrainModelWithModel(self.net, self.trainloader, self.device , epochs=self.localEpochs)
        return FLget_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        FLset_parameters(self.net, parameters)
        output, label = self.__FLmodelPredict(self.net, self.valloader, self.device)
        loss = mean_squared_error(label, output)
        accuracy = np.mean(np.argmax(output, axis=1)==np.argmax(label, axis=1))
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
    
    def __FLtrainModelWithModel(self, model, dataLoader, device, epochs):
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        loss_fn=torch.nn.CrossEntropyLoss().to(device)

        train_unit = MyTrainUnit(module=model, optimizer=optimizer, lr_scheduler=scheduler, loss_fn=loss_fn, totalSteps=len(dataLoader))
        torchtnt.framework.train(train_unit, dataLoader, max_epochs=epochs)
        return model

    def __FLmodelPredict(self, model, dataLoader, device):
        model = model.to(device)
        predUnit = MyPredictUnit(module=model)
        torchtnt.framework.predict(predUnit, dataLoader)
        return predUnit.outputs, predUnit.labels