import torch
import dataAID
from utils import *

nClients = 10
backbone = "resnet18"
n_classes = dataAID.__AID_N_CLASSES__
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainLoaders = dataAID.loadClientsTrainAID(device, nClients)
validLoaders = dataAID.loadClientsTrainAID(device, nClients)

def client_fn(cid) -> FlowerClient:
    net = ModelFromBackbone(backbone, n_classes)
    trainLoader = trainLoaders[int(cid)]
    validLoader = validLoaders[int(cid)]
    return FlowerClient(cid, net, device, trainLoader, validLoader, 5).to_client()

client_resources = None
if device.type == "cuda":
    client_resources = {"num_cpus": 8, "num_gpus": 4}

strategy = FLSetup(n_classes, device, backbone, nClients, 10)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=nClients,
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
    client_resources=client_resources,
)

shadowPreds = []
shadowModels = []
for i in range(128):
    strategy = FLSetup(n_classes, device, backbone, nClients, 10)

    print(fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=nClients,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        client_resources=client_resources,
    ))
