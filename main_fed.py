import torch
import dataAID
from utils import *

nClients = 10
localEpochs = 10
rounds = 10
backbone = "resnet18"
n_classes = dataAID.__AID_N_CLASSES__
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainLoaders = dataAID.loadClientsTrainAID(device, nClients)
validLoaders = dataAID.loadClientsTrainAID(device, nClients)
miaDataLoader = dataAID.loadMIADataAID(device)

def client_fn(cid) -> FlowerClient:
    net = ModelFromBackbone(backbone, n_classes)
    trainLoader = trainLoaders[int(cid)]
    validLoader = validLoaders[int(cid)]
    return FlowerClient(cid, net, device, trainLoader, validLoader, localEpochs).to_client()

client_resources = None
if device.type == "cuda":
    client_resources = {"num_cpus": 8, "num_gpus": 1}

strategy = FLSetup(n_classes, device, backbone, nClients)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=nClients,
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
    client_resources=client_resources,
)

yPred = FLModelPredict(strategy.parameters_aggregated, n_classes, backbone, miaDataLoader, device)

shadowPreds = []
shadowModels = []
for i in range(128):
    strategy = FLSetup(n_classes, device, backbone, nClients)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=nClients,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    shadowPreds.append(FLModelPredict(strategy.parameters_aggregated, n_classes, backbone, miaDataLoader, device))

scores = computeMIAScore(yPred, shadowPreds)
print(np.mean((scores > 0.5) == np.array([0]*(len(scores)//2)+[1]*(len(scores)//2))))