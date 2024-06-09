import time
import torch
import dataAID
from utils import *
from sklearn.metrics import roc_auc_score

nClients = 10
localEpochs = 10
rounds = 10
backbone = "resnet18"
n_classes = dataAID.__AID_N_CLASSES__
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainLoaders = dataAID.loadClientsTrainAID(device, nClients)
validLoaders = dataAID.loadClientsTrainAID(device, nClients)
miaDataLoader, memberLabels = dataAID.loadMIADataAID(device)

def client_fn(cid):
    net = ModelFromBackbone(backbone, n_classes)
    if os.path.isfile(backbone+".weight"):
        net.load_state_dict(torch.load(backbone+".weight"))
    else:
        torch.save(net.state_dict(), backbone+".weight")
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
for i in range(32):
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
print(f"\n\nAttack: {roc_auc_score(memberLabels, scores)}\n\n")
for thr in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
    print(f"\n\nFPR at {thr} TPR: {TPRatFPR(memberLabels, scores, thr)}\n\n")