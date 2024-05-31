import torch
import torchtnt

import dataAID
from model.ModelFromBackbone import *
from model.unit import *

cenTrainDataLoader = dataAID.loadCenTrainAID()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ModelFromBackbone("resnet18", dataAID.__AID_N_CLASSES__).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
loss_fn=torch.nn.CrossEntropyLoss().to(device)

train_unit = MyTrainUnit(module=model, optimizer=optimizer, lr_scheduler=scheduler, loss_fn=loss_fn)
torchtnt.framework.train(train_unit, cenTrainDataLoader, max_epochs=20, device=device)

