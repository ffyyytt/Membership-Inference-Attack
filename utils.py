from model.ModelFromBackbone import *
from model.unit import *

def trainModel(dataLoader, device, n_classes, backbone = "mobilenet_v3_small"):
    model = ModelFromBackbone(backbone, n_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    loss_fn=torch.nn.CrossEntropyLoss().to(device)

    train_unit = MyTrainUnit(module=model, optimizer=optimizer, lr_scheduler=scheduler, loss_fn=loss_fn)
    torchtnt.framework.train(train_unit, dataLoader, max_epochs=20)
    return model