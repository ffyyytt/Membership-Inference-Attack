from model.ModelFromBackbone import *
from model.unit import *


def trainModel(dataLoader, device, n_classes, backbone = "mobilenet_v2"):
    model = ModelFromBackbone(backbone, n_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    loss_fn=torch.nn.CrossEntropyLoss().to(device)

    train_unit = MyTrainUnit(module=model, optimizer=optimizer, lr_scheduler=scheduler, loss_fn=loss_fn, totalSteps=len(dataLoader))
    torchtnt.framework.train(train_unit, dataLoader, max_epochs=20)
    return model

def modelPredict(model, dataLoader, device):
    model = model.to(device)
    predUnit = MyPredictUnit(module=model)
    torchtnt.framework.predict(predUnit, dataLoader)
    print(np.mean(np.argmax(predUnit.outputs, axis=1) == np.argmax(predUnit.labels, axis=1)))
    return predUnit.outputs