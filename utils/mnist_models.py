import torch.nn as nn

def modelA():
    model = nn.Sequential(
        nn.Conv2d(1,64,(5,5), padding='valid'),
        nn.ReLU(),
        nn.Conv2d(64,64,(5,5)),
        nn.ReLU(),
        nn.Dropout2d(0.25),
        nn.Flatten(),
        nn.Linear(25600,128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128,10),
    )
    return model

def modelB():
    model = nn.Sequential(
        nn.Dropout2d(0.2),
        nn.Conv2d(1,64,(8,8), padding=(3,3), stride=(2,2)),
        nn.ReLU(),
        nn.Conv2d(64,128,(6,6), padding='valid', stride=(2,2)),
        nn.ReLU(),
        nn.Conv2d(128,128,(5,5), stride=(1,1)),
        nn.ReLU(),
        nn.Dropout2d(0.5),
        nn.Flatten(),
        nn.Linear(128,10),
    )
    return model

def modelC():
    model = nn.Sequential(
        nn.Conv2d(1,128,(3,3), padding='valid'),
        nn.ReLU(),
        nn.Conv2d(128,64,(3,3)),
        nn.ReLU(),
        nn.Dropout2d(0.25),
        nn.Flatten(),
        nn.Linear(36864,128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128,10)
    )
    return model

def modelD():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 300),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(300, 300),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(300, 300),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(300, 300),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(300, 10),
    )

    return model

def modelE():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )

    return model

def modelF():
    model = nn.Sequential(
        nn.Conv2d(1, 32, (5,5,), padding='valid'),
        nn.ReLU(),
        nn.MaxPool2d((2,2)),
        nn.Conv2d(32, 64, (5,5)),
        nn.ReLU(),
        nn.MaxPool2d((2,2)),
        nn.Flatten(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
    )

    return model

def modelG():
    model = nn.Sequential(
        nn.Conv2d(1, 32, (5,5), padding='same'),
        nn.ReLU(),
        nn.Conv2d(32, 32, (5,5), padding='same'),
        nn.ReLU(),
        nn.MaxPool2d((2,2)),
        nn.Dropout2d(0.25),
        nn.Conv2d(32, 64, (3,3), padding='same'),
        nn.ReLU(),
        nn.Conv2d(64, 64, (3,3), padding='same'),
        nn.ReLU(),
        nn.MaxPool2d((2,2), stride=(2,2)),
        nn.Dropout2d(0.25),
        nn.Flatten(),
        nn.Linear(3136, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 10)
    )

    return model

def ModelLR():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 10),
    )

    return model