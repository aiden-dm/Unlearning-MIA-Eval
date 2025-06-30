import torch
import torch.nn as nn

from Final_Structure.training import get_resnet_model

def get_training_params_resnet18_cifar10():
    model = get_resnet_model(dataset="cifar10")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return {
        "model": model,
        "optimizer": optimizer,
        "criterion": nn.CrossEntropyLoss(),
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True
        ),
        "epochs": 45
    }

def get_training_params_resnet18_cifar100():
    model = get_resnet_model(dataset="cifar100")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return {
        "model": model,
        "optimizer": optimizer,
        "criterion": nn.CrossEntropyLoss(),
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True
        ),
        "epochs": 65
    }