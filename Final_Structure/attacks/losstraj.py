import sys
import torch
from torch.utils.data import ConcatDataset

sys.path.append('/content/Unlearning-MIA-Eval')

from Final_Structure.training import load_model

from Third_Party_Code.MIADisparity.miae.attacks.losstraj_mia import LosstrajAuxiliaryInfo, LosstrajAttack, LosstrajModelAccess
from Third_Party_Code.MIADisparity.experiment.models import ResNet

def get_custom_resnet(dataset):
    num_classes = 10 if dataset == "cifar10" else 100
    num_blocks = [2, 2, 2, 2]                              # ResNet18
    input_size = 32                                        # CIFAR image size

    return ResNet(num_blocks=num_blocks, num_classes=num_classes, input_size=input_size).to('cuda')

def losstraj_mia(target_model_path, loaders, config):
    train_retain_loader = loaders['train_retain_loader']
    train_forget_loader = loaders['train_forget_loader']
    valid_forget_loader = loaders['valid_forget_loader']
    test_forget_loader = loaders['test_forget_loader']

    aux_dataset = train_retain_loader.dataset

    target_model = load_model("cifar10", target_model_path)
    untrained_model = get_custom_resnet("cifar10")

    model_access = LosstrajModelAccess(
        model = target_model,
        untrained_model= untrained_model
    )

    aux_info = LosstrajAuxiliaryInfo(config)
    attack = LosstrajAttack(model_access, aux_info)

    # Prepare the attack with retained data (members)
    attack.prepare(aux_dataset)

    unseen_dataset = ConcatDataset([valid_forget_loader.dataset, test_forget_loader.dataset])

    forget_scores = attack.infer(train_forget_loader.dataset)
    retain_scores = attack.infer(train_retain_loader.dataset)
    unseen_scores = attack.infer(unseen_dataset)
    
    return {
        "forget_scores": forget_scores,
        "retain_scores": retain_scores,
        "unseen_scores": unseen_scores,
        "threshold": attack.threshold,
        "aux_info": aux_info
    }
    



