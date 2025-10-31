import sys
import torch
from torch.utils.data import ConcatDataset
from Third_Party_Code.MIADisparity.experiment.models import ResNet
import torch.serialization
torch.serialization.add_safe_globals([ResNet])

sys.path.append('/content/Unlearning-MIA-Eval')

from Final_Structure.training import load_model

# Import the necessary classes directly from the file
from Third_Party_Code.MIADisparity.miae.attacks.aug_mia import AttackModel, AugAuxiliaryInfo, AugAttack, AugModelAccess
from Third_Party_Code.MIADisparity.experiment.models import ResNet

def get_custom_resnet(dataset):
    num_classes = 10 if dataset == "cifar10" else 100
    num_blocks = [2, 2, 2, 2]                              # ResNet18
    input_size = 32                                        # CIFAR image size

    return ResNet(num_blocks=num_blocks, num_classes=num_classes, input_size=input_size).to('cuda')

def aug_mia(target_model_path, loaders, config):
    train_retain_loader = loaders['train_retain_loader']
    train_forget_loader = loaders['train_forget_loader']
    valid_forget_loader = loaders['valid_forget_loader']
    test_forget_loader = loaders['test_forget_loader']

    aux_dataset = train_retain_loader.dataset

    target_model = load_model("cifar10", target_model_path)
    untrained_model = get_custom_resnet("cifar10")

    model_access = AugModelAccess(
        model = target_model,
        untrained_model= untrained_model
    )

    aux_info = AugAuxiliaryInfo(config)
    attack = AugAttack(model_access, aux_info)

    # Prepare the attack with retained data (members)
    # Modify the prepare method's logic to include weights_only=False when loading the shadow model
    # Note: This assumes the shadow model loading happens within the prepare method's internal logic.
    # We are reproducing the logic from the provided aug_mia.py snippet here for clarity.
    import os
    if os.path.exists(aux_info.shadow_model_path + '/shadow_model.pth'):
        # Added weights_only=False to address the UnpicklingError
        shadow_model = torch.load(aux_info.shadow_model_path + '/shadow_model.pth', weights_only=False)
        attack.shadow_model = shadow_model # Assuming the loaded model is assigned to a shadow_model attribute
    else:
        # Original prepare logic for training the shadow model if it doesn't exist
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
    }