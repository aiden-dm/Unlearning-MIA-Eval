# Imports
import sys
import torch

# Adding necessary paths to the system path
sys.path.append('/content/Unlearning-MIA-Eval')

# Framework imports
from Final_Structure.training import load_model

# Third party imports
from Third_Party_Code.MIADisparity.miae.attacks.calibration_mia import CalibrationAttack, CalibrationModelAccess, CalibrationAuxiliaryInfo
from Third_Party_Code.MIADisparity.experiment.models import ResNet

def get_custom_resnet(dataset):
    num_classes = 10 if dataset == "cifar10" else 100
    num_blocks = [2, 2, 2, 2]                              # ResNet18
    input_size = 32                                        # CIFAR image size

    return ResNet(num_blocks=num_blocks, num_classes=num_classes, input_size=input_size).to('cuda')

def calibration_mia(target_model_path, loaders, config):

    aux_dataset = loaders['train_loader']

    target_model = load_model("cifar10", target_model_path)
    untrained_model = get_custom_resnet("cifar10")
    
    model_access = CalibrationModelAccess(
        model=target_model,
        untrained_model=untrained_model
    )

    aux_info = CalibrationAuxiliaryInfo(config)
    attack = CalibrationAttack(model_access, aux_info)
    
    # Prepare the attack with retained data (members)
    attack.prepare(aux_dataset)

    forget_scores = attack.infer(loaders['train_forget_loader'].dataset)
    retain_scores = attack.infer(loaders['train_retain_loader'].dataset)

    return {
        "forget_scores": forget_scores,
        "retain_scores": retain_scores,
        "threshold": attack.threshold,
        "aux_info": aux_info
    }



    
    