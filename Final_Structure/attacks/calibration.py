# Imports
import sys
import torch

# Adding necessary paths to the system path
sys.path.append('/content/Unlearning-MIA-Eval')

# Framework imports
from Final_Structure.training import load_model, get_resnet_model

# Third party imports
from Third_Party_Code.MIADisparity.miae.attacks.calibration_mia import CalibrationAttack, CalibrationModelAccess, CalibrationAuxiliaryInfo

def calibration_mia(target_model_path, loaders):

    train_loader = loaders['train_loader']
    valid_loader = loaders['valid_loader']
    test_forget_loader = loaders['test_forget_loader']

    aux_dataset = torch.utils.data.ConcatDataset([
        train_loader.dataset,
        valid_loader.dataset
    ])

    target_model = load_model("cifar10", target_model_path)

    untrained_model = get_resnet_model("cifar10")

    model_access = CalibrationModelAccess(
        model=target_model,
        untrained_model=untrained_model
    )

    config = {
        "seed": 42,
        "batch_size": 256,
        "num_classes": 10,
        "lr": 0.001,
        "epochs": 20,
        "num_shadow_models": 1,
        "shadow_train_ratio": 0.5,
        "save_path": "./mia_calibration",
        "log_path": "./mia_calibration/logs",
        "device": "cuda"
    }

    aux_info = CalibrationAuxiliaryInfo(config)
    attack = CalibrationAttack(model_access, aux_info)

    attack.prepare(aux_dataset)
    
    inference_scores = attack.infer(test_forget_loader.dataset)

    return {
        "scores": inference_scores,
        "threshold": attack.threshold,
        "aux_info": aux_info
    }



    
    