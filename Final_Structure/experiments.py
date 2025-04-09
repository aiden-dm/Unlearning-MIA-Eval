# Imports
import sys
import os
from types import SimpleNamespace

# Adding necessary paths to the system path
sys.path.append('/content/Unlearning-MIA-Eval')

# Local imports
from Final_Structure.datasets import get_loaders
from Final_Structure.training import train_resnet, load_model
from Final_Structure.scrub import scrub
from Final_Structure.badt import badt
from Final_Structure.ssd import ssd
from Final_Structure.evaluate import evaluate_model, membership_inference_attack

# Initialize variables for experiment loop
unlearn_methods = ['retrain', 'SCRUB', 'BadTeach', 'SSD']
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Start experiment loop
for cls in classes:
    # Getting all the loaders
    root_path = '/content/Unlearning-MIA-Eval/Final_Structure/data'
    forget_classes = [cls]
    loaders = get_loaders(root=root_path, forget_classes=forget_classes)

    # Defining ResNet18 hyperparameters
    args = SimpleNamespace()
    args.learning_rate = 0.0001
    args.epochs = 25
    args.full_path = f"/content/drive/MyDrive/AIML_Final_Project/checkpoints/resnet_full_cls_{cls}.pt"
    args.retain_path = f"/content/drive/MyDrive/AIML_Final_Project/checkpoints/resnet_retain_cls_{cls}.pt"

    # Train the full and retain versions of the ResNet18 Model
    train_bool = not os.path.isfile(args.full_path) and not os.path.isfile(args.retain_path)
    if train_bool:
        print("Training Full and Retain Versions of ResNet18...")
        train_resnet(root='/content/Unlearning-MIA-Eval/Final_Structure/data',
                     dataset='cifar10',
                     forget_classes=[cls],
                     batch_size=32,
                     args=args)
    else:
        print("Training ResNet18 not necessary, can be loaded from checkpoints!")
    
    # Perform unlearning methods
    full_model_path = f'/content/drive/MyDrive/AIML_Final_Project/checkpoints/resnet_full_cls_{cls}.pt'
    metrics_data = []
    mia_data = []
    for method in unlearn_methods:

        unl_model = None
        unl_args = SimpleNamespace()
        
        if method == 'SCRUB':
            
            unl_args.epochs = 10
            unl_args.learning_rate = 0.0001
            unl_args.msteps = 5             # Unlearning is performed from epoch 0-msteps (inclusive)
            unl_args.t_opt_gamma = 1        # Weight for CE loss
            unl_args.t_opt_alpha = 0.5      # Weight for KL divergence
            unl_args.kd_T = 4               # KL divergence temperature
            unl_args.print_accuracies = True
            unl_args.check_path = f'/content/drive/MyDrive/AIML_Final_Project/checkpoints/scrub_cls_{cls}.pt'
            
            if not os.path.isfile(unl_args.check_path):
                print(f'Performing SCRUB Unlearning on Class {cls}')
                unl_model, _ = scrub(full_model_path, loaders, unl_args)
            else:
                print(f'SCRUB Unlearning Unnecessary, Checkpoint Exists for Class {cls}')
                unl_model = load_model(unl_args.check_path)
        
        elif method == 'BadTeach':
            
            unl_args.epochs = 10
            unl_args.learning_rate = 0.0001
            unl_args.KL_temperature = 4
            unl_args.batch_size = 256
            unl_args.print_accuracies = True
            unl_args.check_path = f'/content/drive/MyDrive/AIML_Final_Project/checkpoints/badt_cls_{cls}.pt'

            if not os.path.isfile(unl_args.check_path):
                print(f'Performing BadTeach Unlearning on Class {cls}')
                unl_model, _ = badt(full_model_path, loaders, unl_args)
            else:
                print(f'BadTeach Unlearning Unnecessary, Checkpoint Exists for Class {cls}')
                unl_model = load_model(unl_args.check_path)
        
        else:
            
            unl_args.learning_rate = 0.0001      # Doesn't impact unlearning
            unl_args.dampening_constant = 1      # Parameter used in the paper
            unl_args.selection_weighting = 10    # Parameter used in the paper
            unl_args.check_path = f'/content/drive/MyDrive/AIML_Final_Project/checkpoints/ssd_cls_{cls}.pt'
            
            if not os.path.isfile(unl_args.check_path):
                print(f'Performing SSD Unlearning on Class {cls}')
                unl_model, _ = ssd(full_model_path, loaders, unl_args)
            else:
                print(f'SSD Unlearning Unnecessary, Checkpoint Exists for Class {cls}')
                unl_model = load_model(unl_args.check_path)

        # Get performance evaluation information
        test_loader = loaders[2]
        test_retain_loader = loaders[8]
        test_forget_loader = loaders[7]
        retain_metrics = evaluate_model(unl_model, test_retain_loader, 'cuda')
        forget_metrics = evaluate_model(unl_model, test_forget_loader, 'cuda')
        mia_mean_metrics, mia_std_metrics = membership_inference_attack(
            unl_model,
            test_loader,
            test_forget_loader,
            device='cuda',
            seed=42
        )
        metrics_data.append((retain_metrics, forget_metrics))
        mia_data.append((mia_mean_metrics, mia_std_metrics))

    print(metrics_data)
    print(mia_data)