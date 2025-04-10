# Imports
import sys
import os
from types import SimpleNamespace
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from tabulate import tabulate

# Adding necessary paths to the system path
sys.path.append('/content/Unlearning-MIA-Eval')

# Local imports
from Final_Structure.datasets import get_loaders
from Final_Structure.training import get_resnet_model, train, load_model
from Final_Structure.scrub import scrub
from Final_Structure.badt import badt
from Final_Structure.ssd import ssd
from Final_Structure.evaluate import evaluate_model, membership_inference_attack

# Initialize variables for experiment loop
unlearn_methods = ['retrain', 'SCRUB', 'BadTeach', 'SSD']
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
class_names = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Train the full version of the ResNet18 model
root_path = '/content/Unlearning-MIA-Eval/Final_Structure/data'
forget_classes = [0]  # Number in here is irrelevant
loaders = get_loaders(root=root_path, forget_classes=forget_classes)
model = get_resnet_model()
full_model_path = f'/content/drive/MyDrive/AIML_Final_Project/checkpoints/resnet_full.pt'
if not os.path.isfile(full_model_path):
    train(model=model, 
        train_loader=loaders[0], 
        criterion=nn.CrossEntropyLoss(), 
        optimizer=optim.Adam(model.parameters(), lr=0.0001), 
        epochs=25, 
        save_path=full_model_path)
else:
    print("No need to train ResNet on full train set, as a checkpoint already exists!")

# Start experiment loop
all_metrics_data = []
all_mia_data = []
for cls in classes:
    # Getting all the loaders
    root_path = '/content/Unlearning-MIA-Eval/Final_Structure/data'
    forget_classes = [cls]
    loaders = get_loaders(root=root_path, forget_classes=forget_classes)

    # Perform unlearning methods
    metrics_data = []
    mia_data = []
    for method in unlearn_methods:

        unl_model = None
        unl_args = SimpleNamespace()
        
        if method == 'retrain':

            unl_model = get_resnet_model()
            unl_args.check_path = f'/content/drive/MyDrive/AIML_Final_Project/checkpoints/retrain_cls_{cls}.pt' 

            if not os.path.isfile(unl_args.check_path):
                train(model=unl_model, 
                    train_loader=loaders[4], 
                    criterion=nn.CrossEntropyLoss(), 
                    optimizer=optim.Adam(unl_model.parameters(), lr=0.0001), 
                    epochs=25, 
                    save_path=unl_args.check_path)
            else:
                print(f'Retrain Unlearning Unnecessary, Checkpoint Exists for Class {cls}')
                unl_model = load_model(unl_args.check_path)
                
        elif method == 'SCRUB':
            
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

    # Save the data for this class for use later
    all_metrics_data.append(metrics_data)
    all_mia_data.append(mia_data)

# Define headers for results tables
performance_cols = ['Forgotten Class', 'Method', 'Accuracy', 'Precision', 'Recall', 'F1 Score']

# Initialize empty DataFrames with proper headers
f_performance_df = pd.DataFrame(columns=performance_cols)
r_performance_df = pd.DataFrame(columns=performance_cols)

# Create performance metrics tables
for i in range(len(classes)): 
    curr_metrics_data = all_metrics_data[i]
    for j, (retain_metrics, forget_metrics) in enumerate(curr_metrics_data): 
    
        class_name = class_names[i]
        method = unlearn_methods[j]

        r_row = {
            'Forgotten Class': class_name,
            'Method': method,
            'Accuracy': retain_metrics['accuracy'],
            'Precision': retain_metrics['precision'],
            'Recall': retain_metrics['recall'],
            'F1 Score': retain_metrics['f1']
        }

        f_row = {
            'Forgotten Class': class_name,
            'Method': method,
            'Accuracy': forget_metrics['accuracy'],
            'Precision': forget_metrics['precision'],
            'Recall': forget_metrics['recall'],
            'F1 Score': forget_metrics['f1']
        }

        # Append rows
        r_performance_df = pd.concat([r_performance_df, pd.DataFrame([r_row])], ignore_index=True)
        f_performance_df = pd.concat([f_performance_df, pd.DataFrame([f_row])], ignore_index=True)

# Print nicely
print(tabulate(r_performance_df, headers='keys', tablefmt='pretty'))
print()
print(tabulate(f_performance_df, headers='keys', tablefmt='pretty'))

# Create the membership inference attack table
mia_cols = []

print()
print(all_mia_data)
