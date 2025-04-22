# Imports
import sys
import os
from types import SimpleNamespace
import torch
import torch.nn as nn
import pandas as pd
from tabulate import tabulate
import argparse

# Adding necessary paths to the system path
sys.path.append('/content/Unlearning-MIA-Eval')

# Local imports
from Final_Structure.datasets import get_loaders
from Final_Structure.training import get_resnet_model, train, load_model
from Final_Structure.scrub import scrub
from Final_Structure.badt import badt
from Final_Structure.ssd import ssd
from Final_Structure.evaluate import evaluate_model, membership_inference_attack

def init_cifar100_params(experiment_params, dataset):
    # Add dataset specific variables
    forget_lists = [list(range(i, i + 10)) for i in range(0, 100, 10)]
    forget_lists_strings = [f"{row[0]}-{row[-1]}" for row in forget_lists]
    experiment_params['dataset'] = {
        'name': 'cifar100',
        'forget_lists': forget_lists,
        'forget_lists_strings': forget_lists_strings
    }

    # ResNet18 parameters for full train dataset
    model = get_resnet_model(dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    experiment_params["full_train"] = {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'scheduler': scheduler,
        'epochs': 65
    }
    
    # ResNet18 parameters for train retain dataset
    model = get_resnet_model(dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    experiment_params["retrain"] = {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'scheduler': scheduler,
        'epochs': 65
    }

    # SCRUB parameters
    args = SimpleNamespace()
    args.epochs = 20
    args.learning_rate = 0.0001
    args.msteps = 10
    args.t_opt_gamma = 1
    args.t_opt_alpha = 0.0
    args.kd_T = 2
    args.print_accuracies = True
    args.dataset = 'cifar100'
    experiment_params['scrub'] = {
        'args': args
    }

    # BadTeach parameters
    args = SimpleNamespace()
    args.epochs = 20
    args.learning_rate = 0.001
    args.KL_temperature = 4
    args.batch_size = 256
    args.print_accuracies = True
    args.dataset = 'cifar100'
    experiment_params['badt'] = {
        'args': args
    }

    # SSD parameters
    args = SimpleNamespace()
    args.learning_rate = 0.001
    args.dampening_constant = 10
    args.selection_weighting = 1
    args.check_path = '/content/Unlearning-MIA-Eval/Final_Structure/checkpoints/ssd.pt'
    args.print_accuracies = True
    args.dataset = 'cifar100'
    experiment_params['ssd'] = {
        'args': args
    }

def create_post_unlearn_tables(unlearn_methods, forget_lists_strings, all_metrics_data):
    # Define headers for results tables
    performance_cols = ['Forgotten Class', 'Method', 'Accuracy', 'Precision', 'Recall', 'F1 Score']

    # Initialize empty DataFrames with proper headers
    f_performance_df = pd.DataFrame(columns=performance_cols)
    r_performance_df = pd.DataFrame(columns=performance_cols)

    # Create performance metrics tables
    for i in range(len(forget_lists_strings)): 
        curr_metrics_data = all_metrics_data[i]
        for j, (retain_metrics, forget_metrics) in enumerate(curr_metrics_data): 
        
            forget_string = forget_lists_strings[i]
            method = unlearn_methods[j]
            round_to_2 = lambda x: round(x, 2)

            r_row = {
                'Forgotten Classes': forget_string,
                'Method': method,
                'Accuracy': round_to_2(retain_metrics['accuracy']),
                'Precision': round_to_2(retain_metrics['precision']),
                'Recall': round_to_2(retain_metrics['recall']),
                'F1 Score': round_to_2(retain_metrics['f1'])
            }

            f_row = {
                'Forgotten Classes': forget_string,
                'Method': method,
                'Accuracy': round_to_2(forget_metrics['accuracy']),
                'Precision': round_to_2(forget_metrics['precision']),
                'Recall': round_to_2(forget_metrics['recall']),
                'F1 Score': round_to_2(forget_metrics['f1'])
            }

            # Append rows
            r_performance_df = pd.concat([r_performance_df, pd.DataFrame([r_row])], ignore_index=True)
            f_performance_df = pd.concat([f_performance_df, pd.DataFrame([f_row])], ignore_index=True)

    return r_performance_df, f_performance_df

def create_mia_table(unlearn_methods, forget_lists_strings, all_mia_data):
    # Create the membership inference attack table
    mia_cols = ['Forgotten Class', 'Method', 'ACC', 
                'TP', 'TN', 'FP', 'FN', 
                'TPR', 'TNR', 'PPV', 'NPV',
                'FPR', 'FNR', 'FDR']
    mia_df = pd.DataFrame(columns=mia_cols)

    rows = []
    for i in range(len(forget_lists_strings)):
        curr_mia_data = all_mia_data[i]
        for j, (mean_metrics, std_metrics) in enumerate(curr_mia_data):
            forget_string = forget_lists_strings[i]
            method = unlearn_methods[j]
            row = {
                'Forgotten Class': forget_string, 
                'Method': method, 
                'ACC': f'{mean_metrics["ACC"]:.2f}±{std_metrics["ACC"]:.2f}', 
                'TP': f'{mean_metrics["TP"]:.2f}±{std_metrics["TP"]:.2f}',
                'TN': f'{mean_metrics["TN"]:.2f}±{std_metrics["TN"]:.2f}', 
                'FP': f'{mean_metrics["FP"]:.2f}±{std_metrics["FP"]:.2f}', 
                'FN': f'{mean_metrics["FN"]:.2f}±{std_metrics["FN"]:.2f}', 
                'TPR': f'{mean_metrics["TPR"]:.2f}±{std_metrics["TPR"]:.2f}', 
                'TNR': f'{mean_metrics["TNR"]:.2f}±{std_metrics["TNR"]:.2f}', 
                'PPV': f'{mean_metrics["PPV"]:.2f}±{std_metrics["PPV"]:.2f}', 
                'NPV': f'{mean_metrics["NPV"]:.2f}±{std_metrics["NPV"]:.2f}',
                'FPR': f'{mean_metrics["FPR"]:.2f}±{std_metrics["FPR"]:.2f}', 
                'FNR': f'{mean_metrics["FNR"]:.2f}±{std_metrics["FNR"]:.2f}', 
                'FDR': f'{mean_metrics["FDR"]:.2f}±{std_metrics["FDR"]:.2f}'
            }
            rows.append(row)

    mia_df = pd.DataFrame(rows, columns=mia_cols)
    return mia_df

def run_experiment(experiment_params, unlearn_methods, seed):

    # Initialize variables for experiment loop
    forget_lists = experiment_params['forget_lists']
    forget_lists_strings = experiment_params['forget_lists_strings']

    # Train the full version of the ResNet18 model
    root_path = '/content/Unlearning-MIA-Eval/Final_Structure/data'
    forget_classes = [0]  # Number in here is irrelevant
    loaders = get_loaders(root=root_path, forget_classes=forget_classes, seed=seed)
    full_model_path = f'/content/drive/MyDrive/AIML_Final_Project/checkpoints/resnet_full_{experiment_params['dataset']['name']}.pt'
    if not os.path.isfile(full_model_path):
        train(
            model=experiment_params['full_train']['model'], 
            train_loader=loaders[0], 
            criterion=experiment_params['full_train']['criterion'], 
            optimizer=experiment_params['full_train']['optimizer'], 
            epochs=experiment_params['full_train']['epochs'],
            scheduler=experiment_params['full_train']['scheduler'], 
            save_path=full_model_path
        )
    else:
        print("No need to train ResNet on full train set, as a checkpoint already exists!")

    # Start experiment loop
    all_metrics_data = []
    all_mia_data = []
    for forget_list, forget_string in zip(forget_lists, forget_lists_strings):
        
        # Getting all the loaders
        root_path = '/content/Unlearning-MIA-Eval/Final_Structure/data'
        loaders = get_loaders(root=root_path, forget_classes=forget_list, seed=seed)
        train_forget_loader = loaders[3]
        train_retain_loader = loaders[4]
        test_forget_loader = loaders[7]

        # Perform unlearning methods
        metrics_data = []
        mia_data = []
        for method in unlearn_methods:

            unl_model = None
            unl_args = SimpleNamespace()
            
            if method == 'retrain':

                if not os.path.isfile(unl_args.check_path):
                    retrain_path = f'/content/drive/MyDrive/AIML_Final_Project/checkpoints/retrain_cls_{forget_string}_{experiment_params['dataset']['name']}.pt' 
                    unl_model = experiment_params['retrain']['model']
                    train(
                        model=unl_model, 
                        train_loader=train_retain_loader, 
                        criterion=experiment_params['retrain']['criterion'], 
                        optimizer=experiment_params['retrain']['optimizer'], 
                        epochs=experiment_params['retrain']['epochs'],
                        scheduler=experiment_params['retrain']['scheduler'], 
                        save_path = retrain_path
                    )
                else:
                    print(f'Retrain Unlearning Unnecessary, Checkpoint Exists for Class/Classes {forget_string}')
                    unl_model = load_model(unl_args.check_path)
                    
            elif method == 'SCRUB':
                
                unl_args = experiment_params['scrub']['args']
                unl_args.check_path = f'/content/drive/MyDrive/AIML_Final_Project/checkpoints/scrub_cls_{forget_string}_{experiment_params['dataset']['name']}.pt'
                
                if not os.path.isfile(unl_args.check_path):
                    print(f'Performing SCRUB Unlearning on Class Class/Classes {forget_string}')
                    unl_model, _ = scrub(full_model_path, loaders, unl_args)
                else:
                    print(f'SCRUB Unlearning Unnecessary, Checkpoint Exists for Class Class/Classes {forget_string}')
                    unl_model = load_model(unl_args.check_path)
            
            elif method == 'BadTeach':
                
                unl_args = experiment_params['badt']['args']
                unl_args.check_path = f'/content/drive/MyDrive/AIML_Final_Project/checkpoints/badt_cls_{forget_string}_{experiment_params['dataset']['name']}.pt'

                if not os.path.isfile(unl_args.check_path):
                    print(f'Performing BadTeach Unlearning on Class {forget_string}')
                    unl_model, _ = badt(full_model_path, loaders, unl_args)
                else:
                    print(f'BadTeach Unlearning Unnecessary, Checkpoint Exists for Class {forget_string}')
                    unl_model = load_model(unl_args.check_path)

            else:
                
                unl_args = experiment_params['ssd']['args']
                unl_args.check_path = f'/content/drive/MyDrive/AIML_Final_Project/checkpoints/ssd_cls_{forget_string}_{experiment_params['dataset']['name']}.pt'
                
                if not os.path.isfile(unl_args.check_path):
                    print(f'Performing SSD Unlearning on Class {forget_string}')
                    unl_model, _ = ssd(full_model_path, loaders, unl_args)
                else:
                    print(f'SSD Unlearning Unnecessary, Checkpoint Exists for Class {forget_string}')
                    unl_model = load_model(unl_args.check_path)

            # Get performance evaluation information
            retain_metrics = evaluate_model(unl_model, train_retain_loader, 'cuda')
            forget_metrics = evaluate_model(unl_model, train_forget_loader, 'cuda')
            mia_mean_metrics, mia_std_metrics = membership_inference_attack(
                unl_model,
                test_forget_loader,
                train_forget_loader,
                device='cuda',
                seed=seed
            )
            metrics_data.append((retain_metrics, forget_metrics))
            mia_data.append((mia_mean_metrics, mia_std_metrics))

        # Save the data for this class for use later
        all_metrics_data.append(metrics_data)
        all_mia_data.append(mia_data)

    # Create Post Unlearning Train Tables
    f_performance_df, r_performance_df = create_post_unlearn_tables(unlearn_methods, forget_lists_strings, all_metrics_data)

    # Print nicely
    print(tabulate(r_performance_df, headers='keys', tablefmt='pretty'))
    print()
    print(tabulate(f_performance_df, headers='keys', tablefmt='pretty'))
    print()

    # Create the membership inference attack table
    mia_df = create_mia_table(unlearn_methods, forget_lists_strings, all_metrics_data)

    # Print nicely
    print(tabulate(mia_df, headers='keys', tablefmt='pretty'))

    # Saving data tables locally
    r_performance_df.to_pickle(f'/content/drive/MyDrive/AIML_Final_Project/retain_performance_train_{experiment_params['dataset']['name']}.pkl')
    f_performance_df.to_pickle(f'/content/drive/MyDrive/AIML_Final_Project/forget_performance_train_{experiment_params['dataset']['name']}.pkl')
    mia_df.to_pickle(f'/content/drive/MyDrive/AIML_Final_Project/mia_results_{experiment_params['dataset']['name']}.pkl')

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Name of dataset")
    parser.add_argument("dataset", help="Name of the dataset to use (e.g. cifar10, cifar100)")

    # Get dataset from the command line
    args = parser.parse_args()
    print(f"Using dataset: {args.dataset}")

    # Define arguments independent of dataset
    unlearn_methods = ['retrain', 'SCRUB', 'BadTeach', 'SSD']
    seed = 2

    # Define experiment parameters depending on dataset
    experiment_params = dict()
    if args.dataset == "cifar10":
        print()
    elif args.dataset == "cifar100":
        init_cifar100_params(experiment_params, args.dataset)   
    else:
        print("Inputted unknown dataset!")

    # Run the experiment
    run_experiment(experiment_params, unlearn_methods, seed)

if __name__ == "__main__":
    main()

