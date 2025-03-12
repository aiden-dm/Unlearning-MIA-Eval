# Regular Imports
import sys
import os
import time
import argparse
import copy
from collections import Counter

# Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim

# Adding SCRUB repository to system path
sys.path.append(os.path.abspath("Third_Party_Code/SCRUB"))

# Direct imports from the SCRUB repository
from Third_Party_Code.SCRUB import datasets
from Third_Party_Code.SCRUB.utils import *
from Third_Party_Code.SCRUB import models
from Third_Party_Code.SCRUB.logger import Logger
from Third_Party_Code.SCRUB.thirdparty.repdistiller.helper.util import adjust_learning_rate as sgda_adjust_learning_rate
from Third_Party_Code.SCRUB.thirdparty.repdistiller.helper.loops import train_distill, validate
from Third_Party_Code.SCRUB.thirdparty.repdistiller.distiller_zoo import DistillKL

def adjust_learning_rate(args, optimizer, epoch):
    if args.step_size is not None:lr = args.lr * 0.1 ** (epoch//args.step_size)
    else:lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def l2_penalty(model,model_init,weight_decay):
    l2_loss = 0
    for (k,p),(k_init,p_init) in zip(model.named_parameters(),model_init.named_parameters()):
        if p.requires_grad:
            l2_loss +=  (p-p_init).pow(2).sum()
    l2_loss *= (weight_decay/2.)
    return l2_loss

def run_epoch(args, model, model_init, train_loader, criterion=torch.nn.CrossEntropyLoss(), optimizer=None, scheduler=None, epoch=0, weight_decay=0.0, mode='train', quiet=False):
    
    # Set model mode: training, evaluation, or dry-run
    if mode == 'train':
        model.train()
    elif mode in ['test', 'dry_run']:
        model.eval()
        if mode == 'dry_run':
            set_batchnorm_mode(model, train=True)  # Enable batch norm updates
    else:
        raise ValueError("Invalid mode. Choose from 'train', 'test', or 'dry_run'.")

    # Disable batch normalization if specified
    if args.disable_bn:
        set_batchnorm_mode(model, train=False)

    # Adjust loss scaling factor for Mean Squared Error (MSE) loss
    loss_multiplier = 0.5 if args.lossfn == 'mse' else 1

    # Initialize metrics tracker
    metrics = AverageMeter()

    # Enable gradient computation only during training
    with torch.set_grad_enabled(mode != 'test'):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data and target to the specified device (GPU or CPU)
            data, target = data.to(args.device), target.to(args.device)

            # Adjust target format for MSE loss
            if args.lossfn == 'mse':
                target = (2 * target - 1).type(torch.cuda.FloatTensor).unsqueeze(1)

            # Flatten data for MNIST dataset
            if 'mnist' in args.dataset:
                data = data.view(data.shape[0], -1)

            # Forward pass
            output = model(data)

            # Compute loss with L2 penalty
            loss = loss_multiplier * criterion(output, target) + l2_penalty(model, model_init, weight_decay)

            # Apply L1 regularization if enabled
            if args.l1:
                l1_loss = sum(p.norm(1) for p in model.parameters())
                loss += weight_decay * l1_loss

            # Update metrics if logging is enabled
            if not quiet:
                metrics.update(n=data.size(0), loss=loss.item(), error=get_error(output, target))

            # Backpropagation and optimization step (only in training mode)
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # Log metrics and print learning rate
    log_metrics(mode, metrics, epoch)
    print(f'Learning Rate: {optimizer.param_groups[0]["lr"]}')

    return metrics

# Function to perform initial training
def train(args):                 # based on SCRUB main.py

    # Process learning rate decay epochs
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = [int(it) for it in iterations]

    # Process classes to forget (if any)
    if args.forget_class is not None:
        clss = args.forget_class.split(',')
        args.forget_class = [int(c) for c in clss]

    # Set manual seed for reproducibility
    manual_seed(args.seed)

    # Set step size if not provided
    if args.step_size is None:
        args.step_size = args.epochs + 1

    # Generate checkpoint name if not provided
    if args.name is None:
        args.name = f"{args.dataset}_{args.model}_{str(args.filters).replace('.','_')}"
        if args.split == 'train':
            args.name += "_forget_None"
        else:
            args.name += f"_forget_{args.forget_class}"
            if args.num_to_forget is not None:
                args.name += f"_num_{args.num_to_forget}"
        if args.unfreeze_start is not None:
            args.name += f"_unfreeze_from_{args.unfreeze_start.replace('.','_')}"
        if args.augment:
            args.name += "_augment"
        args.name += f"_lr_{str(args.lr).replace('.','_')}"
        args.name += f"_bs_{args.batch_size}"
        args.name += f"_ls_{args.lossfn}"
        args.name += f"_wd_{str(args.weight_decay).replace('.','_')}"
        args.name += f"_seed_{args.seed}"

    print(f'Checkpoint name: {args.name}')

    # Create necessary directories
    mkdir('logs')
    os.makedirs('checkpoints', exist_ok=True)

    # Initialize logger
    logger = Logger(index=args.name + '_training')
    logger['args'] = args
    logger['checkpoint'] = os.path.join('models/', logger.index + '.pth')
    logger['checkpoint_step'] = os.path.join('models/', logger.index + '_{}.pth')
    print(f"[Logging in {logger.index}]")

    # Set device (CUDA if available, otherwise CPU)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")

    # Load dataset and data loaders
    train_loader, valid_loader, test_loader = datasets.get_loaders(
        args.dataset, class_to_replace=args.forget_class,
        num_indexes_to_replace=args.num_to_forget, confuse_mode=args.confuse_mode,
        batch_size=args.batch_size, split=args.split, seed=args.seed,
        root=args.dataroot, augment=args.augment
    )

    # Determine number of classes
    num_classes = max(train_loader.dataset.targets) + 1 if args.num_classes is None else args.num_classes
    args.num_classes = num_classes
    print(f"Number of Classes: {num_classes}")

    # Initialize model
    model = models.get_model(args.model, num_classes=num_classes, filters_percentage=args.filters).to(args.device)

    # Define classifier layer names for different architectures
    classifier_name = 'classifier.' if args.model == 'allcnn' else 'linear.' if 'resnet' in args.model else ''

    # Load pre-trained model checkpoint if resuming
    if args.resume is not None:
        state = torch.load(args.resume)
        state = {k: v for k, v in state.items() if not k.startswith(classifier_name)}
        incompatible_keys = model.load_state_dict(state, strict=False)
        assert all(k.startswith(classifier_name) for k in incompatible_keys.missing_keys)

    # Save initial model state for reference
    model_init = copy.deepcopy(model)
    torch.save(model.state_dict(), f"checkpoints/{args.name}_init.pt")

    # Select model parameters for training (handling partial unfreezing)
    parameters = model.parameters()
    if args.unfreeze_start is not None:
        parameters = []
        layer_index = 1e8
        for i, (n, p) in enumerate(model.named_parameters()):
            if args.unfreeze_start in n or i > layer_index:
                layer_index = i
                parameters.append(p)

    # Define optimizer and loss function
    weight_decay = args.weight_decay if not args.l1 else 0.0
    optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=0.0)
    criterion = torch.nn.CrossEntropyLoss().to(args.device) if args.lossfn == 'ce' else torch.nn.MSELoss().to(args.device)

    # Define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=0.1, last_epoch=-1)

    # Training loop
    train_time = 0
    for epoch in range(args.epochs):
        adjust_learning_rate(args, optimizer, epoch)
        t1 = time.time()
        run_epoch(args, model, model_init, train_loader, criterion, optimizer, scheduler, epoch, weight_decay, mode='train', quiet=args.quiet)
        t2 = time.time()
        train_time += np.round(t2 - t1, 2)
        
        # Periodic validation and checkpointing
        if epoch % 500000 == 0:
            if not args.disable_bn:
                run_epoch(args, model, model_init, train_loader, criterion, optimizer, scheduler, epoch, weight_decay, mode='dry_run')
            run_epoch(args, model, model_init, test_loader, criterion, optimizer, scheduler, epoch, weight_decay, mode='test')
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"checkpoints/{args.name}_{epoch}.pt")
        
        print(f'Epoch Time: {np.round(time.time() - t1, 2)} sec')

    print(f'Pure training time: {train_time} sec')

    # Return statements
    return model, train_loader

def get_default_args():
    parser = argparse.ArgumentParser()

    # Data and model configuration
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset name (default: mnist)')
    parser.add_argument('--dataroot', type=str, default='data/', help='Root directory for dataset')
    parser.add_argument('--model', type=str, default='mlp', help='Model architecture (default: mlp)')
    parser.add_argument('--num-classes', type=int, default=None, help='Number of output classes')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=31, help='Number of training epochs (default: 31)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='Weight decay (default: 0.0005)')
    parser.add_argument('--step-size', type=int, default=None, help='Step size for learning rate scheduler')
    parser.add_argument('--lossfn', type=str, default='ce', choices=['ce', 'mse'], help='Loss function: Cross Entropy (ce) or Mean Squared Error (mse)')
    
    # Learning rate schedule
    parser.add_argument('--lr_decay_epochs', type=str, default='30,30,30', help='Comma-separated list of epochs for LR decay')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Learning rate decay rate')
    parser.add_argument('--sgda-learning-rate', type=float, default=0.01, help='SGDA learning rate')

    # Regularization
    parser.add_argument('--filters', type=float, default=1.0, help='Percentage of filters to use in the model')
    parser.add_argument('--l1', action='store_true', default=False, help='Use L1 regularization instead of L2')

    # Forgetting parameters
    parser.add_argument('--forget-class', type=str, default=None, help='Class to forget')
    parser.add_argument('--num-to-forget', type=int, default=None, help='Number of samples to forget')
    parser.add_argument('--confuse-mode', action='store_true', default=False, help='Enable interclass confusion test')

    # Model modification options
    parser.add_argument('--unfreeze-start', type=str, default=None, help='Unfreeze layers starting from this point')

    # CUDA and performance settings
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA training')
    parser.add_argument('--disable-bn', action='store_true', default=False, help='Disable batch normalization updates')

    # Experiment tracking
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint file to resume training from')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--split', type=str, choices=['train', 'forget'], default='train', help='Dataset split (train/forget)')

    # Augmentation and debugging
    parser.add_argument('--augment', action='store_true', default=False, help='Enable data augmentation')
    parser.add_argument('--quiet', action='store_true', default=False, help='Suppress output')
    parser.add_argument('--print_freq', type=int, default=500, help='Print frequency during training')

    args = parser.parse_args([])  # Provide an empty list to get default values
    return args

#-------------------------------------------------------------------#

# Define namespace for arguments
args = get_default_args()

# Initialize parameters
args.dataset = 'small_cifar6'
args.model = 'allcnn'
args.root = 'data/'
args.filters = 1.0
args.lr = 0.001
args.resume = 'checkpoints/small_cifar6_0.pt'
args.diable_bn = True
args.weight_decay = 0.1
args.batch_size = 128
args.epochs = 31
args.seed = 3

# Running the initial training function
model, train_loader = train(args)

'''
%run main.py 
    --dataset small_lacuna6 
    --model allcnn 
    --dataroot=data/lacuna10/ 
    --filters 1.0 
    --lr 0.001 \
    --resume checkpoints/lacuna100_allcnn_1_0_forget_None_lr_0_1_bs_128_ls_ce_wd_0_0005_seed_1_30.pt 
    --disable-bn \
    --weight-decay 0.1 
    --batch-size 128 
    --epochs 31 
    --seed 3
'''
