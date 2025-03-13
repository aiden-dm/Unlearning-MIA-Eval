# Regular Imports
import sys
import os
import time
import argparse
import copy
from collections import Counter
from matplotlib import pyplot as plt
from copy import deepcopy

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

def run_epoch(args, model, model_init, train_loader, logger, criterion=torch.nn.CrossEntropyLoss(), optimizer=None, scheduler=None, epoch=0, weight_decay=0.0, mode='train', quiet=False):
    
    if mode == 'train':
        model.train()
    elif mode == 'test':
        model.eval()
    elif mode == 'dry_run':
        model.eval()
        set_batchnorm_mode(model, train=True)
    else:
        raise ValueError("Invalid mode.")
    
    if args.disable_bn:
        set_batchnorm_mode(model, train=False)
    
    mult=0.5 if args.lossfn=='mse' else 1
    metrics = AverageMeter()

    with torch.set_grad_enabled(mode != 'test'):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            
            if args.lossfn=='mse':
                target=(2*target-1)
                target = target.type(torch.cuda.FloatTensor).unsqueeze(1)
            elif args.lossfn=='ce':
                target=target.long()
                
            if 'mnist' in args.dataset:
                data=data.view(data.shape[0],-1)
                
            output = model(data)
            loss = mult*criterion(output, target) + l2_penalty(model,model_init,weight_decay)
            
            if args.l1:
                l1_loss = sum([p.norm(1) for p in model.parameters()])
                loss += args.weight_decay * l1_loss

            if ~quiet:
                metrics.update(n=data.size(0), loss=loss.item(), error=get_error(output, target))
            
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    log_metrics(mode, metrics, epoch)
    logger.append('train' if mode=='train' else 'test', epoch=epoch, loss=metrics.avg['loss'], error=metrics.avg['error'], 
                  lr=optimizer.param_groups[0]['lr'])
    print('Learning Rate : {}'.format(optimizer.param_groups[0]['lr']))
    return metrics

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
        run_epoch(args, model, model_init, train_loader, logger, criterion, optimizer, scheduler, epoch, weight_decay, mode='train', quiet=args.quiet)
        t2 = time.time()
        train_time += np.round(t2 - t1, 2)
        
        # Periodic validation and checkpointing
        if epoch % 500000 == 0:
            if not args.disable_bn:
                run_epoch(args, model, model_init, train_loader, logger, criterion, optimizer, scheduler, epoch, weight_decay, mode='dry_run')
            run_epoch(args, model, model_init, test_loader, logger, criterion, optimizer, scheduler, epoch, weight_decay, mode='test')
        if epoch % 1 == 0:
            torch.save(model.state_dict(), f"checkpoints/{args.name}_{epoch}.pt")
        
        print(f'Epoch Time: {np.round(time.time() - t1, 2)} sec')

    print(f'Pure training time: {train_time} sec')

    # Return statements
    return model, train_loader, valid_loader

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

    # Optimization parameters
    parser.add_argument('--optim', type=str, default='adam', help='Optimizer type')
    parser.add_argument('--gamma', type=float, default=1, help='Gamma value for learning rate scheduling')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha value for optimization')
    parser.add_argument('--beta', type=float, default=0, help='Beta value for optimization')
    parser.add_argument('--smoothing', type=float, default=0.5, help='Label smoothing factor')
    parser.add_argument('--msteps', type=int, default=3, help='Number of m-steps for optimization')
    parser.add_argument('--clip', type=float, default=0.2, help='Gradient clipping threshold')
    parser.add_argument('--sstart', type=int, default=10, help='Start epoch for a specific strategy')
    parser.add_argument('--kd_T', type=int, default=2, help='Temperature for knowledge distillation')
    parser.add_argument('--distill', type=str, default='kd', help='Distillation method')

    args = parser.parse_args([])  # Provide an empty list to get default values
    return args

def split_dataset_for_forgetting(dataset, class_to_forget, num_to_forget, args, seed=1):
    
    # Load the full dataset (training, validation, and test sets)
    train_loader_full, valid_loader_full, test_loader_full = datasets.get_loaders(
        dataset, batch_size=args.batch_size, seed=seed, root=args.dataroot, augment=False, shuffle=True
    )

    # Load the dataset but mark the samples that need to be forgotten
    marked_loader, _, _ = datasets.get_loaders(
        dataset, class_to_replace=class_to_forget, num_indexes_to_replace=num_to_forget, only_mark=True, 
        batch_size=1, seed=seed, root=args.dataroot, augment=False, shuffle=True
    )

    def replace_loader_dataset(dataset, batch_size, seed, shuffle=True):
        """
        Creates a new DataLoader for a given dataset.
        """
        torch.manual_seed(seed)  # Ensure reproducibility
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=shuffle
        )

    # Create a deep copy of the marked dataset to isolate the samples to forget
    forget_dataset = copy.deepcopy(marked_loader.dataset)
    marked = forget_dataset.targets < 0  # Identify samples marked for forgetting
    forget_dataset.data = forget_dataset.data[marked]
    forget_dataset.targets = - forget_dataset.targets[marked] - 1  # Restore original labels

    # Create DataLoader for the forget dataset
    forget_loader = replace_loader_dataset(forget_dataset, batch_size=args.forget_bs, seed=seed, shuffle=True)

    # Create a deep copy of the marked dataset to retain the remaining samples
    retain_dataset = copy.deepcopy(marked_loader.dataset)
    marked = retain_dataset.targets >= 0  # Identify samples to retain
    retain_dataset.data = retain_dataset.data[marked]
    retain_dataset.targets = retain_dataset.targets[marked]

    # Create DataLoader for the retain dataset
    retain_loader = replace_loader_dataset(retain_dataset, batch_size=args.retain_bs, seed=seed, shuffle=True)

    # Ensure the split was performed correctly
    assert len(forget_dataset) + len(retain_dataset) == len(train_loader_full.dataset)

    return forget_loader, retain_loader

def load_pretrained_models(model, args, train_loader, training_epochs):

    # Create deep copies of the model to store different versions
    model0 = copy.deepcopy(model)  # Model after forgetting
    model_initial = copy.deepcopy(model)  # Initial model before any training

    # Extract arguments for model configuration
    arch = args.model  # Model architecture
    filters = args.filters  # Number of filters in CNN (if applicable)
    arch_filters = arch + '_' + str(filters).replace('.', '_')  # Format architecture name
    dataset = args.dataset  # Dataset name
    class_to_forget = args.forget_class  # Class to forget
    init_checkpoint = f"checkpoints/{args.name}_init.pt"  # Path to initial model checkpoint
    num_to_forget = args.num_to_forget  # Number of samples to forget
    num_total = len(train_loader.dataset)  # Total number of samples
    num_to_retain = num_total - 300  # Retaining remaining samples (may be modified)
    seed = args.seed  # Random seed
    unfreeze_start = None  # Placeholder for unfreezing layers

    # Formatting hyperparameter tags for checkpoint filenames
    learningrate = f"lr_{str(args.lr).replace('.', '_')}"
    batch_size = f"_bs_{str(args.batch_size)}"
    lossfn = f"_ls_{args.lossfn}"
    wd = f"_wd_{str(args.weight_decay).replace('.', '_')}"
    seed_name = f"_seed_{args.seed}_"
    num_tag = '' if num_to_forget is None else f'_num_{num_to_forget}'
    unfreeze_tag = '_' if unfreeze_start is None else f'_unfreeze_from_{unfreeze_start}_'
    augment_tag = '' if not False else f'augment_'

    # Define checkpoint filenames
    m_name = f'checkpoints/{dataset}_{arch_filters}_forget_None{unfreeze_tag}{augment_tag}{learningrate}{batch_size}{lossfn}{wd}{seed_name}{training_epochs}.pt'
    m0_name = f'checkpoints/{dataset}_{arch_filters}_forget_{class_to_forget}{num_tag}{unfreeze_tag}{augment_tag}{learningrate}{batch_size}{lossfn}{wd}{seed_name}{training_epochs}.pt'

    # Load pre-trained weights into models
    model.load_state_dict(torch.load(m_name))  # Model before forgetting
    model0.load_state_dict(torch.load(m0_name))  # Model after forgetting
    model_initial.load_state_dict(torch.load(init_checkpoint))  # Initial model before training

    # Create teacher and student models for knowledge distillation
    teacher = copy.deepcopy(model)
    student = copy.deepcopy(model)

    # Move models to GPU if available
    model.cuda()
    model0.cuda()

    # Store initial parameter copies for potential weight updates
    for p in model.parameters():
        p.data0 = p.data.clone()
    for p in model0.parameters():
        p.data0 = p.data.clone()

    return model, model0, model_initial, teacher, student

def train_and_scrub(teacher, student, retain_loader, forget_loader, args_f):
    
    # Deep copy models
    model_t = copy.deepcopy(teacher)
    model_s = copy.deepcopy(student)

    # Module lists
    module_list = nn.ModuleList([model_s])
    trainable_list = nn.ModuleList([model_s])

    # Define loss functions
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args_f.kd_T)
    criterion_kd = DistillKL(args_f.kd_T)

    criterion_list = nn.ModuleList([criterion_cls, criterion_div, criterion_kd])

    # Define optimizer
    if args_f.optim == "sgd":
        optimizer = optim.SGD(trainable_list.parameters(),
                              lr=args_f.sgda_learning_rate,
                              momentum=args_f.sgda_momentum,
                              weight_decay=args_f.sgda_weight_decay)
    elif args_f.optim == "adam":
        optimizer = optim.Adam(trainable_list.parameters(),
                               lr=args_f.sgda_learning_rate,
                               weight_decay=args_f.sgda_weight_decay)
    elif args_f.optim == "rmsp":
        optimizer = optim.RMSprop(trainable_list.parameters(),
                                  lr=args_f.sgda_learning_rate,
                                  momentum=args_f.sgda_momentum,
                                  weight_decay=args_f.sgda_weight_decay)

    # Add teacher model to the module list
    module_list.append(model_t)

    # Track accuracy
    acc_rs, acc_fs = [], []

    # Training loop
    for epoch in range(1, args_f.sgda_epochs + 1):
        lr = sgda_adjust_learning_rate(epoch, args_f, optimizer)
        print("==> Scrub unlearning ...")

        # Validate on retained and forgotten data
        acc_r, _, _ = validate(retain_loader, model_s, criterion_cls, args_f, True)
        acc_f, _, _ = validate(forget_loader, model_s, criterion_cls, args_f, True)

        acc_rs.append(100 - acc_r.item())
        acc_fs.append(100 - acc_f.item())

        # Train model
        maximize_loss = 0
        if epoch <= args_f.msteps:
            maximize_loss = train_distill(epoch, forget_loader, module_list, None, criterion_list, optimizer, args_f, "maximize")

        train_acc, train_loss = train_distill(epoch, retain_loader, module_list, None, criterion_list, optimizer, args_f, "minimize")

        print(f"Epoch {epoch}: maximize loss: {maximize_loss:.2f}, minimize loss: {train_loss:.2f}, train_acc: {train_acc}")

    # Plot results
    plt.plot(range(len(acc_rs)), acc_rs, marker='*', alpha=1, label='retain-set')
    plt.plot(range(len(acc_fs)), acc_fs, marker='o', alpha=1, label='forget-set')
    plt.legend(prop={'size': 14})
    plt.tick_params(labelsize=12)
    plt.title('Scrub retain- and forget-set error', size=18)
    plt.xlabel('Epoch', size=14)
    plt.ylabel('Error', size=14)
    plt.show()



#-------------------------------------------------------------------#

# PRE-TRAINING

# Define common arguments
args = get_default_args()
args.dataset = 'small_cifar6'
args.model = 'allcnn'
args.root = 'data/'
args.filters = 1.0
args.lr = 0.001
args.disable_bn = True
args.weight_decay = 0.1
args.batch_size = 128
args.epochs = 5
args.seed = 3
args.retain_bs = 32
args.forget_bs = 64

# Create a copy for forgetting phase and modify only necessary fields
args_f = deepcopy(args)
args_f.forget_class = '1,2'
args_f.num_to_forget = None
args_f.split = 'forget'

# Train model on the full dataset
model, train_loader, test_loader = train(args)

# Train model with forget classes omitted
model_f, train_loader_f, test_loader_f = train(args_f)

# Split dataset into retain & forget sets
forget_loader, retain_loader = split_dataset_for_forgetting(
    dataset=args.dataset,
    class_to_forget=[1, 2],
    num_to_forget=args_f.num_to_forget,
    args=args,
    seed=args.seed
)

# Load pre-trained models
training_epochs = 4
model, model0, model_initial, teacher, student = load_pretrained_models(
    model=model, 
    args=args_f, 
    train_loader=train_loader, 
    training_epochs=training_epochs
)

# APPLYING SCRUB UNLEARNING

# Adding additional arguments
args.optim = 'adam'
args.gamma = 1
args.alpha = 0.5
args.beta = 0
args.smoothing = 0.5
args.msteps = 3
args.clip = 0.2
args.sstart = 10
args.kd_T = 2
args.distill = 'kd'

# SGDA parameters
args.sgda_epochs = 10
args.sgda_learning_rate = 0.0005
args.lr_decay_epochs = [5, 8, 9]
args.lr_decay_rate = 0.1
args.sgda_weight_decay = 0.1
args.sgda_momentum = 0.9

# Perform SCRUB unlearning
train_and_scrub(teacher, student, retain_loader, forget_loader, args)

# EVALUATING UNLEARNING

