# Regular Imports
import sys
import os
import time
import argparse
import copy
from matplotlib import pyplot as plt
from copy import deepcopy

# Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim

# Adding the SCRUB repo to the system path
SCRUB_PATH = os.path.abspath("../../Third_Party_Code/SCRUB")
if SCRUB_PATH not in sys.path:
    sys.path.append(SCRUB_PATH)

# Direct imports from the SCRUB repository
from Third_Party_Code.SCRUB import datasets
from Third_Party_Code.SCRUB.utils import *
from Third_Party_Code.SCRUB import models
from Third_Party_Code.SCRUB.logger import Logger
from Third_Party_Code.SCRUB.thirdparty.repdistiller.helper.util import adjust_learning_rate as sgda_adjust_learning_rate
from Third_Party_Code.SCRUB.thirdparty.repdistiller.helper.loops import train_distill, validate
from Third_Party_Code.SCRUB.thirdparty.repdistiller.distiller_zoo import DistillKL

# My own imports from my code
from Unlearning_Models.evaluate import all_readouts
from Unlearning_Models.train import get_default_args, train

def split_dataset_for_forgetting(dataset, class_to_forget, num_to_forget, args, seed=1):
    
    # Load the full dataset (training, validation, and test sets)
    train_loader_full, _, _ = datasets.get_loaders(
        dataset, batch_size=args.batch_size, seed=seed, root=args.dataroot, augment=False, shuffle=True
    )

    # Load the dataset but mark the samples that need to be forgotten
    marked_loader, _, _ = datasets.get_loaders(
        dataset, class_to_replace=class_to_forget, num_indexes_to_replace=num_to_forget, only_mark=True, 
        batch_size=1, seed=seed, root=args.dataroot, augment=False, shuffle=True
    )

    def replace_loader_dataset(dataset, batch_size, seed, shuffle=True):
        torch.manual_seed(seed)  # Ensure reproducibility
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=shuffle
        )

    # Create a deep copy of the marked dataset to isolate the samples to forget
    forget_dataset = copy.deepcopy(marked_loader.dataset)
    marked = forget_dataset.targets < 0                            # Identify samples marked for forgetting
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

def load_pretrained_models(model, args, target_epoch):

    # Create deep copies of the model to store different versions
    model0 = copy.deepcopy(model)         # Model after forgetting
    model_initial = copy.deepcopy(model)  # Initial model before any training

    # Extract arguments for model configuration
    arch = args.model                                           # Model architecture
    filters = args.filters                                      # Number of filters in CNN (if applicable)
    arch_filters = arch + '_' + str(filters).replace('.', '_')  # Format architecture name
    dataset = args.dataset                                      # Dataset name
    class_to_forget = args.forget_class                         # Class to forget
    init_checkpoint = f"checkpoints/{args.name}_init.pt"        # Path to initial model checkpoint
    num_to_forget = args.num_to_forget                          # Number of samples to forget                                           # Random seed
    unfreeze_start = None                                       # Placeholder for unfreezing layers

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
    m_name = f'checkpoints/{dataset}_{arch_filters}_forget_None{unfreeze_tag}{augment_tag}{learningrate}{batch_size}{lossfn}{wd}{seed_name}{target_epoch}.pt'
    m0_name = f'checkpoints/{dataset}_{arch_filters}_forget_{class_to_forget}{num_tag}{unfreeze_tag}{augment_tag}{learningrate}{batch_size}{lossfn}{wd}{seed_name}{target_epoch}.pt'
     
    # Load pre-trained weights into models
    model.load_state_dict(torch.load(m_name))                   # Model before forgetting
    model0.load_state_dict(torch.load(m0_name))                 # Model after forgetting
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

def train_and_scrub(teacher, student, retain_loader, forget_loader, valid_loader_full, args_f):
    
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
    acc_rs, acc_fs, acc_vs, acc_fvs = [], [], [], []

    # Initializations for validation calculations
    forget_validation_loader = copy.deepcopy(valid_loader_full)
    fgt_cls = list(np.unique(forget_loader.dataset.targets))
    indices = [i in fgt_cls for i in forget_validation_loader.dataset.targets]
    forget_validation_loader.dataset.data = forget_validation_loader.dataset.data[indices]
    forget_validation_loader.dataset.targets = forget_validation_loader.dataset.targets[indices]

    # Training loop
    for epoch in range(1, args_f.sgda_epochs + 1):
        #lr = sgda_adjust_learning_rate(epoch, args_f, optimizer)
        print("==> Scrub unlearning ...")

        # Validate on retained and forgotten data
        acc_r, _, _ = validate(retain_loader, model_s, criterion_cls, args_f, True)
        acc_f, _, _ = validate(forget_loader, model_s, criterion_cls, args_f, True)
        acc_v, _, _ = validate(valid_loader_full, model_s, criterion_cls, args, True)
        acc_fv, _, _ = validate(forget_validation_loader, model_s, criterion_cls, args, True)

        # Storing accuracy results
        acc_rs.append(100 - acc_r.item())
        acc_fs.append(100 - acc_f.item())
        acc_vs.append(100-acc_v.item())
        acc_fvs.append(100-acc_fv.item())

        # Train model
        maximize_loss = 0
        if epoch <= args_f.msteps:
            maximize_loss = train_distill(epoch, forget_loader, module_list, None, criterion_list, optimizer, args_f, "maximize")
        train_acc, train_loss = train_distill(epoch, retain_loader, module_list, None, criterion_list, optimizer, args_f, "minimize")

        # Save the model to a checkpoint file
        curr_name = 'checkpoints/scrub_{}_{}_seed{}_step{}.pt'.format(args.model, args.dataset, args.seed, epoch-1)
        #torch.save(model_s.state_dict(), curr_name)
        print(f"Epoch {epoch}: maximize loss: {maximize_loss:.2f}, minimize loss: {train_loss:.2f}, train_acc: {train_acc}")
    
    # Saving final accuracies after training
    acc_r, _, _ = validate(retain_loader, model_s, criterion_cls, args_f, True)
    acc_f, _, _ = validate(forget_loader, model_s, criterion_cls, args_f, True)
    acc_v, _, _ = validate(valid_loader_full, model_s, criterion_cls, args, True)
    acc_fv, _, _ = validate(forget_validation_loader, model_s, criterion_cls, args, True)
    acc_rs.append(100 - acc_r.item())
    acc_fs.append(100 - acc_f.item())
    acc_vs.append(100-acc_v.item())
    acc_fvs.append(100-acc_fv.item())

    # Plotting results
    indices = list(range(0,len(acc_rs)))
    plt.plot(indices, acc_rs, marker='*', color=u'#1f77b4', alpha=1, label='retain-set')
    plt.plot(indices, acc_fs, marker='o', color=u'#ff7f0e', alpha=1, label='forget-set')
    plt.plot(indices, acc_vs, marker='^', color=u'#2ca02c',alpha=1, label='validation-set')
    plt.plot(indices, acc_fvs, marker='.', color='red',alpha=1, label='forget-validation-set')
    plt.legend(prop={'size': 14})
    plt.tick_params(labelsize=12)
    plt.xlabel('epoch',size=14)
    plt.ylabel('error',size=14)
    plt.grid()
    plt.show()

    '''
    # Plot results
    plt.plot(range(len(acc_rs)), acc_rs, marker='*', alpha=1, label='retain-set')
    plt.plot(range(len(acc_fs)), acc_fs, marker='o', alpha=1, label='forget-set')
    plt.legend(prop={'size': 14})
    plt.tick_params(labelsize=12)
    plt.title('Scrub retain- and forget-set error', size=18)
    plt.xlabel('Epoch', size=14)
    plt.ylabel('Error', size=14)
    plt.show()'
    '''

    # Return model information
    try:
        selected_idx, _ = min(enumerate(acc_fs), key=lambda x: abs(x[1]-acc_fvs[-1]))
    except:
        selected_idx = len(acc_fs) - 1
    print ("the selected index is {}".format(selected_idx))
    selected_model = "checkpoints/scrub_{}_{}_seed{}_step{}.pt".format(args.model, args.dataset, args.seed, int(selected_idx))
    model_s_final = copy.deepcopy(model_s)
    model_s.load_state_dict(torch.load(selected_model))
    
    return model_s, model_s_final

def init_model_for_experiment(args):
    # Process classes to forget (if any)
    if isinstance(args.forget_class, str):
        clss = args.forget_class.split(',')
        args.forget_class = [int(c) for c in clss]

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
    
    # Create and return the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.get_model(args.model, num_classes=args.num_classes).to(device)
    return model

def pre_exp_model_training(args, args_f):
    # Train the model on the entire train set
    train(args)

    # Train the model on the train set excluding the forgotten classes
    train(args_f)

def perform_scrub_unlearning(args_f):
    # Initialize the model for the scrub unlearning
    model = init_model_for_experiment(args_f)

    # Load pre-trained models
    target_epoch_num = 4
    model, model0, model_initial, teacher, student = load_pretrained_models(
        model=model, 
        args=args_f, 
        target_epoch=target_epoch_num
    )

    # Split dataset into retain & forget sets (works on the train set)
    forget_loader, retain_loader = split_dataset_for_forgetting(
        dataset=args.dataset,
        class_to_forget=[1, 2],
        num_to_forget=args_f.num_to_forget,
        args=args_f,
        seed=args_f.seed
    )

    # Loading the full validation set loader
    train_loader_full, valid_loader_full, test_loader_full = datasets.get_loaders(
            args_f.dataset, batch_size=args_f.batch_size, seed=args_f.seed, root=args_f.dataroot, augment=False, shuffle=True
    )

    # Perform SCRUB unlearning
    model_s, model_s_final = train_and_scrub(teacher, student, retain_loader, forget_loader, valid_loader_full, args_f)

    # Define the loaders array
    loaders = [forget_loader, retain_loader, train_loader_full, valid_loader_full, test_loader_full]

    # Call the all readouts function
    readouts = {}
    readouts["SCRUB"] = all_readouts(copy.deepcopy(model_s_final),loaders, args_f, thresh=0.1, name='SCRUB', seed=args_f.seed)

    print(readouts)

#-------------------------------------------------------------------#

# Defining arguments for a model without any forgetting info
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
args.num_classes = 6

# Adding additional arguments necessary for forgetting
args_f = deepcopy(args)
args_f.forget_class = '1,2'
args_f.num_to_forget = None
args_f.split = 'forget'
args_f.device = 'cuda'

# Perform initial model training if necessary
pre_training = True
if pre_training:
    pre_exp_model_training(args, args_f)

# These arguments are necessary for the SCRUB learning process
args_f.optim = 'adam'
args_f.gamma = 1
args_f.alpha = 0.5
args_f.beta = 0
args_f.smoothing = 0.5
args_f.msteps = 3
args_f.clip = 0.2
args_f.sstart = 10
args_f.kd_T = 2
args_f.distill = 'kd'
args_f.sgda_epochs = 10
args_f.sgda_learning_rate = 0.0005
args_f.lr_decay_epochs = [5, 8, 9]
args_f.lr_decay_rate = 0.1
args_f.sgda_weight_decay = 0.1
args_f.sgda_momentum = 0.9

# Perform the SCRUB unlearning and save the results
perform_scrub_unlearning(args_f)