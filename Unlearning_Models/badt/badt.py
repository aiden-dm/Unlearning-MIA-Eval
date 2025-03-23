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
from Third_Party_Code.SCRUB.thirdparty.repdistiller.helper.loops import validate, train_bad_teacher
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
    bad_teacher = models.get_model(args.model, num_classes=args.num_classes).to(args.device)

    # Move models to GPU if available
    model.cuda()
    model0.cuda()

    # Store initial parameter copies for potential weight updates
    for p in model.parameters():
        p.data0 = p.data.clone()
    for p in model0.parameters():
        p.data0 = p.data.clone()

    return teacher, bad_teacher, student

def badt(gteacher, bteacher, student, loaders, args):
    
    model_gt = copy.deepcopy(gteacher)
    model_bt = copy.deepcopy(bteacher)
    model_s = copy.deepcopy(student)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.bt_kd_T)
    criterion_kd = DistillKL(args.bt_kd_T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    if args.bt_optim == "sgd":
        optimizer = optim.SGD(trainable_list.parameters(),
                              lr=args.bt_learning_rate,
                              momentum=args.bt_momentum,
                              weight_decay=args.bt_weight_decay)
    elif args.bt_optim == "adam": 
        optimizer = optim.Adam(trainable_list.parameters(),
                              lr=args.bt_learning_rate,
                              weight_decay=args.bt_weight_decay)
    elif args.bt_optim == "rmsp":
        optimizer = optim.RMSprop(trainable_list.parameters(),
                              lr=args.bt_learning_rate,
                              momentum=args.bt_momentum,
                              weight_decay=args.bt_weight_decay)

    module_list.append(model_gt)
    module_list.append(model_bt)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    acc_rs = []
    acc_fs = []
    acc_vs = []

    retain_loader, forget_loader, valid_loader_full = loaders
    
    print("==> Bad Teacher Unlearning ...")
    for epoch in range(1, args.bt_epochs + 1):

        acc_r, _, _ = validate(retain_loader, model_s, criterion_cls, args, True)
        acc_f, _, _ = validate(forget_loader, model_s, criterion_cls, args, True)
        acc_v, _, _ = validate(valid_loader_full, model_s, criterion_cls, args, True)
        acc_rs.append(100-acc_r.item())
        acc_fs.append(100-acc_f.item())
        acc_vs.append(100-acc_v.item())

        #lr = sgda_adjust_learning_rate(epoch, args, optimizer)
        train_acc, loss = train_bad_teacher(epoch, retain_loader, forget_loader, module_list, criterion_list, optimizer, args)

        print ("loss: {:.2f}\t train_acc: {}".format(loss, train_acc))
        
    acc_r, _, _ = validate(retain_loader, model_s, criterion_cls, args, True)
    acc_f, _, _ = validate(forget_loader, model_s, criterion_cls, args, True)
    acc_v, _, _ = validate(valid_loader_full, model_s, criterion_cls, args, True)
    acc_rs.append(100-acc_r.item())
    acc_fs.append(100-acc_f.item())
    acc_vs.append(100-acc_v.item())

    indices = list(range(0,len(acc_rs)))
    plt.plot(indices, acc_rs, marker='*', color=u'#1f77b4', alpha=1, label='retain-set')
    plt.plot(indices, acc_fs, marker='o', color=u'#ff7f0e', alpha=1, label='forget-set')
    plt.plot(indices, acc_vs, marker='^', color=u'#2ca02c',alpha=1, label='validation-set')
    plt.legend(prop={'size': 14})
    plt.tick_params(labelsize=12)
    #plt.title('sgda retain- and forget- set error',size=18)
    plt.xlabel('epoch',size=14)
    plt.ylabel('error',size=14)
    plt.grid()
    #plt.ylim(0,0.4)
    #plt.xlim(-5,2)
    #plt.savefig('Plots/small_cifar5_allcnn_forget0_num5_epochs25_'+title+'.png')
    plt.show()
    
    return model_s

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

def perform_badt_unlearning(args_f):
    # Initialize the model for the scrub unlearning
    model = init_model_for_experiment(args_f)

    # Load pre-trained models
    target_epoch_num = 4
    teacher, bad_teacher, student = load_pretrained_models(
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
    loaders_bt = [retain_loader, forget_loader, valid_loader_full]
    model_bt = badt(
        gteacher = teacher, 
        bteacher = bad_teacher, 
        student = student, 
        loaders = loaders_bt, 
        args = args_f
    )

    # Define the loaders array
    loaders = [forget_loader, retain_loader, train_loader_full, valid_loader_full, test_loader_full]

    # Call the all readouts function
    readouts = {}
    readouts["BadT"] = all_readouts(copy.deepcopy(model_bt), loaders, args_f, thresh=0.1,name='BadT', seed=args_f.seed)

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
pre_training = False
if pre_training:
    pre_exp_model_training(args, args_f)

# These arguments are necessary for the BadTeach unlearning process
args_f.bt_optim = 'adam'
args_f.bt_alpha = 1
args_f.bt_beta = 1
args_f.bt_kd_T = 4
args_f.bt_distill = 'kd'
args_f.bt_epochs = 1
args_f.bt_learning_rate = 0.001
args_f.bt_lr_decay_epochs = [10,10,10]
args_f.bt_lr_decay_rate = 0.1
args_f.bt_weight_decay = 5e-4
args_f.bt_momentum = 0.9

# Perform the BadTeach unlearning and save the results
perform_badt_unlearning(args_f)