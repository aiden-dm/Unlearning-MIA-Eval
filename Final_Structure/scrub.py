# Imports
import copy
import torch.nn as nn
import torch.optim as optim
import sys
import matplotlib.pyplot as plt
from types import SimpleNamespace
import torch

# Adding necessary paths to the system path
sys.path.append('/content/Unlearning-MIA-Eval')

from Final_Structure.evaluate import train_validation
from Final_Structure.training import load_model

# Imports from the SCRUB repository
from Third_Party_Code.SCRUB.thirdparty.repdistiller.distiller_zoo import DistillKL
from Third_Party_Code.SCRUB.thirdparty.repdistiller.helper.loops import train_distill, validate

def scrub(full_model_path, loaders, args):

    # Load full model 
    model = load_model(full_model_path)

    # Extracting the loaders that we want
    train_forget_loader = loaders[3]
    train_retain_loader = loaders[4]
    valid_forget_loader = loaders[5]
    valid_retain_loader = loaders[6]

    # Defining hyperparameters
    kd_T = args.kd_T
    learning_rate = args.learning_rate
    epochs = args.epochs
    msteps = args.msteps

    # Define teacher and student models
    model_t = copy.deepcopy(model)
    model_s = copy.deepcopy(model)

    # Module lists
    module_list = nn.ModuleList([model_s])
    trainable_list = nn.ModuleList([model_s])

    # Define loss functions
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(kd_T)
    criterion_kd = DistillKL(kd_T)          # This doesn't do anything placeholder
    criterion_list = nn.ModuleList([criterion_cls, criterion_div, criterion_kd])

    # Define the optimizer
    optimizer = optim.Adam(trainable_list.parameters(),
                               lr=learning_rate,
                           )
    
    # Add teacher model to the module list
    module_list.append(model_t)

    # Track accuracy
    tf_accs = []
    tr_accs = []
    vf_accs = []
    vr_accs = []
    losses = []
    epoch_list = []

    # Define validate args
    v_opt = SimpleNamespace()
    v_opt.print_freq = 0

    # Define train distill args
    t_opt = SimpleNamespace()
    t_opt.distill = 'kd'
    t_opt.gamma = args.t_opt_gamma      # Classification weight
    t_opt.alpha = args.t_opt_alpha      # KL divergence weight
    t_opt.beta = 0
    t_opt.print_freq = 0

    # Training loop
    for epoch in range(1, epochs + 1):
        #lr = sgda_adjust_learning_rate(epoch, args_f, optimizer)

        # Train model
        maximize_loss = 0
        if epoch <= msteps:
            maximize_loss = train_distill(epoch, train_forget_loader, module_list, None, criterion_list, optimizer, t_opt, "maximize", quiet=True)
        train_acc, train_loss = train_distill(epoch, train_retain_loader, module_list, None, criterion_list, optimizer, t_opt, "minimize", quiet=True)

        losses.append(train_loss)
        epoch_list.append(epoch)
        acc_dict = train_validation(model_s, 
                                    train_retain_loader, 
                                    train_forget_loader, 
                                    valid_retain_loader, 
                                    valid_forget_loader)
        tr_accs.append(acc_dict['tr_acc'])
        tf_accs.append(acc_dict['tf_acc'])
        vr_accs.append(acc_dict['vr_acc'])
        vf_accs.append(acc_dict['vf_acc'])

        print(f"Epoch {epoch}: maximize loss: {maximize_loss:.2f}, minimize loss: {train_loss:.2f}, train_acc: {train_acc}")

        # Print epoch progress
        if args.print_accuracies:
            print(f"   tr_acc: {acc_dict['tr_acc']}")
            print(f"   tf_acc: {acc_dict['tf_acc']}")
            print(f"   vr_acc: {acc_dict['vr_acc']}")
            print(f"   vf_acc: {acc_dict['vf_acc']}")

    # Save a copy of the student model to use in evaluation
    torch.save(model_s.state_dict(), args.check_path)

    history = {
        'losses': losses,
        'epoch_list': epoch_list,
        'tr_accs': tr_accs,
        'tf_accs': tf_accs,
        'vr_accs': vr_accs,
        'vf_accs': vf_accs
    }

    return model_s, history
