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

# Imports from the SCRUB repository
from Third_Party_Code.SCRUB.thirdparty.repdistiller.distiller_zoo import DistillKL
from Third_Party_Code.SCRUB.thirdparty.repdistiller.helper.loops import train_distill, validate

def scrub(model, loaders, args):

    # Extracting the loaders that we want
    train_forget_loader = loaders[3]
    train_retain_loader = loaders[4]
    valid_forget_loader = loaders[5]
    valid_retain_loader = loaders[6]

    # Defining hyperparameters
    kd_T = args.kd_T
    sgda_learning_rate = args.sgda_learning_rate
    sgda_weight_decay = args.sgda_weight_decay
    sgda_epochs = args.sgda_epochs
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
    criterion_kd = DistillKL(kd_T)
    criterion_list = nn.ModuleList([criterion_cls, criterion_div, criterion_kd])

    # Define the optimizer
    optimizer = optim.Adam(trainable_list.parameters(),
                               lr=sgda_learning_rate,
                               weight_decay=sgda_weight_decay)
    
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
    v_opt.print_freq = args.v_opt_print_freq

    # Define train distill args
    t_opt = SimpleNamespace()
    t_opt.distill = args.t_opt_distill
    t_opt.alpha = args.t_opt_alpha
    t_opt.beta = args.t_opt_beta
    t_opt.gamma = args.t_opt_gamma
    t_opt.print_freq = args.t_opt_print_freq

    # Training loop
    print("PERFROMING SCRUB UNLEARNING...")
    for epoch in range(1, sgda_epochs + 1):
        #lr = sgda_adjust_learning_rate(epoch, args_f, optimizer)

        # Train model
        maximize_loss = 0
        if epoch <= msteps:
            maximize_loss = train_distill(epoch, train_forget_loader, module_list, None, criterion_list, optimizer, t_opt, "maximize")
        train_acc, train_loss = train_distill(epoch, train_retain_loader, module_list, None, criterion_list, optimizer, t_opt, "minimize")

        losses.append(train_loss)
        epoch_list.append(epoch)
        acc_dict = train_validation(model, 
                                    train_retain_loader, 
                                    train_forget_loader, 
                                    valid_retain_loader, 
                                    valid_forget_loader)
        tr_accs.append(acc_dict['tr_acc'])
        tf_accs.append(acc_dict['tf_acc'])
        vr_accs.append(acc_dict['vr_acc'])
        vf_accs.append(acc_dict['vf_acc'])

        # Print checkpoint progress
        print(f"Epoch {epoch}: maximize loss: {maximize_loss:.2f}, minimize loss: {train_loss:.2f}, train_acc: {train_acc}")
        print(f'tr_acc: {acc_dict['tr_acc']}')
        print(f'tf_acc: {acc_dict['tf_acc']}')
        print(f'vr_acc: {acc_dict['vr_acc']}')
        print(f'vf_acc: {acc_dict['vf_acc']}')

    # Save a copy of the student model to use in evaluation
    save_path = "/content/Unlearning-MIA-Eval/Final_Structure/checkpoints/scrub_applied.pt"
    torch.save(model_s.state_dict(), save_path)

    history = {
        'losses': losses,
        'epoch_list': epoch_list,
        'tr_accs': tr_accs,
        'tf_accs': tf_accs,
        'vr_accs': vr_accs,
        'vf_accs': vf_accs
    }

    return model_s, history
