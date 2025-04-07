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

# Imports from the SCRUB repository
from Third_Party_Code.SCRUB.thirdparty.repdistiller.distiller_zoo import DistillKL
from Third_Party_Code.SCRUB.thirdparty.repdistiller.helper.loops import train_distill, validate

def scrub(model, loaders, args):

    # Extracting the loaders that we want
    valid_loader = loaders[1]
    train_forget_loader = loaders[3]
    train_retain_loader = loaders[4]
    valid_forget_loader = loaders[5]

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
    acc_rs, acc_fs, acc_vs, acc_fvs = [], [], [], []

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

        # Validate on retained and forgotten data
        acc_r, _, _ = validate(train_retain_loader, model_s, criterion_cls, v_opt, True)
        acc_f, _, _ = validate(train_forget_loader, model_s, criterion_cls, v_opt, True)
        acc_v, _, _ = validate(valid_loader, model_s, criterion_cls, v_opt, True)
        acc_fv, _, _ = validate(valid_forget_loader, model_s, criterion_cls, v_opt, True)

        # Storing accuracy results
        acc_rs.append(100 - acc_r.item())
        acc_fs.append(100 - acc_f.item())
        acc_vs.append(100 - acc_v.item())
        acc_fvs.append(100-acc_fv.item())

        # Train model
        maximize_loss = 0
        if epoch <= msteps:
            maximize_loss = train_distill(epoch, train_forget_loader, module_list, None, criterion_list, optimizer, t_opt, "maximize")
        train_acc, train_loss = train_distill(epoch, train_retain_loader, module_list, None, criterion_list, optimizer, t_opt, "minimize")

        # Print checkpoint progress
        print(f"Epoch {epoch}: maximize loss: {maximize_loss:.2f}, minimize loss: {train_loss:.2f}, train_acc: {train_acc}")
    
    # Saving final accuracies after training
    acc_r, _, _ = validate(train_retain_loader, model_s, criterion_cls, t_opt, True)
    acc_f, _, _ = validate(train_forget_loader, model_s, criterion_cls, t_opt, True)
    acc_v, _, _ = validate(valid_loader, model_s, criterion_cls, t_opt, True)
    acc_fv, _, _ = validate(valid_forget_loader, model_s, criterion_cls, t_opt, True)
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
    plt.savefig('/content/Unlearning-MIA-Eval/Final_Structure/checkpoints/scrub_unlearning_accuracy_plot.png', bbox_inches='tight')
    
    # Save a copy of the student model to use in evaluation
    save_path = "/content/Unlearning-MIA-Eval/Final_Structure/checkpoints/scrub_applied.pt"
    torch.save(model_s.state_dict(), save_path)

    return model_s
