# Imports
import copy
import torch.nn as nn
import torch.optim as optim
import os
import sys
import matplotlib.pyplot as plt
from types import SimpleNamespace

# Adding the SCRUB repo to the system path
SCRUB_PATH = os.path.abspath("../../Third_Party_Code/SCRUB")
if SCRUB_PATH not in sys.path:
    sys.path.append(SCRUB_PATH)

# Import from our local files
from training import load_model, get_loaders

# Imports from the SCRUB repository
from Third_Party_Code.SCRUB.thirdparty.repdistiller.distiller_zoo import DistillKL
from Third_Party_Code.SCRUB.thirdparty.repdistiller.helper.loops import train_distill, validate

def scrub(model, loaders):

    # Unpacking the data loaders
    [train_loader, 
     valid_loader, 
     test_loader, 
     train_forget_loader, 
     train_retain_loader, 
     valid_forget_loader,
     valid_retain_loader, 
     test_forget_loader, 
     test_retain_loader
    ] = loaders

    # Defining hyperparameters
    kd_T = 2
    sgda_learning_rate = 0.0005
    sgda_weight_decay = 0.1
    sgda_epochs = 10
    msteps = 3

    '''
    args.sgda_epochs = 10
    args.sgda_learning_rate = 0.0005
    args.lr_decay_epochs = [5,8,9]
    args.lr_decay_rate = 0.1
    args.sgda_weight_decay = 0.1#5e-4
    args.sgda_momentum = 0.9
    '''

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
    v_opt.print_freq = 1

    # Define train distill args
    t_opt = SimpleNamespace()
    t_opt.distill = 'kd'
    t_opt.alpha = 0.5
    t_opt.beta = 0
    t_opt.gamma = 1
    t_opt.print_freq = 1
    #t_opt.clip_grad

    '''
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
    '''

    # Training loop
    for epoch in range(1, sgda_epochs + 1):
        #lr = sgda_adjust_learning_rate(epoch, args_f, optimizer)
        print("PERFROMING SCRUB UNLEARNING...")

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

        # Save the model to a checkpoint file
        # WE CAN WRITE LOGIC TO SAVE A CHECKPOINT
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
    plt.show()

    # Return model information
    try:
        selected_idx, _ = min(enumerate(acc_fs), key=lambda x: abs(x[1]-acc_fvs[-1]))
    except:
        selected_idx = len(acc_fs) - 1
    print ("the selected index is {}".format(selected_idx))
    #selected_model = "checkpoints/scrub_{}_{}_seed{}_step{}.pt".format(args.model, args.dataset, args.seed, int(selected_idx))
    model_s_final = copy.deepcopy(model_s)
    #model_s.load_state_dict(torch.load(selected_model))
    
    return model_s, model_s_final

model = load_model("./checkpoints/resnet_full.pt")
loaders = get_loaders(root='./data', forget_classes=[1])
scrub(model, loaders)
