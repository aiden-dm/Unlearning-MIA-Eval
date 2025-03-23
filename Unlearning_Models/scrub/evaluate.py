# Imports
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import Namespace
from tqdm.autonotebook import tqdm

# Torch Imports
import torch
import torch.nn as nn

# Adding the SCRUB repo to the system path
SCRUB_PATH = os.path.abspath("../../Third_Party_Code/SCRUB")
if SCRUB_PATH not in sys.path:
    sys.path.append(SCRUB_PATH)

# Third party imports from the SCRUB repository
from Third_Party_Code.SCRUB.utils import *

def cm_score(estimator, X, y):
    y_pred = estimator.predict(X)
    cnf_matrix = confusion_matrix(y, y_pred)
    
    FP = cnf_matrix[0][1] 
    FN = cnf_matrix[1][0] 
    TP = cnf_matrix[0][0] 
    TN = cnf_matrix[1][1]

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    # Specificity or true negative rate
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    # Precision or positive predictive value
    PPV = TP / (TP + FP) if (TP + FP) > 0 else 0
    # Negative predictive value
    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0
    # Fall out or false positive rate
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    # False negative rate
    FNR = FN / (TP + FN) if (TP + FN) > 0 else 0
    # False discovery rate
    FDR = FP / (TP + FP) if (TP + FP) > 0 else 0
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0

    print(f"FPR:{FPR:.2f}, FNR:{FNR:.2f}, FP:{FP:.2f}, TN:{TN:.2f}, TP:{TP:.2f}, FN:{FN:.2f}")
    return ACC

def evaluate_attack_model(sample_loss,
                          members,
                          n_splits = 5,
                          random_state = None):

  unique_members = np.unique(members)
  if not np.all(unique_members == np.array([0, 1])):
    raise ValueError("members should only have 0 and 1s")

  attack_model = LogisticRegression()
  cv = StratifiedShuffleSplit(
      n_splits=n_splits, random_state=random_state)
  return cross_val_score(attack_model, sample_loss, members, cv=cv, scoring=cm_score)

def membership_inference_attack(model, t_loader, f_loader, seed, args):
    
    fgt_cls = list(np.unique(f_loader.dataset.targets))
    indices = [i in fgt_cls for i in t_loader.dataset.targets]
    t_loader.dataset.data = t_loader.dataset.data[indices]
    t_loader.dataset.targets = t_loader.dataset.targets[indices]

    cr = nn.CrossEntropyLoss(reduction='none')
    test_losses = []
    forget_losses = []
    model.eval()
    mult = 0.5 if args.lossfn=='mse' else 1
    dataloader = torch.utils.data.DataLoader(t_loader.dataset, batch_size=128, shuffle=False)
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(args.device), target.to(args.device)            
        if args.lossfn=='mse':
            target=(2*target-1)
            target = target.type(torch.cuda.FloatTensor).unsqueeze(1)
        if 'mnist' in args.dataset:
            data=data.view(data.shape[0],-1)
        output = model(data)
        loss = mult*cr(output, target)
        test_losses = test_losses + list(loss.cpu().detach().numpy())
    del dataloader
    dataloader = torch.utils.data.DataLoader(f_loader.dataset, batch_size=128, shuffle=False)
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(args.device), target.to(args.device)            
        if args.lossfn=='mse':
            target=(2*target-1)
            target = target.type(torch.cuda.FloatTensor).unsqueeze(1)
        if 'mnist' in args.dataset:
            data=data.view(data.shape[0],-1)
        output = model(data)
        loss = mult*cr(output, target)
        forget_losses = forget_losses + list(loss.cpu().detach().numpy())
    del dataloader

    np.random.seed(seed)
    random.seed(seed)
    if len(forget_losses) > len(test_losses):
        forget_losses = list(random.sample(forget_losses, len(test_losses)))
    elif len(test_losses) > len(forget_losses):
        test_losses = list(random.sample(test_losses, len(forget_losses)))
    
    fig, ax = plt.subplots()
    sns.histplot(np.array(test_losses), kde=False, bins=30, label='test-loss', ax=ax)
    sns.histplot(np.array(forget_losses), kde=False, bins=30, label='forget-loss', ax=ax)
    ax.legend(prop={'size': 14})
    ax.tick_params(labelsize=12)
    ax.set_title("Loss Histograms", size=18)
    ax.set_xlabel('Loss Values', size=14)
    plt.show()

    print("Test Losses - Max:", np.max(test_losses), "Min:", np.min(test_losses))
    print("Forget Losses - Max:", np.max(forget_losses), "Min:", np.min(forget_losses))

    test_labels = [0]*len(test_losses)
    forget_labels = [1]*len(forget_losses)
    features = np.array(test_losses + forget_losses).reshape(-1,1)
    labels = np.array(test_labels + forget_labels).reshape(-1)
    features = np.clip(features, -100, 100)
    score = evaluate_attack_model(features, labels, n_splits=5, random_state=seed)

    return score

def l2_penalty(model,model_init,weight_decay):
    l2_loss = 0
    for (k,p),(k_init,p_init) in zip(model.named_parameters(),model_init.named_parameters()):
        if p.requires_grad:
            l2_loss += (p-p_init).pow(2).sum()
    l2_loss *= (weight_decay/2.)
    return l2_loss

def run_train_epoch(model: nn.Module, model_init, data_loader: torch.utils.data.DataLoader, 
                    loss_fn: nn.Module, args: Namespace,
                    optimizer: torch.optim.SGD, split: str, epoch=0, ignore_index=None,
                    negative_gradient=False, negative_multiplier=-1, random_labels=False,
                    quiet=False,delta_w=None,scrub_act=False):
    model.eval()
    metrics = AverageMeter()    
    num_labels = data_loader.dataset.targets.max().item() + 1
    
    with torch.set_grad_enabled(split != 'test'):
        for idx, batch in enumerate(tqdm(data_loader, leave=False)):
            batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            input, target = batch
            output = model(input)
            if split=='test' and scrub_act:
                G = []
                for cls in range(args.num_classes):
                    grads = torch.autograd.grad(output[0,cls],model.parameters(),retain_graph=True)
                    grads = torch.cat([g.view(-1) for g in grads])
                    G.append(grads)
                grads = torch.autograd.grad(output_sf[0,cls],model_scrubf.parameters(),retain_graph=False)
                G = torch.stack(G).pow(2)
                delta_f = torch.matmul(G,delta_w)
                output += delta_f.sqrt()*torch.empty_like(delta_f).normal_()
            loss = loss_fn(output, target) + l2_penalty(model,model_init,args.weight_decay)
            metrics.update(n=input.size(0), loss=loss_fn(output,target).item(), error=get_error(output, target))
            
            if split != 'test':
                model.zero_grad()
                loss.backward()
                optimizer.step()
    #if not quiet:
        #log_metrics(split, metrics, epoch)
    return metrics.avg

def readout_retrain(model, data_loader, test_loader, args, lr=0.1, epochs=500, threshold=0.01, quiet=True):
    torch.manual_seed(args.seed)
    model = copy.deepcopy(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0)
    sampler = torch.utils.data.RandomSampler(data_loader.dataset, replacement=True, num_samples=500)
    data_loader_small = torch.utils.data.DataLoader(data_loader.dataset, batch_size=data_loader.batch_size, sampler=sampler, num_workers=data_loader.num_workers)
    metrics = []
    model_init=copy.deepcopy(model)
    for epoch in range(epochs):
        metrics.append(run_train_epoch(model, model_init, test_loader, loss_fn, args, optimizer, split='test', epoch=epoch, ignore_index=None, quiet=quiet))
        if metrics[-1]['loss'] <= threshold:
            break
        run_train_epoch(model, model_init, data_loader_small, loss_fn, args, optimizer, split='train', epoch=epoch, ignore_index=None, quiet=quiet)
    return epoch, metrics

def extract_retrain_time(metrics, threshold=0.1):
    losses = np.array([m['loss'] for m in metrics])
    return np.argmax(losses < threshold)

def test(model, data_loader, args):
    loss_fn = nn.CrossEntropyLoss()
    model_init=copy.deepcopy(model)
    return run_train_epoch(model, model_init, data_loader, loss_fn, args, optimizer=None, split='test', ignore_index=None, quiet=True)

def all_readouts(model, loaders, args, thresh=0.1, name='method', seed=0):

    # Unpacking the loaders for use in the function
    forget_loader, retain_loader, train_loader_full, valid_loader_full, test_loader_full = loaders

    MIA = membership_inference_attack(model, copy.deepcopy(test_loader_full), forget_loader, seed, args)
    #train_loader = torch.utils.data.DataLoader(train_loader_full.dataset, batch_size=128, shuffle=True)
    retrain_time, _ = 0,0 
    #readout_retrain(model, train_loader, forget_loader, args, epochs=100, lr=0.001, threshold=thresh)
    test_error = test(model, test_loader_full, args)['error']*100
    forget_error = test(model, forget_loader, args)['error']*100
    retain_error = test(model, retain_loader, args)['error']*100
    val_error = test(model, valid_loader_full, args)['error']*100
    
    print(f"{name} ->"
          f"\tFull test error: {test_error:.2f}"
          f"\tForget error: {forget_error:.2f}\tRetain error: {retain_error:.2f}\tValid error: {val_error:.2f}"
          f"\tFine-tune time: {retrain_time+1} steps\tMIA: {np.mean(MIA):.2f}±{np.std(MIA):0.1f}")
    
    return(dict(test_error=test_error, forget_error=forget_error, retain_error=retain_error, val_error=val_error, retrain_time=retrain_time+1, MIA=np.mean(MIA)))



