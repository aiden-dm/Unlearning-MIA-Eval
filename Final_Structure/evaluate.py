# Imports
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import numpy as np
import random
import pandas as pd

def compute_accuracy(model, loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            outputs = model(x)
            
            # Get the predicted class by taking the argmax along the class dimension
            _, predicted = torch.max(outputs, 1)
            
            # Count correct predictions
            correct += (predicted == y).sum().item()
            total += y.size(0)
    
    accuracy = 100 * correct / total
    return accuracy


def train_validation(model, train_retain, train_forget, val_retain, val_forget, device='cuda'):
    return_dict = dict()

    t_retain_acc = compute_accuracy(model, train_retain, device)
    t_forget_acc = compute_accuracy(model, train_forget, device)
    v_retain_acc = compute_accuracy(model, val_retain, device)
    v_forget_acc = compute_accuracy(model, val_forget, device)

    return_dict['tr_acc'] = t_retain_acc
    return_dict['tf_acc'] = t_forget_acc
    return_dict['vr_acc'] = v_retain_acc
    return_dict['vf_acc'] = v_forget_acc

    return return_dict

# Function that calculates a variety of evaluation metrics post unlearning
def evaluate_model(model, dataloader, device):
    # Putting model in evaluation mode
    model.eval()

    # Defining variables to store evaluation information
    all_preds = []
    all_labels = []
    losses = []
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Evaluation loop
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            losses.append(criterion(outputs, labels).cpu())

    # Organizing predicted and true values
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    # Compute evaluation metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Full per-class report
    report = classification_report(y_true, y_pred, zero_division=0)

    # Preparing losses for return
    losses = torch.cat(losses)

    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'report': report,
        'losses': losses
    }

    return metrics

def cm_score(estimator, X, y):
    y_pred = estimator.predict(X)
    cnf_matrix = confusion_matrix(y, y_pred)

    FP = cnf_matrix[0][1]
    FN = cnf_matrix[1][0]
    TP = cnf_matrix[0][0]
    TN = cnf_matrix[1][1]

    TPR = TP / (TP + FN + 1e-10)
    TNR = TN / (TN + FP + 1e-10)
    PPV = TP / (TP + FP + 1e-10)
    NPV = TN / (TN + FN + 1e-10)
    FPR = FP / (FP + TN + 1e-10)
    FNR = FN / (TP + FN + 1e-10)
    FDR = FP / (TP + FP + 1e-10)
    ACC = (TP + TN) / (TP + FP + FN + TN + 1e-10)

    metrics = {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'TPR': TPR, 'TNR': TNR, 'PPV': PPV, 'NPV': NPV,
        'FPR': FPR, 'FNR': FNR, 'FDR': FDR,
        'ACC': ACC
    }
    
    return metrics

def evaluate_attack_model(sample_loss, members, n_splits = 5, random_state = None):
    
    attack_model = LogisticRegression()
    cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=random_state)

    all_metrics = []

    for train_idx, test_idx in cv.split(sample_loss, members):
        X_train, X_test = sample_loss[train_idx], sample_loss[test_idx]
        y_train, y_test = members[train_idx], members[test_idx]

        attack_model.fit(X_train, y_train)
        fold_metrics = cm_score(attack_model, X_test, y_test)
        all_metrics.append(fold_metrics)

    return all_metrics
    
def membership_inference_attack(model, t_loader, f_loader, device, seed):

    # Initialization
    cr = nn.CrossEntropyLoss(reduction='none')
    test_losses = []
    forget_losses = []
    model.eval()

    # Calculating test losses
    for batch_idx, (data, target) in enumerate(t_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = cr(output, target)
        test_losses.extend(loss.cpu().detach().numpy())

    # Calculating forget losses
    for batch_idx, (data, target) in enumerate(f_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = cr(output, target)
        forget_losses.extend(loss.cpu().detach().numpy())

    # Ensuring test and forget losses are same length
    if len(forget_losses) > len(test_losses):
        forget_losses = list(random.sample(forget_losses, len(test_losses)))
    elif len(test_losses) > len(forget_losses):
        test_losses = list(random.sample(test_losses, len(forget_losses)))

    # Preparing data for evaluation
    test_labels = [0]*len(test_losses)
    forget_labels = [1]*len(forget_losses)
    features = np.array(test_losses + forget_losses).reshape(-1,1)
    labels = np.array(test_labels + forget_labels).reshape(-1)
    features = np.clip(features, -100, 100)
    all_metrics = evaluate_attack_model(features, labels, n_splits=5, random_state=seed)

    # Calculating mean and std for each metric
    df_metrics = pd.DataFrame(all_metrics)
    mean_metrics = df_metrics.mean()
    std_metrics = df_metrics.std()

    return mean_metrics, std_metrics
