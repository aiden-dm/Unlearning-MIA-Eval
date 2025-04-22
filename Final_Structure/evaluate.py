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
from sklearn.utils import resample
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_tuning_accs(history):
    plt.plot(history['epoch_list'], history['tr_accs'], marker='*', color=u'#1f77b4', alpha=1, label='Train Retain Set')
    plt.plot(history['epoch_list'], history['tf_accs'], marker='o', color=u'#ff7f0e', alpha=1, label='Train Forget Set')
    plt.plot(history['epoch_list'], history['vr_accs'], marker='^', color=u'#2ca02c', alpha=1, label='Validation Retain Set')
    plt.plot(history['epoch_list'], history['vf_accs'], marker='.', color='red', alpha=1, label='Validation Forget Set')
    plt.legend(prop={'size': 14})
    plt.tick_params(labelsize=12)
    plt.xlabel('epoch',size=14)
    plt.ylabel('accuracy (%)',size=14)
    plt.grid()
    plt.show()

def plot_loss_distributions(test_losses, forget_losses, bins=50):
    plt.figure()
    bins = np.linspace(0, 15, 150)
    plt.hist(test_losses,  bins=bins, density=True, alpha=0.6, label="Test")
    plt.hist(forget_losses,bins=bins, density=True, alpha=0.6, label="Forgotten")
    plt.xlabel("Cross‑Entropy Loss")
    plt.ylabel("Count")
    plt.title("Loss Distributions: Test vs Forgotten")
    plt.legend()
    plt.show()

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

def membership_inference_attack(model, t_loader, f_loader, device, seed, plot_dist):

    # Initialization
    cr = nn.CrossEntropyLoss(reduction='none')
    test_losses = []
    forget_losses = []
    model.eval()

    # Calculating test losses for class 0-9 (members)
    for batch_idx, (data, target) in enumerate(t_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = cr(output, target)
        test_losses.extend(loss.cpu().detach().numpy())

    # Calculating forget losses for class 1 (forgotten class)
    for batch_idx, (data, target) in enumerate(f_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = cr(output, target)
        forget_losses.extend(loss.cpu().detach().numpy())

    # Plot the distributions of losses
    if plot_dist:
        plot_loss_distributions(test_losses, forget_losses)

    # Undersample the majority class (test data)
    if len(test_losses) > len(forget_losses):
        test_losses = resample(test_losses, replace=False, n_samples=len(forget_losses), random_state=seed)
    elif len(forget_losses) > len(test_losses):
        forget_losses = resample(forget_losses, replace=False, n_samples=len(test_losses), random_state=seed)

    # Preparing data for evaluation
    test_labels = [0] * len(test_losses)  # Members (test set)
    forget_labels = [1] * len(forget_losses)  # Non-members (forgotten set)
    features = np.array(test_losses + forget_losses).reshape(-1, 1)
    labels = np.array(test_labels + forget_labels).reshape(-1)

    # Evaluate the attack model
    all_metrics = evaluate_attack_model(features, labels, n_splits=5, random_state=seed)

    # Calculate mean and std for each metric
    df_metrics = pd.DataFrame(all_metrics)
    mean_metrics = df_metrics.mean()
    std_metrics = df_metrics.std()

    return mean_metrics, std_metrics