import torch

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