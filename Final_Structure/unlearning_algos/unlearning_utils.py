"""
Unlearning utilities adapted from project1 for project2 framework
"""
import copy
import torch
import torch.nn as nn
import time
from types import SimpleNamespace

from Final_Structure.training import load_model

def l1_regularization(model):
    """L1 regularization for model parameters"""
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def l2_regularization(model):
    """L2 regularization for model parameters"""
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=2)


def convert_loaders_for_unlearning(loaders):
    """Convert project2 loader format to project1 unlearning format"""
    return {
        'retain': loaders['train_retain_loader'],
        'forget': loaders['train_forget_loader'],
        'test': loaders['test_loader']
    }


def create_unlearning_args(base_args, **unlearning_specific):
    """Create args for unlearning with defaults"""
    args = SimpleNamespace()
    
    # Copy base args
    for key, value in vars(base_args).items():
        setattr(args, key, value)
    
    # Set unlearning specific defaults
    args.unlearn_lr = getattr(base_args, 'unlearn_lr', 0.01)
    args.unlearn_epochs = getattr(base_args, 'unlearn_epochs', 10)
    args.momentum = getattr(base_args, 'momentum', 0.9)
    args.weight_decay = getattr(base_args, 'weight_decay', 5e-4)
    args.decreasing_lr = getattr(base_args, 'decreasing_lr', '5,8')
    args.alpha = getattr(base_args, 'alpha', 0.001)
    args.warmup = getattr(base_args, 'warmup', 0)
    args.no_l1_epochs = getattr(base_args, 'no_l1_epochs', 0)
    args.print_freq = getattr(base_args, 'print_freq', 50)
    
    # Override with specific args
    for key, value in unlearning_specific.items():
        setattr(args, key, value)
    
    return args


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def iterative_unlearn_wrapper(unlearn_func):
    """
    Wrapper to handle iterative unlearning adapted for project2
    """
    def wrapper(full_model_path, loaders, args):
        # Load model
        model = load_model(dataset=args.dataset, checkpoint_path=full_model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Convert loaders
        data_loaders = convert_loaders_for_unlearning(loaders)
        
        # Create criterion
        criterion = nn.CrossEntropyLoss()
        
        # Create unlearning args
        unlearn_args = create_unlearning_args(args)
        
        # Setup optimizer
        optimizer = torch.optim.SGD(
            model.parameters(),
            unlearn_args.unlearn_lr,
            momentum=unlearn_args.momentum,
            weight_decay=unlearn_args.weight_decay,
        )
        
        # Setup scheduler
        decreasing_lr = list(map(int, unlearn_args.decreasing_lr.split(",")))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=decreasing_lr, gamma=0.1
        )
        
        # Run unlearning
        for epoch in range(unlearn_args.unlearn_epochs):
            start_time = time.time()
            print(f"Epoch #{epoch}, Learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")
            
            train_acc = unlearn_func(data_loaders, model, criterion, optimizer, epoch, unlearn_args)
            scheduler.step()
            
            print(f"One epoch duration: {time.time() - start_time:.2f}s")
        
        return model
    
    return wrapper


def validate_unlearning(model, data_loaders, criterion, device):
    """Validate unlearning performance"""
    model.eval()
    
    results = {}
    
    for split, loader in data_loaders.items():
        if loader is None:
            continue
            
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = running_loss / len(loader)
        acc = 100. * correct / total
        results[split] = {'loss': avg_loss, 'accuracy': acc}
        
        print(f"{split} - Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
    
    return results