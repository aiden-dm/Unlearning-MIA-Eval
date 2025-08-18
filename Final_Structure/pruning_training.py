"""
Training utilities for pruning, adapted from project1
"""
import torch
import torch.nn as nn
import time
from types import SimpleNamespace


def get_optimizer_and_scheduler(model, args):
    """Get optimizer and scheduler based on args"""
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=getattr(args, 'momentum', 0.9),
        weight_decay=getattr(args, 'weight_decay', 5e-4),
    )
    
    decreasing_lr = [int(x) for x in getattr(args, 'decreasing_lr', '30,50').split(',')]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )
    
    return optimizer, scheduler


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if (i + 1) % 100 == 0:
            print(f'Batch [{i+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def train_with_rewind(model, optimizer, scheduler, train_loader, criterion, args):
    """Train model with early stopping for rewind state"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    rewind_epoch = getattr(args, 'rewind_epoch', 8)
    epochs = getattr(args, 'epochs', 100)
    
    rewind_state_dict = None
    
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Save rewind state
        if epoch == rewind_epoch - 1:
            rewind_state_dict = model.state_dict().copy()
            print(f"Saved rewind state at epoch {epoch+1}")
        
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        
        # Early stopping at rewind epoch for initial training
        if hasattr(args, 'stop_at_rewind') and args.stop_at_rewind and epoch >= rewind_epoch:
            break
    
    return rewind_state_dict if rewind_state_dict is not None else model.state_dict().copy()


def convert_loaders_for_project1(loaders):
    """Convert project2 loader format to project1 format"""
    return {
        'retain': loaders['train_retain_loader'],
        'forget': loaders['train_forget_loader'],
        'test': loaders['test_loader']
    }


def create_pruning_args(base_args, **pruning_specific):
    """Create args for pruning with defaults"""
    args = SimpleNamespace()
    
    # Copy base args
    for key, value in vars(base_args).items():
        setattr(args, key, value)
    
    # Set pruning specific defaults
    args.learning_rate = getattr(base_args, 'learning_rate', 0.1)
    args.momentum = getattr(base_args, 'momentum', 0.9)
    args.weight_decay = getattr(base_args, 'weight_decay', 5e-4)
    args.epochs = getattr(base_args, 'epochs', 100)
    args.rewind_epoch = getattr(base_args, 'rewind_epoch', 8)
    args.decreasing_lr = getattr(base_args, 'decreasing_lr', '30,50')
    args.rate = getattr(base_args, 'rate', 0.2)
    args.random_prune = getattr(base_args, 'random_prune', False)
    args.stop_at_rewind = True  # For initial training
    
    # Override with specific args
    for key, value in pruning_specific.items():
        setattr(args, key, value)
    
    return args