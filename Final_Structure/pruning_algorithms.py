"""
Pruning algorithms adapted from project1 for project2 framework
"""
import copy
import torch
import torch.nn as nn
from types import SimpleNamespace

from .pruning_utils import (
    check_sparsity, pruning_model, pruning_model_random, 
    extract_mask, remove_prune, prune_model_custom, global_prune_model
)
from .pruning_training import (
    get_optimizer_and_scheduler, train_with_rewind, validate, 
    convert_loaders_for_project1, create_pruning_args
)
from .training import load_model


def omp_pruning(full_model_path, loaders, args):
    """
    One-shot Magnitude Pruning (OMP) adapted for project2 framework
    """
    print("Starting OMP (One-shot Magnitude Pruning)")
    
    # Load model
    model = load_model(dataset=args.dataset, checkpoint_path=full_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Convert loaders to project1 format
    train_loader = loaders['train_loader']
    test_loader = loaders['test_loader']
    
    # Create criterion
    criterion = nn.CrossEntropyLoss()
    
    # Create pruning specific args
    pruning_args = create_pruning_args(args)
    
    # Get optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(model, pruning_args)
    
    # Check initial sparsity
    check_sparsity(model)
    
    # Train with rewind (to get rewind state)
    print("Training to get rewind state...")
    rewind_state_dict = train_with_rewind(
        model, optimizer, scheduler, train_loader, criterion, pruning_args
    )
    
    # Validate before pruning
    print("Performance before pruning:")
    val_loss, val_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.2f}%")
    
    # Pruning
    print(f"Applying pruning with rate: {pruning_args.rate}")
    if getattr(pruning_args, 'random_prune', False):
        print("Random pruning")
        pruning_model_random(model, pruning_args.rate)
    else:
        print("L1 magnitude pruning")
        pruning_model(model, pruning_args.rate)
    
    # Check sparsity after pruning
    check_sparsity(model)
    current_mask = extract_mask(model.state_dict())
    remove_prune(model)
    
    # Weight rewinding
    print("Weight rewinding...")
    model.load_state_dict(rewind_state_dict, strict=False)
    prune_model_custom(model, current_mask)
    
    # Retraining
    print("Retraining...")
    optimizer, scheduler = get_optimizer_and_scheduler(model, pruning_args)
    
    # Learning rate rewinding
    if pruning_args.rewind_epoch:
        for _ in range(pruning_args.rewind_epoch):
            scheduler.step()
    
    check_sparsity(model)
    
    # Final training
    pruning_args.stop_at_rewind = False  # Train full epochs
    train_with_rewind(model, optimizer, scheduler, train_loader, criterion, pruning_args)
    
    # Final validation
    print("Performance after pruning and retraining:")
    val_loss, val_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.2f}%")
    
    # Remove pruning hooks to get clean model weights
    print("Removing pruning hooks...")
    remove_prune(model)
    check_sparsity(model)
    
    return model


def synflow_pruning(full_model_path, loaders, args):
    """
    SynFlow pruning adapted for project2 framework
    """
    print("Starting SynFlow Pruning")
    
    # Load model
    model = load_model(dataset=args.dataset, checkpoint_path=full_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Convert loaders
    train_loader = loaders['train_loader']
    test_loader = loaders['test_loader']
    
    # Create criterion
    criterion = nn.CrossEntropyLoss()
    
    # Create pruning specific args
    pruning_args = create_pruning_args(args)
    
    # Get optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(model, pruning_args)
    
    # Apply SynFlow pruning
    if pruning_args.rate != 0:
        print(f"Applying SynFlow pruning with rate: {pruning_args.rate}")
        global_prune_model(model, pruning_args.rate, "synflow", train_loader)
        check_sparsity(model)
    
    # Training
    print("Training with SynFlow pruned model...")
    pruning_args.stop_at_rewind = False  # Train full epochs
    train_with_rewind(model, optimizer, scheduler, train_loader, criterion, pruning_args)
    
    # Final sparsity check
    check_sparsity(model)
    
    # Final validation
    print("Performance after SynFlow pruning and training:")
    val_loss, val_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.2f}%")
    
    # Remove pruning hooks if any exist
    try:
        print("Removing pruning hooks...")
        remove_prune(model)
        check_sparsity(model)
    except:
        print("No pruning hooks to remove")
    
    return model


def iterative_magnitude_pruning(full_model_path, loaders, args):
    """
    Iterative Magnitude Pruning (IMP) adapted for project2 framework
    """
    print("Starting IMP (Iterative Magnitude Pruning)")
    
    # Load model
    model = load_model(dataset=args.dataset, checkpoint_path=full_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Convert loaders
    train_loader = loaders['train_loader']
    test_loader = loaders['test_loader']
    
    # Create criterion
    criterion = nn.CrossEntropyLoss()
    
    # Create pruning specific args
    pruning_args = create_pruning_args(args)
    
    # Get initial rewind state
    optimizer, scheduler = get_optimizer_and_scheduler(model, pruning_args)
    print("Training to get initial rewind state...")
    rewind_state_dict = train_with_rewind(
        model, optimizer, scheduler, train_loader, criterion, pruning_args
    )
    
    # Iterative pruning
    pruning_times = getattr(pruning_args, 'pruning_times', 5)
    rate_per_iteration = pruning_args.rate
    
    print(f"Performing {pruning_times} pruning iterations with rate {rate_per_iteration} each")
    
    for iteration in range(pruning_times):
        print(f"\n=== Pruning Iteration {iteration + 1}/{pruning_times} ===")
        
        # Check current sparsity
        check_sparsity(model)
        
        # Prune
        if getattr(pruning_args, 'random_prune', False):
            pruning_model_random(model, rate_per_iteration)
        else:
            pruning_model(model, rate_per_iteration)
        
        # Check sparsity after pruning
        check_sparsity(model)
        current_mask = extract_mask(model.state_dict())
        remove_prune(model)
        
        # Weight rewinding
        model.load_state_dict(rewind_state_dict, strict=False)
        prune_model_custom(model, current_mask)
        
        # Retrain
        optimizer, scheduler = get_optimizer_and_scheduler(model, pruning_args)
        
        # Learning rate rewinding
        if pruning_args.rewind_epoch:
            for _ in range(pruning_args.rewind_epoch):
                scheduler.step()
        
        # Train
        pruning_args.stop_at_rewind = False
        train_with_rewind(model, optimizer, scheduler, train_loader, criterion, pruning_args)
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        print(f"Iteration {iteration + 1} - Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.2f}%")
    
    # Final validation
    print("\nFinal performance after all pruning iterations:")
    check_sparsity(model)
    val_loss, val_acc = validate(model, test_loader, criterion, device)
    print(f"Final Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.2f}%")
    
    # Remove pruning hooks to get clean model weights
    print("Removing pruning hooks...")
    remove_prune(model)
    check_sparsity(model)
    
    return model
    
    return model