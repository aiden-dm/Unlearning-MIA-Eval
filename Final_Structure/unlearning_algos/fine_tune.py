import time
import torch
import torch.nn as nn
import torch.optim as optim

from Final_Structure.unlearning_algos.unlearning_inputs import FineTuneInput
from Final_Structure.training import load_model
from Final_Structure.unlearning_algos.unlearning_utils import (
    l1_regularization, accuracy, AverageMeter
)

def ft_iter(data_loaders, model, args: FineTuneInput):
    """Fine-tuning iteration"""
    train_loader = data_loaders["train_retain_loader"]
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    model.train()
    device = next(model.parameters()).device
    
    start = time.time()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    epoch = args.epoch
    with_l1 = args.with_l1
    
    for i, (image, target) in enumerate(train_loader):
        image = image.to(device)
        target = target.to(device)
        
        # Calculate alpha for L1 regularization
        if epoch < args.unlearn_epochs - args.no_l1_epochs:
            current_alpha = args.alpha * (
                1 - epoch / (args.unlearn_epochs - args.no_l1_epochs)
            )
        elif args.unlearn_epochs - args.no_l1_epochs == 0:
            current_alpha = args.alpha
        else:
            current_alpha = 0
        
        # Forward pass
        output_clean = model(image)
        loss = criterion(output_clean, target)
        
        # Add L1 regularization if requested
        if with_l1:
            loss += current_alpha * l1_regularization(model)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record statistics
        output = output_clean.float()
        loss = loss.float()
        prec1 = accuracy(output.data, target)[0]
        
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))
        
        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print(f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                  f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                  f"Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                  f"Time {end - start:.2f}")
            start = time.time()
    
    print(f"train_accuracy {top1.avg:.3f}")
    return top1.avg

def fine_tune(loaders, args: FineTuneInput):

    model = load_model(dataset=args.dataset, checkpoint_path=args.model_path)
    ft_iter(loaders, model, args)

    if args.check_path is not None:
        torch.save(model.state_dict(), args.check_path)

    return model