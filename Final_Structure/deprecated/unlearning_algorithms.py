"""
Unlearning algorithms adapted from project1 for project2 framework
"""
import copy
import torch
import torch.nn as nn
from torch.autograd import grad
from tqdm import tqdm
import time

from .unlearning_utils import (
    iterative_unlearn_wrapper, l1_regularization, l2_regularization,
    convert_loaders_for_unlearning, create_unlearning_args,
    accuracy, AverageMeter, validate_unlearning
)
from ..training import load_model, get_resnet_model


def ft_iter(data_loaders, model, criterion, optimizer, epoch, args, with_l1=False):
    """Fine-tuning iteration"""
    train_loader = data_loaders["retain"]
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    model.train()
    device = next(model.parameters()).device
    
    start = time.time()
    
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


@iterative_unlearn_wrapper
def ft_unlearning(data_loaders, model, criterion, optimizer, epoch, args):
    """Fine-tuning unlearning"""
    return ft_iter(data_loaders, model, criterion, optimizer, epoch, args)


@iterative_unlearn_wrapper
def ft_l1_unlearning(data_loaders, model, criterion, optimizer, epoch, args):
    """Fine-tuning with L1 regularization"""
    return ft_iter(data_loaders, model, criterion, optimizer, epoch, args, with_l1=True)


@iterative_unlearn_wrapper
def ga_unlearning(data_loaders, model, criterion, optimizer, epoch, args):
    """Gradient Ascent unlearning - faithful reproduction of project1 implementation"""
    train_loader = data_loaders["forget"]
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    model.train()
    device = next(model.parameters()).device
    
    start = time.time()
    
    for i, (image, target) in enumerate(train_loader):
        image = image.to(device)
        target = target.to(device)
        
        # Forward pass
        output_clean = model(image)
        loss = -criterion(output_clean, target)  # Negative for gradient ascent
        
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


@iterative_unlearn_wrapper
def ga_l1_unlearning(data_loaders, model, criterion, optimizer, epoch, args):
    """Gradient Ascent with L1 regularization - faithful reproduction of project1 implementation"""
    train_loader = data_loaders["forget"]
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    model.train()
    device = next(model.parameters()).device
    
    start = time.time()
    
    for i, (image, target) in enumerate(train_loader):
        image = image.to(device)
        target = target.to(device)
        
        # Forward pass
        output_clean = model(image)
        loss = -criterion(output_clean, target) + args.alpha * l1_regularization(model)
        
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


def fisher_information_matrix(model, train_dl, device):
    """Calculate Fisher Information Matrix"""
    model.eval()
    fisher_approximation = []
    for parameter in model.parameters():
        fisher_approximation.append(torch.zeros_like(parameter).to(device))
    
    total = 0
    for i, (data, label) in enumerate(tqdm(train_dl)):
        data = data.to(device)
        label = label.to(device)
        predictions = torch.log_softmax(model(data), dim=-1)
        real_batch = data.shape[0]
        
        epsilon = 1e-8
        for i in range(real_batch):
            label_i = label[i]
            prediction = predictions[i][label_i]
            gradient = grad(
                prediction, model.parameters(), retain_graph=True, create_graph=False
            )
            for j, derivative in enumerate(gradient):
                fisher_approximation[j] += (derivative + epsilon) ** 2
        total += real_batch
    
    for i, parameter in enumerate(model.parameters()):
        fisher_approximation[i] = fisher_approximation[i] / total
    
    return fisher_approximation


def fisher_unlearning(full_model_path, loaders, args):
    """Fisher Information unlearning"""
    print("Starting Fisher Information unlearning")
    
    # Load model
    model = load_model(dataset=args.dataset, checkpoint_path=full_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Convert loaders
    data_loaders = convert_loaders_for_unlearning(loaders)
    retain_loader = data_loaders["retain"]
    
    # Create unlearning args
    unlearn_args = create_unlearning_args(args)
    
    # Calculate Fisher Information Matrix
    print("Calculating Fisher Information Matrix...")
    fisher_approximation = fisher_information_matrix(model, retain_loader, device)
    
    # Apply Fisher noise
    print("Applying Fisher noise...")
    for i, parameter in enumerate(model.parameters()):
        noise = torch.sqrt(unlearn_args.alpha / fisher_approximation[i]).clamp(
            max=1e-3
        ) * torch.empty_like(parameter).normal_(0, 1)
        
        # Special handling for last layer
        if parameter.shape[-1] == 10:  # Assuming 10 classes for CIFAR-10
            noise = noise * 10
        
        parameter.data = parameter.data + noise
    
    # Validate
    criterion = nn.CrossEntropyLoss()
    results = validate_unlearning(model, data_loaders, criterion, device)
    
    return model


def ft_prune_unlearning(full_model_path, loaders, args):
    """Fine-tuning with pruning unlearning"""
    print("Starting FT-Prune unlearning")
    
    # Load model
    model = load_model(dataset=args.dataset, checkpoint_path=full_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Convert loaders
    data_loaders = convert_loaders_for_unlearning(loaders)
    test_loader = data_loaders["test"]
    
    # Save checkpoint
    initialization = copy.deepcopy(model.state_dict())
    
    # Apply FT with L1 regularization
    print("Applying Fine-tuning with L1 regularization...")
    model = ft_l1_unlearning(full_model_path, loaders, args)
    
    # Validate
    criterion = nn.CrossEntropyLoss()
    results = validate_unlearning(model, data_loaders, criterion, device)
    
    return model


def sam_grad(model, loss):
    """Get flattened gradients for all model parameters"""
    params = []
    for param in model.parameters():
        params.append(param)
    sample_grad = grad(loss, params)
    sample_grad = [x.view(-1) for x in sample_grad]
    return torch.cat(sample_grad)


def apply_perturb(model, v):
    """Apply perturbation vector to model parameters"""
    curr = 0
    with torch.no_grad():
        for param in model.parameters():
            length = param.view(-1).shape[0]
            param += v[curr : curr + length].view(param.shape)
            curr += length


def woodfisher(model, train_dl, device, criterion, v):
    """Woodfisher algorithm for computing parameter updates"""
    model.eval()
    k_vec = torch.clone(v)
    N = 1000
    o_vec = None
    
    for idx, (data, label) in enumerate(tqdm(train_dl)):
        model.zero_grad()
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        sample_grad = sam_grad(model, loss)
        
        with torch.no_grad():
            if o_vec is None:
                o_vec = torch.clone(sample_grad)
            else:
                tmp = torch.dot(o_vec, sample_grad)
                k_vec -= (torch.dot(k_vec, sample_grad) / (N + tmp)) * o_vec
                o_vec -= (tmp / (N + tmp)) * o_vec
        
        if idx > N:
            return k_vec
    return k_vec


def wfisher_unlearning(full_model_path, loaders, args):
    """Woodfisher unlearning algorithm"""
    print("Starting Woodfisher unlearning")
    
    # Load model
    model = load_model(dataset=args.dataset, checkpoint_path=full_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Convert loaders
    data_loaders = convert_loaders_for_unlearning(loaders)
    retain_loader = data_loaders["retain"]
    forget_loader = data_loaders["forget"]
    
    # Create single-batch loaders for Woodfisher
    retain_grad_loader = torch.utils.data.DataLoader(
        retain_loader.dataset, batch_size=getattr(args, 'batch_size', 256), shuffle=False
    )
    retain_single_loader = torch.utils.data.DataLoader(
        retain_loader.dataset, batch_size=1, shuffle=False
    )
    forget_batch_loader = torch.utils.data.DataLoader(
        forget_loader.dataset, batch_size=getattr(args, 'batch_size', 256), shuffle=False
    )
    
    # Create unlearning args
    unlearn_args = create_unlearning_args(args)
    
    # Prepare gradient storage
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    forget_grad = torch.zeros_like(torch.cat(params)).to(device)
    retain_grad = torch.zeros_like(torch.cat(params)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    # Compute forget gradients
    print("Computing forget gradients...")
    total_forget = 0
    for i, (data, label) in enumerate(tqdm(forget_batch_loader)):
        model.zero_grad()
        real_num = data.shape[0]
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        f_grad = sam_grad(model, loss) * real_num
        forget_grad += f_grad
        total_forget += real_num
    
    # Compute retain gradients
    print("Computing retain gradients...")
    total_retain = 0
    for i, (data, label) in enumerate(tqdm(retain_grad_loader)):
        model.zero_grad()
        real_num = data.shape[0]
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        r_grad = sam_grad(model, loss) * real_num
        retain_grad += r_grad
        total_retain += real_num
    
    # Normalize gradients
    retain_grad *= total_forget / ((total_forget + total_retain) * total_retain)
    forget_grad /= total_forget + total_retain
    
    # Apply Woodfisher algorithm
    print("Applying Woodfisher algorithm...")
    perturb = woodfisher(
        model,
        retain_single_loader,
        device=device,
        criterion=criterion,
        v=forget_grad - retain_grad,
    )
    
    # Apply perturbation
    apply_perturb(model, unlearn_args.alpha * perturb)
    
    # Validate
    results = validate_unlearning(model, data_loaders, criterion, device)
    
    return model


def retrain_unlearning(full_model_path, loaders, args):
    """Retrain from scratch unlearning"""
    print("Starting Retrain unlearning")
    
    # Create fresh model architecture (don't load weights)
    model = get_resnet_model(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Convert loaders
    data_loaders = convert_loaders_for_unlearning(loaders)
    
    # Create unlearning args with more epochs for retraining
    unlearn_args = create_unlearning_args(args, unlearn_epochs=50, unlearn_lr=0.1)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        unlearn_args.unlearn_lr,
        momentum=unlearn_args.momentum,
        weight_decay=unlearn_args.weight_decay,
    )
    
    decreasing_lr = list(map(int, unlearn_args.decreasing_lr.split(",")))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Train on retain data
    print("Retraining on retain data...")
    for epoch in range(unlearn_args.unlearn_epochs):
        start_time = time.time()
        print(f"Epoch #{epoch}, Learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")
        
        train_acc = ft_iter(data_loaders, model, criterion, optimizer, epoch, unlearn_args)
        scheduler.step()
        
        print(f"One epoch duration: {time.time() - start_time:.2f}s")
    
    # Validate
    results = validate_unlearning(model, data_loaders, criterion, device)
    
    return model