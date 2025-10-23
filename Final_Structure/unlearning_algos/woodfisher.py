import torch
import torch.nn as nn
from torch.autograd import grad
from tqdm import tqdm

from Final_Structure.unlearning_algos.unlearning_inputs import WoodFisherInput
from Final_Structure.training import load_model

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


def woodfisher_alg(model, train_dl, device, criterion, v):
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


def woodfisher(loaders, args: WoodFisherInput):
    print("Starting Woodfisher unlearning")
    
    # Load model
    model = load_model(dataset=args.dataset, checkpoint_path=args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Convert loaders
    retain_loader = loaders["train_retain_loader"]
    forget_loader = loaders["train_forget_loader"]
    
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
    perturb = woodfisher_alg(
        model,
        retain_single_loader,
        device=device,
        criterion=criterion,
        v=forget_grad - retain_grad,
    )
    
    # Apply perturbation
    apply_perturb(model, args.alpha * perturb)

    if args.check_path is not None:
        torch.save(model.state_dict(), args.check_path)
    
    return model