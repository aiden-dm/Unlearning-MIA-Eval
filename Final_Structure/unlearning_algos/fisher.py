import torch
from torch.autograd import grad
import torch.nn as nn
from tqdm import tqdm

from Final_Structure.unlearning_algos.unlearning_inputs import FisherInput
from Final_Structure.training import load_model
from Final_Structure.unlearning_algos.unlearning_utils import (
    validate_unlearning
)

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


def fisher(loaders, args: FisherInput):
    print("Starting Fisher Information unlearning")
    
    # Load model
    model = load_model(dataset=args.dataset, checkpoint_path=args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Convert loaders
    retain_loader = loaders["train_retain_loader"]
    
    # Calculate Fisher Information Matrix
    print("Calculating Fisher Information Matrix...")
    fisher_approximation = fisher_information_matrix(model, retain_loader, device)
    
    # Apply Fisher noise
    print("Applying Fisher noise...")
    for i, parameter in enumerate(model.parameters()):
        noise = torch.sqrt(args.alpha / fisher_approximation[i]).clamp(
            max=1e-3
        ) * torch.empty_like(parameter).normal_(0, 1)
        
        # Special handling for last layer
        if parameter.shape[-1] == 10:  # Assuming 10 classes for CIFAR-10
            noise = noise * 10
        
        parameter.data = parameter.data + noise

    if args.check_path is not None:
        torch.save(model.state_dict(), args.check_path)

    return model