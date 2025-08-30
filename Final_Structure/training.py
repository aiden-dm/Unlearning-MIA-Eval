# Imports
import torch
import torch.nn as nn
import torchvision.models as models
import sys
from tqdm import tqdm

# Adding the local files to the system path
sys.path.append('/content/Unlearning-MIA-Eval/Final_Structure')

# Setting device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained ResNet model
def get_resnet_model(dataset):
    
    # Get the output layer size depending on dataset
    out_layer_size = 0
    if dataset == "cifar10":
        out_layer_size = 10
    elif dataset == "cifar100":
        out_layer_size = 100

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, out_layer_size)
    return model.to(DEVICE)

def train(
    model, 
    train_loader, 
    valid_loader, 
    criterion, 
    optimizer, 
    epochs=10, 
    scheduler=None, 
    save_path="resnet_cifar.pth"
):
    
    model.to(DEVICE)
    
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                valid_correct_predictions = 0
                total_samples = 0
                for _, data in enumerate(valid_loader):
                    images, labels = data
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    valid_correct_predictions += (predicted == labels).sum().item()
                    total_samples += labels.size(0)
                valid_accuracy = valid_correct_predictions / total_samples

                train_correct_predictions = 0
                total_samples = 0
                for _, data in enumerate(train_loader):
                    images, labels = data
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    train_correct_predictions += (predicted == labels).sum().item()
                    total_samples += labels.size(0)
                train_accuracy = train_correct_predictions / total_samples
                print(f"Epoch {epoch}, train_acc: {train_accuracy * 100:.2f}%, valid_acc: {valid_accuracy * 100:.2f}%, Loss: {train_loss:.4f}, lr: {scheduler.get_last_lr()[0]:.4f}")
    
    # Save checkpoint
    torch.save(model.state_dict(), save_path)

def load_model(dataset, checkpoint_path="resnet_cifar.pt"):
    model = get_resnet_model(dataset)
    
    # Load checkpoint with compatibility for pruned models
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Handle pruned models (convert weight_orig + weight_mask to weight)
    cleaned_state_dict = convert_pruned_state_dict(state_dict)
    
    model.load_state_dict(cleaned_state_dict)
    return model


def convert_pruned_state_dict(state_dict):
    """Convert pruned model state dict to regular format"""
    cleaned_state_dict = {}
    
    # Check if this is a pruned model
    has_pruned_weights = any('weight_orig' in key for key in state_dict.keys())
    
    if has_pruned_weights:
        print("Detected pruned model, converting weights...")
        
        # Convert weight_orig + weight_mask to weight
        for key in state_dict.keys():
            if key.endswith('weight_orig'):
                # Get the base name (remove _orig suffix)
                base_name = key[:-5]  # Remove '_orig'
                mask_name = base_name + '_mask'
                
                if mask_name in state_dict:
                    # Compute actual weight: weight_orig * weight_mask
                    actual_weight = state_dict[key] * state_dict[mask_name]
                    cleaned_state_dict[base_name] = actual_weight
                    print(f"Converted {key} + {mask_name} -> {base_name}")
                else:
                    # If no mask, just use original weight
                    cleaned_state_dict[base_name] = state_dict[key]
            elif not key.endswith('weight_mask'):
                # Copy non-weight parameters as-is
                cleaned_state_dict[key] = state_dict[key]
    else:
        # Not a pruned model, return as-is
        cleaned_state_dict = state_dict
    
    return cleaned_state_dict