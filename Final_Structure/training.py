# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# Our local imports
from datasets import get_loaders

# Setting device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained ResNet model and modify for CIFAR-10 (10 classes)
def get_resnet_model():
    model = models.resnet18(weights=None)           # No pre-trained weights
    model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust output layer for 10 classes
    return model.to(DEVICE)

def train(model, train_loader, criterion, optimizer, epochs=10, save_path="resnet_cifar.pth"):
    # Putting model in training mode
    model.train()

    # Starting training process
    for epoch in range(epochs):
        # Initializing current epoch variables
        running_loss = 0.0
        correct, total = 0, 0

        print(f'Starting Epoch [{epoch+1}/{epochs}]...')
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Updating gradients, loss and optimizer
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Updating loss and tally information
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {correct/total:.4f}")
        
    # Save checkpoint
    torch.save(model.state_dict(), save_path)


def train_resnet(root='./data', 
                 dataset='cifar10', 
                 forget_classes=[],
                 batch_size = 32):
    
    # Get ResNet model
    model = get_resnet_model()
    
    [
        train_loader,
        _,
        _,
        _,
        train_retain_loader,
        _,
        _
    ] = get_loaders(root, forget_classes, batch_size)

    # Define some training variables
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 1
    full_path = "./checkpoints/resnet_full.pt"
    retain_path = "./checkpoints/resnet_retain.pt"

    # Train on the entire train dataset
    train(model, train_loader, criterion, optimizer, epochs, full_path)

    # Train on the retain set
    train(model, train_retain_loader, criterion, optimizer, epochs, retain_path)

def load_model(checkpoint_path="resnet_cifar.pt"):
    model = get_resnet_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    return model