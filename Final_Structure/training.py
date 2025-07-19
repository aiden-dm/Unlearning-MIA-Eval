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

def train(model, train_loader, criterion, optimizer, epochs=10, scheduler=None, save_path="resnet_cifar.pth"):
    # Make sure model is on the GPU
    model.to(DEVICE)
    
    # Putting model in training mode
    model.train()

    # Starting training process
    for epoch in tqdm(range(epochs)):
        # Initializing current epoch variables
        running_loss = 0.0
        correct, total = 0, 0

        print(f'Starting Epoch [{epoch+1}/{epochs}]...')
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        if scheduler:
            scheduler.step(epoch_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
    # Save checkpoint
    torch.save(model.state_dict(), save_path)

def load_model(dataset, checkpoint_path="resnet_cifar.pt"):
    model = get_resnet_model(dataset)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    return model