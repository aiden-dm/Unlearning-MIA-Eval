# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import sys

# Adding the local files to the system path
sys.path.append('/content/Unlearning-MIA-Eval/Final_Structure')

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

# Function that trains ResNet18 on full train and train retain sets
# Used primarily in hyperparameter tuning process
def train_resnet(train_loader, train_retain_loader, args=None):
    
    # Get ResNet model
    model = get_resnet_model()

    # Define some training variables
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    epochs = args.epochs
    full_path = args.full_path
    retain_path = args.retain_path

    # Train on the entire train dataset
    print("Training ResNet18 model on the full dataset...")
    train(model, train_loader, criterion, optimizer, epochs, full_path)

    # Reset model weights and optimizer
    model = get_resnet_model()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train on the retain set
    print("Training the ResNet model on the train retain dataset...")
    train(model, train_retain_loader, criterion, optimizer, epochs, retain_path)

    # Indicate that training is finished
    print('Training Complete!')

def load_model(checkpoint_path="resnet_cifar.pt"):
    model = get_resnet_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    return model