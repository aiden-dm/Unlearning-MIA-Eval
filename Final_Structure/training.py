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
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=False))
    return model