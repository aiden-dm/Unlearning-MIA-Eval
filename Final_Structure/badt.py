# Imports
import sys
import os
import copy
import torch

# Adding the SCRUB repo to the system path
BADT_PATH = os.path.abspath("../../Third_Party_Code/BadTeach")
if BADT_PATH not in sys.path:
    sys.path.append(BADT_PATH)

# Import from our local files
from training import load_model, get_loaders

# Imports from the BadTeach GitHub repository
from Third_Party_Code.BadTeach.unlearn import blindspot_unlearner

def badt(model, loaders):
    # Unpacking the data loaders
    [train_loader, 
     valid_loader, 
     test_loader, 
     train_forget_loader, 
     train_retain_loader, 
     valid_forget_loader,
     valid_retain_loader, 
     test_forget_loader, 
     test_retain_loader
    ] = loaders

    unlearning_teacher = copy.deepcopy(model)
    student_model = copy.deepcopy(model)
    model = model.eval()
    
    KL_temperature = 1
    optimizer = torch.optim.Adam(student_model.parameters(), lr = 0.0001)
    batch_size = 32
    num_workers = 8
    device = "cuda"

    blindspot_unlearner(model = student_model, unlearning_teacher = unlearning_teacher, full_trained_teacher = model, 
          retain_data = train_retain_loader.dataset, forget_data = train_forget_loader.dataset, epochs = 15, optimizer = optimizer, lr = 0.0001, 
          batch_size = batch_size, num_workers =  num_workers, device = device, KL_temperature = KL_temperature)

    return student_model

model = load_model("./checkpoints/resnet_full.pt")
loaders = get_loaders(root='./data', forget_classes=[1])
badt(model, loaders)