# Imports
import sys
import os
import copy
import torch

# Adding the SCRUB repo to the system path
SSD_PATH = os.path.abspath("../../Third_Party_Code/SSD")
if SSD_PATH not in sys.path:
    sys.path.append(SSD_PATH)

# Import from our local files
from training import load_model, get_loaders

# Imports from the BadTeach GitHub repository
from Third_Party_Code.SSD.src.unlearn import blindspot_unlearner

def ssd(loaders):
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

    model = load_model("./checkpoints/resnet_full.pt")
    unlearning_teacher = load_model("./checkpoints/resnet_retain.pt")
    full_trained_teacher = load_model("./checkpoints/resnet_full.pt")

    unl_model = blindspot_unlearner(model=model, 
                                    unlearning_teacher=unlearning_teacher, 
                                    full_trained_teacher=full_trained_teacher, 
                                    retain_data=train_retain_loader.dataset, 
                                    forget_data=train_forget_loader.dataset)

loaders = get_loaders(root='./data', forget_classes=[1])
ssd(loaders)