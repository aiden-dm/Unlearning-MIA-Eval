# Imports
import sys
import os
import copy
import torch

# Adding necessary paths to the system path
sys.path.append('/content/Unlearning-MIA-Eval')

# Import from our local files
from Final_Structure.training import load_model

# Imports from the BadTeach GitHub repository
from Third_Party_Code.SSD.src.unlearn import blindspot_unlearner

def ssd(model, loaders, args):
    # Unpacking the data loaders
    train_forget_loader = loaders[3]
    train_retain_loader = loaders[4]

    model = load_model("/content/Unlearning-MIA-Eval/Final_Structure/checkpoints/resnet_full.pt")
    unlearning_teacher = load_model("/content/Unlearning-MIA-Eval/Final_Structure/checkpoints/resnet_retain.pt")
    full_trained_teacher = load_model("/content/Unlearning-MIA-Eval/Final_Structure/checkpoints/resnet_full.pt")

    epochs = args.epochs,
    lr = args.learning_rate,
    batch_size = args.batch_size,
    KL_temperature = args.KL_temperature

    unl_model = blindspot_unlearner(model=model, 
                                    unlearning_teacher=unlearning_teacher, 
                                    full_trained_teacher=full_trained_teacher, 
                                    retain_data=train_retain_loader.dataset, 
                                    forget_data=train_forget_loader.dataset,
                                    epochs=epochs,
                                    lr=lr,
                                    batch_size=batch_size,
                                    KL_temperature=KL_temperature)
    
    # Save a copy of the student model as a checkpoint
    save_path = "/content/Unlearning-MIA-Eval/Final_Structure/checkpoints/ssd_applied.pt"
    torch.save(unl_model.state_dict(), save_path)

    return unl_model