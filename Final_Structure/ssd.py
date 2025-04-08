# Imports
import sys
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
    valid_forget_loader = loaders[5]
    valid_retain_loader = loaders[6]

    # Create ssd_loaders list for training validation
    ssd_loaders = [
        train_retain_loader,
        train_forget_loader,
        valid_retain_loader,
        valid_forget_loader
    ]

    unlearning_teacher = copy.deepcopy(model)
    full_trained_teacher = copy.deepcopy(model)
    model = model.eval()

    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    KL_temperature = args.KL_temperature

    unl_model, history = blindspot_unlearner(model=model, 
                                    unlearning_teacher=unlearning_teacher, 
                                    full_trained_teacher=full_trained_teacher, 
                                    retain_data=train_retain_loader.dataset, 
                                    forget_data=train_forget_loader.dataset,
                                    loaders=ssd_loaders,
                                    epochs=epochs,
                                    lr=lr,
                                    batch_size=batch_size,
                                    KL_temperature=KL_temperature,
                                    print_accuracies = args.print_accuracies)
    
    # Save a copy of the student model as a checkpoint
    save_path = "/content/Unlearning-MIA-Eval/Final_Structure/checkpoints/ssd_applied.pt"
    torch.save(unl_model.state_dict(), save_path)

    return unl_model, history