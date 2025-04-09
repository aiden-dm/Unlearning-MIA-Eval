# Imports
import sys
import copy
import torch

# Adding necessary paths to the system path
sys.path.append('/content/Unlearning-MIA-Eval')

# Imports from local code
from Final_Structure.training import load_model, get_resnet_model

# Imports from the BadTeach GitHub repository
from Third_Party_Code.BadTeach.unlearn import blindspot_unlearner

def badt(full_model_path, loaders, args):
    # Unpacking the data loaders
    train_forget_loader = loaders[3]
    train_retain_loader = loaders[4]
    valid_forget_loader = loaders[5]
    valid_retain_loader = loaders[6]

    # Create ssd_loaders list for training validation
    badt_loaders = [
        train_retain_loader,
        train_forget_loader,
        valid_retain_loader,
        valid_forget_loader
    ]

    model = load_model(checkpoint_path=full_model_path)
    unlearning_teacher = get_resnet_model()
    student_model = load_model(checkpoint_path=full_model_path)
    
    KL_temperature = args.KL_temperature
    optimizer = torch.optim.Adam(student_model.parameters(), lr = args.learning_rate)
    batch_size = args.batch_size
    num_workers = 8
    device = "cuda"

    history = blindspot_unlearner(model = student_model, 
                        unlearning_teacher = unlearning_teacher, 
                        full_trained_teacher = model, 
                        retain_data = train_retain_loader.dataset, 
                        forget_data = train_forget_loader.dataset, 
                        loaders = badt_loaders,
                        epochs = args.epochs, optimizer = optimizer, 
                        lr = args.learning_rate, 
                        batch_size = batch_size, 
                        num_workers =  num_workers, 
                        device = device, 
                        KL_temperature = KL_temperature,
                        print_accuracies = args.print_accuracies)

    # Save a copy of the student model as a checkpoint
    save_path = "/content/Unlearning-MIA-Eval/Final_Structure/checkpoints/badt_applied.pt"
    torch.save(student_model.state_dict(), save_path)

    return student_model, history