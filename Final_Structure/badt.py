# Imports
import sys
import torch

# Adding necessary paths to the system path
sys.path.append('/content/Unlearning-MIA-Eval')

# Framework imports
from Final_Structure.training import load_model, get_resnet_model

# Third party code imports
from Third_Party_Code.BadTeach.unlearn import blindspot_unlearner

def badt(full_model_path, loaders, args):
    
    # Unpacking the data loaders
    train_forget_loader = loaders['train_forget_loader']
    train_retain_loader = loaders['train_retain_loader']
    valid_forget_loader = loaders['valid_forget_loader']
    valid_retain_loader = loaders['valid_retain_loader']

    # Create ssd_loaders list for training validation
    badt_loaders = [
        train_retain_loader,
        train_forget_loader,
        valid_retain_loader,
        valid_forget_loader
    ]

    model = load_model(dataset=args.dataset, checkpoint_path=full_model_path)
    unlearning_teacher = get_resnet_model(dataset=args.dataset)
    student_model = load_model(dataset=args.dataset, checkpoint_path=full_model_path)
    
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
    if args.save_checkpoint:
        torch.save(student_model.state_dict(), args.check_path)

    return student_model, history