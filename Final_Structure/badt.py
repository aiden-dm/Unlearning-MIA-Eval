# Imports
import sys
import copy
import torch

# Adding necessary paths to the system path
sys.path.append('/content/Unlearning-MIA-Eval')

# Import from our local files
from Final_Structure.training import load_model, get_loaders

# Imports from the BadTeach GitHub repository
from Third_Party_Code.BadTeach.unlearn import blindspot_unlearner

def badt(model, loaders):
    # Unpacking the data loaders
    train_forget_loader = loaders[3]
    train_retain_loader = loaders[4]

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

    # Save a copy of the student model as a checkpoint
    save_path = "/content/Unlearning-MIA-Eval/Final_Structure/checkpoints/badt_applied.pt"
    torch.save(student_model.state_dict(), save_path)

model = load_model("/content/Unlearning-MIA-Eval/Final_Structure/checkpoints/resnet_full.pt")
loaders = get_loaders(root='/content/Unlearning-MIA-Eval/Final_Structure/data', forget_classes=[1])
badt(model, loaders)