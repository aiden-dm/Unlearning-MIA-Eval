# Imports
import sys
import torch

# Adding necessary paths to the system path
sys.path.append('/content/Unlearning-MIA-Eval')

# Framework imports
from Final_Structure.evaluate import train_validation
from Final_Structure.training import load_model
from Final_Structure.unlearning_algos.unlearning_inputs import SSDInput

# Third party code imports
import Third_Party_Code.SSD.src.ssd as ssd_file

def ssd(loaders, args: SSDInput):
    
    # Unpacking the data loaders
    train_loader = loaders['train_loader']
    train_forget_loader = loaders['train_forget_loader']
    train_retain_loader = loaders['train_retain_loader']
    valid_forget_loader = loaders['valid_forget_loader']
    valid_retain_loader = loaders['valid_retain_loader']

    parameters = {
        "lower_bound": 1,        # unused
        "exponent": 1,           # unused
        "magnitude_diff": None,  # unused
        "min_layer": -1,         # -1: all layers are available for modification
        "max_layer": -1,         # -1: all layers are available for modification
        "forget_threshold": 1,   # unused
        "dampening_constant": args.dampening_constant,   # Lambda from paper
        "selection_weighting": args.selection_weighting, # Alpha from paper
    }

    # Loading the fully trained ResNet model
    model = load_model(dataset=args.dataset, checkpoint_path=args.model_path)

    # load the trained model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    pdr = ssd_file.ParameterPerturber(model, optimizer, args.device, parameters)

    model = model.eval()

    # Calculation of the forget set importances
    sample_importances = pdr.calc_importance(train_forget_loader)

    # Calculate the importances of D (see paper); this can also be done at any point before forgetting.
    original_importances = pdr.calc_importance(train_loader)

    # Dampen selected parameters
    pdr.modify_weight(original_importances, sample_importances)

    # Validate results
    acc_dict = train_validation(model, 
                                train_retain_loader, 
                                train_forget_loader, 
                                valid_retain_loader, 
                                valid_forget_loader)
    
    # Save model checkpoint
    if args.check_path is not None:
        torch.save(model.state_dict(), args.check_path)
    
    return model, acc_dict




    