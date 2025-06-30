import numpy as np

from Final_Structure.algo_args import sample_args_ns
from Final_Structure.evaluate import evaluate_model, membership_inference_attack
from Final_Structure.unlearning_registry import get_unlearning_function

def calculate_loss(
    model,
    alpha,
    beta,
    gamma,
    train_retain_loader, 
    train_forget_loader,
    valid_forget_loader,
    mia_seed
):

  # Calculate evaluation metrics
  retain_metrics = evaluate_model(model, train_retain_loader, 'cuda')
  forget_metrics = evaluate_model(model, train_forget_loader, 'cuda')
  mia_mean_metrics, _ = membership_inference_attack(
      model,
      valid_forget_loader,
      train_forget_loader,
      device='cuda',
      seed=mia_seed,
      plot_dist=False
  )

  # Calculate the loss function
  rps = np.mean([
      retain_metrics["accuracy"],
      retain_metrics["precision"],
      retain_metrics["recall"],
      retain_metrics["f1"]
  ])
  fps = np.mean([
      forget_metrics["accuracy"],
      forget_metrics["precision"],
      forget_metrics["recall"],
      forget_metrics["f1"]
  ])
  Lr = 1 - rps
  Lf = fps
  Lp = 2 * np.abs(mia_mean_metrics["ACC"] - 0.5)
  loss = alpha * Lr + beta * Lf + gamma * Lp

  return loss


def tune_unlearning_method(
      method_name,
      dataset,
      loaders,
      trials=1,
      alpha=0.25,
      beta=0.25,
      gamma=0.25,
      mia_seed=1
):

    best_loss = float('inf')
    best_config = None

    train_retain_loader = loaders['train_retain_loader']
    train_forget_loader = loaders['train_forget_loader']
    valid_forget_loader = loaders['valid_forget_loader']

    for trial in range(trials):

        # Sample parameter and add necessary fixed parameters
        args = sample_args_ns(method_name)
        args.save_checkpoint = False
        args.check_path = None
        args.print_accuracies = False
        args.dataset = dataset

        print(f"Trial {trial+1}/{trials} | Params: {args}")

        # Perform unlearning
        full_path = "/content/drive/MyDrive/resnet_full.pt"
        unl_func = get_unlearning_function(method_name)
        unl_model, _ = unl_func(full_path, loaders, args)

        # Calculate loss
        loss = calculate_loss(
            unl_model,
            alpha,
            beta,
            gamma,
            train_retain_loader,
            train_forget_loader,
            valid_forget_loader,
            mia_seed=mia_seed
        )

        # Save best set of hyperparameters
        if loss < best_loss:
            best_loss = loss
            best_config = args
    
    return best_config, best_loss