import random

from types import SimpleNamespace

scrub_argspace = {
    "epochs": [10, 15, 20],
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "msteps": [5, 10, 15],
    "t_opt_alpha": [0.0],
    "t_opt_gamma": [1.0],
    "kd_T": [1, 2, 5]
}

badt_argspace = {
    "epochs": [5, 7, 10],
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "KL_temperature": [2, 4, 8],
    "batch_size": [128, 256, 512]
}

ssd_argspace = {
    "learning_rate": [0.001],               # this is arbitrary doesn't impact unlearning
    "dampening_constant": [1, 5, 10, 20],
    "selection_weighting": [0.5, 1, 2]
}

argspace_registry = {
    "SCRUB": scrub_argspace,
    "BadTeach": badt_argspace,
    "SSD": ssd_argspace
}

def sample_args(method_name):
    if method_name not in argspace_registry:
        raise ValueError(f"ERROR: Method '{method_name}' not found in argspace_registry.")
    space = argspace_registry[method_name]
    return {k: random.choice(v) for k, v, in space.items()}

def sample_args_ns(method_name):
    params = sample_args(method_name)
    return SimpleNamespace(**params)