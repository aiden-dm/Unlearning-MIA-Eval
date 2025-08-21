from dataclasses import dataclass

@dataclass
class SCRUBInput:
    dataset: str
    kd_T: float
    learning_rate: float
    msteps: int
    epochs: int
    t_opt_gamma: float
    t_opt_alpha: float
    model_path: str
    check_path: str
    print_accuracies: bool

@dataclass
class BadTeachInput:
    dataset: str
    KL_temperature: float
    learning_rate: float
    batch_size: int
    num_workers: int
    device: str
    epochs: int
    print_accuracies: bool
    model_path: str
    check_path: str

@dataclass
class SSDInput:
    dataset: str
    dampening_constant: float
    selection_weighting: float
    learning_rate: float
    device: str
    model_path: str
    check_path: str