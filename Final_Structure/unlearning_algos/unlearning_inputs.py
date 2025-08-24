from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim

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

@dataclass
class FineTuneInput:
    dataset: str
    epoch: int
    with_l1: bool
    unlearn_epochs: int
    no_l1_epochs: int
    alpha: float
    learning_rate: float
    print_freq: int
    model_path: str
    check_path: str

@dataclass
class GradientAscentInput:
    dataset: str
    epoch: int
    with_l1: bool
    alpha: float
    learning_rate: float
    print_freq: int
    model_path: str
    check_path: str

@dataclass
class FisherInput:
    dataset: str
    alpha: float
    model_path: str
    check_path: str