# Unlearning Algorithm Benchmarking Framework
This repository contains the code of {INSERT-FRAMEWORK-NAME}, a framework to benchmark and compare unlearning algorithms performing complete single- or multi-class unlearning. This README provides general information about the framework that hopefully is useful to someone first becoming familiar with it.

## Implemented Unlearning Algorithms
Our algorithm implements the following unlearning algorithms.

* SCRUB
* BadTeach
* SSD
* Remainder of this list to be completed by Bhavya

## Membership Inference Attacks

Membership Inference Attacks (MIAs) are a critical component in evaluating and comparing unlearning algorithms, as they provide a concrete measure of how effectively a model has forgotten specific data. Conceptually, they offer insight into the "quality" of unlearning by assessing residual memorization. We implement the following MIAs in our framework:

* Yeom --> we have already implemented this one
* Calibration
* LiRA
* Reference
* RMIA

## Structure of An Unlearning Algorithm Implementation

Each unlearning algorithm is implemented in its own Python file and can be executed by calling the algorithm's function contained within. Each unlearning algorithm function call has the same structure:

`unlearning_function_name(full_model_path, loaders, args)`, where:
* `full_model_path`: the file path to the PyTorch checkpoint of the model on which you wish to execute the unlearning algorithm. In the context of an unlearning experiment, this is usually a common model (ex. ResNet18) trained on the full training set of your experiment.
* `loaders`: a dictionary of loaders that can be fetched using the `get_loaders()` function within `datasets.py`.
* `args`: a Python Simple Namespace that contains all of the arguments/hyperparameters necessary to run the unlearning algorithm. There are arguments that are shared by many or all of the unlearning algorithms, but there are also many unique arguments to a given algorithm. A good place to view which arguments are applied to each algorithm is `algo_args.py` which provides the argument spaces for each argument for each unleanring algorithm, which is drawn from in our framework's automatic hyperparameter tuning process.

## Structure of A Membership Inference Attack Implementation

This is to be determined, have not yet gotten a chance to standardize this.
