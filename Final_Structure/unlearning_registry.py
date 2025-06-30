
from Final_Structure.scrub import scrub
from Final_Structure.badt import badt
from Final_Structure.ssd import ssd

unlearning_fn_registry = {
    "SCRUB": scrub,
    "SSD": ssd,
    "BadTeach": badt
}

def list_available_methods():
    return list(unlearning_fn_registry.keys())

def get_unlearning_function(method_name):
    if method_name not in unlearning_fn_registry:
        raise ValueError(f"ERROR: Unknown learning method: {method_name}. "
                         f"Available methods: {list_available_methods()}")
    return unlearning_fn_registry[method_name]