from linchemin import settings
from linchemin.rem.graph_distance import (
    get_available_ged_algorithms,
    get_ged_parameters,
)


def get_parallelization_dict():
    return {
        "parallelization": {
            "name_or_flags": ["-parallelization"],
            "default": settings.FACADE["parallelization"],
            "required": False,
            "type": bool,
            "choices": [True, False],
            "help": "Whether parallelization should be used",
            "dest": "parallelization",
        },
        "n_cpu": {
            "name_or_flags": ["-n_cpu"],
            "default": 'All the available CPUs as given by "multiprocessing.cpu_count()"',
            "required": False,
            "type": int,
            "choices": None,
            "help": "Number of CPUs to be used in parallelization",
            "dest": "n_cpu",
        },
    }


def get_ged_dict():
    return {
        "ged_method": {
            "name_or_flags": ["-ged_method"],
            "default": settings.FACADE["ged_method"],
            "required": False,
            "type": str,
            "choices": get_available_ged_algorithms(),
            "help": "Method to be used to calculate the GED",
            "dest": "ged_method",
        },
        "ged_params": {
            "name_or_flags": ["-ged_params"],
            "default": settings.FACADE["ged_params"],
            "required": False,
            "type": dict,
            "choices": get_ged_parameters(),
            "help": "Parameters of the molecular and/or reaction chemical similarity",
            "dest": "ged_params",
        },
    }
