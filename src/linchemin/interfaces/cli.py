import argparse
from dataclasses import dataclass, field
from typing import List

from linchemin.interfaces.workflows import get_workflow_options, process_routes


class KeyValueAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = getattr(namespace, self.dest) or {}
        for value in values:
            try:
                key, value = value.split("=")
                my_dict[key] = value
            except ValueError:
                raise argparse.ArgumentError(
                    self, f"Could not parse argument {value} as key=value pair."
                )
        setattr(namespace, self.dest, my_dict)


@dataclass
class Argument:
    type: str = field(repr=False)
    required: bool
    help: str
    dest: str
    name_or_flags: List[str] = field(default_factory=list)
    action: str = field(default="store")
    default: str = field(default=None)
    choices: List = field(default_factory=list)
    nargs: str = field(default="?")


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    helper_data = get_workflow_options()
    arguments = [Argument(**v) for k, v in helper_data.items()]

    for argument in arguments:
        kwargs = {
            "action": KeyValueAction if argument.type == dict else argument.action,
            "type": str,
            "nargs": "*" if argument.type in (list, dict) else argument.nargs,
            "default": argument.default,
            "choices": argument.choices if argument.choices else None,
            "help": argument.help,
            "dest": argument.dest,
            "required": argument.required,
        }
        parser.add_argument(*argument.name_or_flags, **kwargs)

    return parser


def parse_arguments(argv=None):
    parser = create_parser()
    return parser.parse_args(argv)


def linchemin_cli(argv=None):
    print("START: LinChemIn")
    parsed_args = parse_arguments(argv)
    process_routes(**vars(parsed_args))
    print("END: LinChemIn")


if __name__ == "__main__":
    linchemin_cli()
