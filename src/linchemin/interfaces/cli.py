import argparse
import pprint
import sys
from dataclasses import dataclass, field
from typing import Any, List, Union

from linchemin.interfaces.workflows import get_workflow_options, process_routes


class keyvalue(argparse.Action):
    # Constructor calling
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())

        for value in values:
            # split it into key and value
            key, value = value.split("=")
            # assign into dictionary
            getattr(namespace, self.dest)[key] = value


@dataclass
class Argument:
    """
    Class for storing values to pass to the add_argument() method of the  argparse.ArgumentParser() class.
    https://docs.python.org/3/library/argparse.html#the-add-argument-method
    """

    type: str = field(repr=False)
    required: bool
    help: str
    # metavar:
    dest: str
    name_or_flags: list[str] = field(default_factory=list)
    action: str = field(default="store")
    default: str = field(default=None)
    choices: list = field(default_factory=list)
    nargs: str = field(default="?")
    # const:


def wrap_facade(parser):
    helper_data = get_workflow_options()
    arguments = [Argument(**v) for k, v in helper_data.items()]

    for argument in arguments:
        if argument.type == list:
            argument.type = str
            argument.nargs = "*"
        elif argument.type == dict:
            argument.action = keyvalue
            argument.type = str
            argument.nargs = "*"

        parser.add_argument(
            *argument.name_or_flags,
            default=argument.default,
            choices=argument.choices,
            help=argument.help,
            dest=argument.dest,
            required=argument.required,
            action=argument.action,
            nargs=argument.nargs,
        )

    return parser


def linchemin_cli(argv=None):
    print("START: LinChemIn")
    # 1) create the overall parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # 2) add a parser argument for each facade we want to expose
    parser = wrap_facade(parser)

    parsed = parser.parse_args()
    # 4) pass the arguments as a dictionary to the function

    output = process_routes(**vars(parsed))

    print("END: LinChemIn")


if __name__ == "__main__":
    print("xx")
    linchemin_cli()
    # path = "../../../tests/cgu/data/ibmrxn_retro.json"
    # output, metadata = facade('read_and_convert', path)
    # helper = facade_helper('read_and_convert')
    # print(helper)
