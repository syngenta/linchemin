import hashlib
import itertools
import operator
from dataclasses import dataclass, field
from typing import List


def list_of_dict_groupby(data_input: list[dict], keys: list):
    """function to group a list of dictionaries by the value of one or more keys"""
    input_data_sorted = sorted(data_input, key=operator.itemgetter(*keys))
    return {
        i: list(g)
        for i, g in itertools.groupby(input_data_sorted, key=operator.itemgetter(*keys))
    }


@dataclass
class OutcomeMetadata:
    """Class for storing information about a stage of a process."""

    name: str
    is_successful: bool
    log: dict = field(init=True, compare=True, default_factory={})


def create_hash(text: str) -> int:
    """replaces the built in python hash function with one that is deterministic
    it returns a 128 bit integer
    """
    text = text.encode("utf8")
    digest = hashlib.md5(text).hexdigest()
    return int(digest, 16)


def camel_to_snake(camel_case_string: str) -> str:
    """transforms a CamelCase string into a snake_case string"""
    res = [camel_case_string[0].lower()]
    for c in camel_case_string[1:]:
        if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            res.append("_")
            res.append(c.lower())
        else:
            res.append(c)
    return "".join(res)
