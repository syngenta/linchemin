from abc import ABC, abstractmethod
from typing import Union

from linchemin.cgu.syngraph import BipartiteSynGraph, MonopartiteReacSynGraph
from linchemin.cgu.syngraph_operations import find_path
from linchemin.cheminfo.models import ChemicalEquation, Molecule
from linchemin.rem.route_descriptors import descriptor_calculator
from linchemin.utilities import console_logger

"""Module containing functions and classes for computing
route metrics that depend on external information"""
logger = console_logger(__name__)


class DistanceStrategy(ABC):
    """Abstract strategy for distance terms calculation"""

    @abstractmethod
    def distance(
        self, route: BipartiteSynGraph, node: Molecule, root: Molecule
    ) -> Union[int, float]:
        pass


class SimpleDistanceStrategy(DistanceStrategy):
    """
    Concrete distance strategy that simply returns
    the distance between the considered node and the root.
    The distance is calculated considering only
    ChemicalEquation nodes.
    """

    def distance(self, route: BipartiteSynGraph, node: Molecule, root: Molecule) -> int:
        path = find_path(route, node, root)
        return len([n for n in path if isinstance(n, ChemicalEquation)])


class LongestLinearSequenceWeightedDistanceStrategy(DistanceStrategy):
    """
    Concrete distance strategy that is equal to 1 when the distance of the considered node from the root is 1 and
    that is equal to 0 when the distance is the length of the longest linear sequence+1.
    The +1 guarantees that the leaf nodes still have a weight in the term (otherwise they would always have 0 impact)
    """  # noqa: E501

    def distance(self, route: BipartiteSynGraph, node: Molecule, root: Molecule):
        lls = descriptor_calculator(route, "longest_seq") + 1.0
        d = len(
            [n for n in find_path(route, node, root) if isinstance(n, ChemicalEquation)]
        )
        return round(-1.0 / lls * d + 1, 3)


# Context class
class DistanceContext:
    _distance_strategies = {
        "simple": SimpleDistanceStrategy(),
        "longest_sequence": LongestLinearSequenceWeightedDistanceStrategy(),
    }

    def __init__(self):
        self.strategy = None
        self.distance = None

    def set_strategy(self, strategy):
        if strategy not in self._distance_strategies:
            logger.error("Distance strategy not defined")
            raise KeyError
        self.strategy = self._distance_strategies[strategy]

    def calculate_distance(self, route, node, root):
        self.distance = self.strategy.distance(route, node, root)

    def get_distance(self):
        return self.distance


def distance_function_calculator(
    strategy_name: str,
    route: Union[BipartiteSynGraph, MonopartiteReacSynGraph],
    node: Union[Molecule, ChemicalEquation],
    root: Union[Molecule, ChemicalEquation],
) -> Union[float, int]:
    """
    To compute the term related to the distance to the target of a node in a route

    Parameters:
    -------------
    strategy_name: str
        The name of the distance function to be used
    route: Union[BipartiteSynGraph, MonopartiteReacSynGraph]
        The route to be considered
    node: Union[Molecule, ChemicalEquation]
        The node for which the distance should be computed
    root: Union[Molecule, ChemicalEquation]
        The root of the route

    Returns:
    ---------
    distance value: Union[float, int]
        The value of the selected distance function

    """  # noqa: E501
    context = DistanceContext()
    context.set_strategy(strategy_name)
    context.calculate_distance(route, node, root)
    return context.get_distance()
