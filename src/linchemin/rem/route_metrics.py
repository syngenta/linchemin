from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Type, Union

import linchemin.cheminfo.functions as cif
from linchemin.cgu.convert import converter
from linchemin.cgu.syngraph import (
    BipartiteSynGraph,
    MonopartiteMolSynGraph,
    MonopartiteReacSynGraph,
)
from linchemin.cgu.syngraph_operations import find_all_paths, find_path
from linchemin.cheminfo.models import ChemicalEquation, Molecule
from linchemin.rem.route_descriptors import descriptor_calculator
from linchemin.utilities import console_logger

"""Module containing functions and classes for computing route metrics that depend on external information"""
logger = console_logger(__name__)


@dataclass
class MetricOutput:
    """Dataclass to store the output and metadata of a route metric calculation"""

    metric_value: float = field(default=0.0)
    """ The value of the route metrics"""
    raw_data: dict = field(default_factory=dict)
    """ Raw data of the route metric calculation (metric non-normalized value, normalization term"""


class RouteMetric(ABC):
    @abstractmethod
    def compute_metric(
        self,
        route: Union[
            BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph
        ],
        external_info: dict,
    ) -> MetricOutput:
        """
        To compute a route metric dependent on external information

        Parameters:
        -----------
        route: Union[MonopartiteReacSynGraph, MonopartiteMolSynGraph, BipartiteSynGraph]
            The input SynGraph for which the metrics should be computed
        external_info: TOBEDEFINED
            The external information necessary for computing the route metric

        Returns:
        --------
        route_metric: float
            The computed route metric
        """
        pass


class MetricsFactory:
    """
    Factory to access the route metrics
    Attributes:
    -------------
    _metrics: a dictionary
        It maps the string representing the names of the available metrics to the correct RouteMetrics subclass
    """

    _metrics = {}

    @classmethod
    def register_metrics(cls, name: str):
        """
        Decorator for registering a metrics.

        Parameters:
        ------------
        name: str
            The name of the metric to be used as a key in the registry

        Returns:
        ---------
        function: The decorator function.
        """

        def decorator(metric_class: Type[RouteMetric]):
            cls._metrics[name.lower()] = metric_class
            return metric_class

        return decorator

    @classmethod
    def list_metrics(cls):
        """
        To list the names of all available metrics.

        Returns:
        ---------
        metrics: list
            The names and information of the available metrics.
        """
        return cls._metrics

    @classmethod
    def get_metric_instance(cls, name: str) -> RouteMetric:
        """
        Get an instance of the specified RouteMetric.

        Parameters:
        ------------
        name: str
            The name of the metric

        Returns:
        ---------
        RouteMetric: An instance of the specified metric.

        Raises:
        --------
        KeyError: If the specified metric is not registered.
        """
        metric = cls._metrics.get(name.lower())
        if metric is None:
            logger.error(f"Metric '{name}' not found")
            raise KeyError
        return metric()


@MetricsFactory.register_metrics("starting_materials_amount")
class StartingMaterialsAmount(RouteMetric):
    """
    Subclass to compute the amount of starting materials needed to produce a certain amount of target
    The external data is expected to have the following format:
    {'target_amount': n (gr),
     'yield': {ChemicalEquation.smiles: yield (0 < y < 1}}
    """

    def compute_metric(
        self,
        route: Union[
            BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph
        ],
        external_info: dict,
    ) -> MetricOutput:
        """To compute the amount of starting materials needed to produce a certain amount of target"""
        route = self.check_route_format(route)
        output = MetricOutput()
        root = route.get_molecule_roots()[0]
        leaves = route.get_molecule_leaves()
        target_amount_mol = self.get_target_amount_in_mol(
            root, external_info["target_amount"]
        )
        all_paths = find_all_paths(route)
        yields = external_info["yield"]
        quantities = {"starting_materials": {}, "intermediates": {}}
        for path in all_paths:
            quantities = self.get_path_quantities(
                path, yields, root, leaves, target_amount_mol, quantities
            )

        output.raw_data["quantities"] = quantities
        output.raw_data["target_amount"] = {
            "grams": external_info["target_amount"],
            "moles": round(target_amount_mol, 3),
        }
        return output

    def get_path_quantities(
        self,
        path: list,
        yields: dict,
        root: Molecule,
        leaves: list,
        target_amount_mol: float,
        quantities: dict,
    ) -> dict:
        """To get the quantities of the reactants in the input path"""
        while path:
            step = path.pop(-1)
            step_yield = yields[step.smiles]
            reactants = step.get_reactants()
            product = step.get_products()[0]
            if product == root:
                prod_amount = target_amount_mol
            else:
                prod_amount = quantities["intermediates"][product.smiles]["moles"]
            for reactant in reactants:
                amount_in_moles, amount_in_grams = self.get_reactant_quantity(
                    reactant, step, step_yield, prod_amount
                )
                smiles = reactant.smiles
                if reactant in leaves:
                    quantities["starting_materials"].update(
                        {
                            smiles: {
                                "moles": round(amount_in_moles, 3),
                                "grams": amount_in_grams,
                            }
                        }
                    )

                else:
                    quantities["intermediates"].update(
                        {
                            smiles: {
                                "moles": round(amount_in_moles, 3),
                                "grams": amount_in_grams,
                            }
                        }
                    )
        return quantities

    @staticmethod
    def get_reactant_quantity(
        reactant: Molecule,
        step: ChemicalEquation,
        step_yield: float,
        prod_amount: float,
    ) -> tuple:
        """To compute the needed amount of the input reactant"""
        reactant_mw = cif.compute_molecular_weigth(reactant.rdmol)
        stoich = step.stoichiometry_coefficients["reactants"][reactant.uid]
        amount_in_moles = prod_amount / stoich / step_yield
        return round(amount_in_moles, 3), round(amount_in_moles * reactant_mw, 3)

    @staticmethod
    def check_route_format(
        route: Union[
            BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph
        ],
    ) -> MonopartiteReacSynGraph:
        """To ensure that the route is in the expected format"""
        if isinstance(route, MonopartiteReacSynGraph):
            return route
        return converter(route, "monopartite_reactions")

    @staticmethod
    def get_target_amount_in_mol(target: Molecule, target_amount_gr: float) -> float:
        """To get the required amount of target in moles"""
        molecular_weight = cif.compute_molecular_weigth(target.rdmol)
        return target_amount_gr / molecular_weight


@MetricsFactory.register_metrics("reactant_availability")
class ReactantAvailability(RouteMetric):
    """
    Subclass to compute the reactant availability metric.
    The external data is expected to have the following format:
    {'starting material smiles': 'string indicating its availability'}
    """

    def compute_metric(
        self,
        route: Union[
            BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph
        ],
        external_info: dict,
    ) -> MetricOutput:
        """To compute the reactant availability metric"""
        route = self.check_route_format(route)
        starting_materials = route.get_leaves()
        data = self.fix_external_info_format(external_info)
        output = MetricOutput()
        distance_function = "simple"
        score = 0
        normalization = 0
        root = route.get_roots()[0]
        for sm in starting_materials:
            distance_term = distance_function_calculator(
                distance_function, route, sm, root
            )
            normalization += distance_term
            score += distance_term * data[sm.smiles]
        output.metric_value = round(score / normalization, 2)
        output.raw_data = {
            "not_normalized_metric": score,
            "normalization_term": normalization,
            "distance_function": distance_function,
        }
        return output

    @staticmethod
    def check_route_format(
        route: Union[
            BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph
        ],
    ) -> BipartiteSynGraph:
        """To ensure that the route is in the expected format"""
        if isinstance(route, BipartiteSynGraph):
            return route
        return converter(route, "bipartite")

    @staticmethod
    def fix_external_info_format(external_info: dict) -> dict:
        new_data = {}
        for smiles, provider in external_info.items():
            if provider == "syngenta":
                cat_value = 1.0
            elif provider == "vendor":
                cat_value = 0.5
            else:
                cat_value = 0.0
            new_data[smiles] = cat_value
        return new_data


@MetricsFactory.register_metrics("yield")
class YieldMetric(RouteMetric):
    """
    Subclass to compute the yield metric.
    The external data is expected to have the following format:
    {'reaction_smiles': yield (float between 0 and 1)}
    """

    def compute_metric(
        self,
        route: Union[
            BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph
        ],
        external_data: dict,
    ) -> MetricOutput:
        """To compute the yield metric"""
        route = self.check_route_format(route)
        output = MetricOutput()
        distance_function = "longest_sequence"
        score = 0.0
        normalization = 0.0
        reaction_root = route.get_roots()[0]
        for reaction in route.graph:
            distance_term = distance_function_calculator(
                distance_function, route, reaction, reaction_root
            )
            normalization += distance_term
            score += distance_term * external_data[reaction.smiles]
        output.metric_value = round(score / normalization, 2)
        output.raw_data = {
            "not_normalized_metric": score,
            "normalization_term": normalization,
            "distance_function": distance_function,
        }
        return output

    @staticmethod
    def check_route_format(
        route: Union[
            BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph
        ],
    ) -> MonopartiteReacSynGraph:
        """To ensure that the route is in the expected format"""
        if isinstance(route, MonopartiteReacSynGraph):
            return route
        return converter(route, "monopartite_reactions")


def route_metric_calculator(
    metric_name: str,
    route: Union[BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph],
    external_data: dict,
) -> MetricOutput:
    """
    To compute a route metric

    Parameters:
    -----------
    metric_name: str
        The name of the metric to be computed
    route: Union[BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph]
        The route for which the metric should be computed
    external_data: dict
        The external data needed for computing the route metric

    Returns:
    ----------
    output: MetricOutput
        Its attributes contain (i) the metric value (ii) a dictionary containing the raw data
        (not-normalized metric,normalization term)

    """
    if isinstance(
        route, (BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph)
    ):
        metric = MetricsFactory.get_metric_instance(metric_name)
        return metric.compute_metric(route, external_data)
    logger.error(
        f"The input route must be a SynGraph object. {type(route)} cannot be processed."
    )
    raise TypeError


############################################
# Distance functions Strategy
class DistanceStrategy(ABC):
    """Abstract srtategy for distance terms calculation"""

    @abstractmethod
    def distance(
        self, route: BipartiteSynGraph, node: Molecule, root: Molecule
    ) -> Union[int, float]:
        pass


class SimpleDistanceStrategy(DistanceStrategy):
    """
    Concrete distance strategy that simply returns the distance between the considered node and the root.
    The distance is calculated considering only ChemicalEquation nodes.
    """

    def distance(self, route: BipartiteSynGraph, node: Molecule, root: Molecule) -> int:
        path = find_path(route, node, root)
        return len([n for n in path if isinstance(n, ChemicalEquation)])


class LongestLinearSequenceWeigthedDistanceStrategy(DistanceStrategy):
    """
    Concrete distance strategy that is equal to 1 when the distance of the considered node from the root is 1 and
    that is equal to 0 when the distance is the length of the longest linear sequence+1.
    The +1 guarantees that the leaf nodes still have a weigth in the term (otherwise they would always have 0 impact)
    """

    def distance(self, route: BipartiteSynGraph, node: Molecule, root: Molecule):
        lls = descriptor_calculator(route, "longest_seq") + 1.0
        d = len(
            [n for n in find_path(route, node, root) if isinstance(n, ChemicalEquation)]
        )
        return -1.0 / lls * d + 1


# Context class
class DistanceContext:
    _distance_strategies = {
        "simple": SimpleDistanceStrategy(),
        "longest_sequence": LongestLinearSequenceWeigthedDistanceStrategy(),
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

    """
    context = DistanceContext()
    context.set_strategy(strategy_name)
    context.calculate_distance(route, node, root)
    return context.get_distance()
