import copy
import itertools
from abc import ABC, abstractmethod
from typing import List, Type, Union

import networkx as nx

from linchemin.cgu.convert import converter
from linchemin.cgu.syngraph import (
    BipartiteSynGraph,
    MonopartiteMolSynGraph,
    MonopartiteReacSynGraph,
)
from linchemin.cgu.translate import translator
from linchemin.cheminfo.models import ChemicalEquation, Molecule
from linchemin.rem.route_descriptors import find_path
from linchemin.utilities import console_logger

""" Module containing functions and classes to check the presence of issues in routes and handle them """
logger = console_logger(__name__)


class RouteChecker(ABC):
    """Abstract class for route checkers"""

    @abstractmethod
    def check_route(self, route: MonopartiteReacSynGraph) -> MonopartiteReacSynGraph:
        pass

    @abstractmethod
    def handle_issue(
        self, route: Union[MonopartiteReacSynGraph, nx.DiGraph], nodes_to_remove: List
    ) -> Union[MonopartiteReacSynGraph, nx.DiGraph]:
        pass


# Checkers factory definition
class CheckerFactory:
    """Route checker Factory to give access to concrete route checkers.

    Attributes:
    -------------
    _route_checkers: a dictionary
        It maps the strings representing the 'name' of a checker to the correct RouteChecker subclass

    """

    _route_checkers = {}

    @classmethod
    def register_checker(cls, name: str, info: str):
        """
        Decorator for registering a new route checker.

        Parameters:
        ------------
        name: str
            The name of the route checker to be used as a key in the registry
        info: str
            A brief description of the checker

        Returns:
        ---------
        function: The decorator function.
        """

        def decorator(checker_class: Type[RouteChecker]):
            cls._route_checkers[name.lower()] = {"class": checker_class, "info": info}
            return checker_class

        return decorator

    @classmethod
    def list_route_checkers(cls):
        """List the names of all available formats.

        :return:
            list: The names and information of the available route checkers.
        """
        return cls._route_checkers

    @classmethod
    def get_route_checker(cls, name: str) -> RouteChecker:
        """
        Get an instance of the specified RouteChecker.

        Parameters:
        ------------
        name: str
            The name of the route checker.

        Returns:
        ---------
        RouteChecker: An instance of the specified route checker

        Raises:
        ---------
            KeyError: If the specified checker is not registered.
        """
        route_checker = cls._route_checkers.get(name.lower())
        if route_checker is None:
            logger.error(f"Route checker '{name}' not found")
            raise KeyError
        return route_checker["class"]()


# Concrete checkers definitions
@CheckerFactory.register_checker("cycle_check", "To handle any cycle in the route")
class CyclesChecker(RouteChecker):
    """Concrete class to check and handle cycles in route"""

    def check_route(self, route: MonopartiteReacSynGraph):
        nx_route = translator("syngraph", route, "networkx", "monopartite_reactions")
        try:
            if cycle_info := nx.find_cycle(nx_route, orientation="original"):
                cycle_nodes = self.identify_cycle_nodes(cycle_info)
                route = self.handle_issue(nx_route, cycle_nodes)
        except nx.NetworkXNoCycle:
            return route

        d = [
            {"query_id": n, "output_string": reaction}
            for n, reaction in enumerate(list(nx_route.nodes()))
        ]
        return MonopartiteReacSynGraph(d)

    @staticmethod
    def identify_cycle_nodes(cycle_info: List) -> List:
        """To identify the nodes belonging to a cycle"""
        cycle_nodes = []
        for tup in cycle_info:
            cycle_nodes.extend((tup[0], tup[1]))
        return cycle_nodes

    def handle_issue(self, route: nx.DiGraph, problematic_nodes: List) -> nx.DiGraph:
        """To remove the redundant nodes from a cycle"""
        nodes_to_remove = []
        for (i, n1), (j, n2) in itertools.product(
            enumerate(problematic_nodes), enumerate(problematic_nodes)
        ):
            if (
                j >= i
                and route.number_of_edges(u=n1, v=n2)
                + route.number_of_edges(u=n2, v=n1)
                > 1
            ):
                nodes_to_remove.extend([n1, n2])
        route.remove_nodes_from(nodes_to_remove)
        return route


@CheckerFactory.register_checker(
    "isolated_nodes_check", "To handle any isolated sequence of nodes in the route"
)
class IsolatedNodesChecker(RouteChecker):
    """Concrete class to check and handle isolated sequences of nodes in a route"""

    def check_route(self, route: MonopartiteReacSynGraph) -> MonopartiteReacSynGraph:
        """To check if the input route could contain isolated sequences of nodes"""
        # if there is only one molecule root, there are no isolated sequences of nodes
        mol_roots = route.get_molecule_roots()
        if len(mol_roots) == 1:
            return route

        return self.search_isolated_nodes(route, mol_roots)

    def search_isolated_nodes(
        self, route: MonopartiteReacSynGraph, mol_roots: List[Molecule]
    ) -> MonopartiteReacSynGraph:
        """To search any isolated sequences of nodes"""
        reaction_roots = route.get_roots()
        reaction_leaves = route.get_leaves()
        route_copy = copy.deepcopy(route)
        for mol_root in mol_roots:
            # if a reaction leaf having the mol root as reagents exists
            if leaf := self.identify_reaction_leaf(reaction_leaves, mol_root):
                # identify the reaction root that has the mol root as product
                reaction_root = self.identify_reaction_root(reaction_roots, mol_root)
                # if there is a path between the reaction leaf and the reaction root, the route is ok
                if find_path(route, leaf, reaction_root):
                    return route_copy

                # otherwise, the path between them is retrieved and deleted from the route
                for reaction_leaf in reaction_leaves:
                    if path := find_path(route, reaction_leaf, reaction_root):
                        self.handle_issue(route_copy, path)
        return route_copy

    @staticmethod
    def identify_reaction_root(
        reaction_roots: List[ChemicalEquation], mol_root: Molecule
    ) -> ChemicalEquation:
        """To identify the reaction root having the molecule root as product"""
        return next(r for r in reaction_roots if mol_root.uid in r.role_map["products"])

    @staticmethod
    def identify_reaction_leaf(
        reaction_leaves: List[ChemicalEquation], mol_root: Molecule
    ) -> ChemicalEquation:
        """To identify the reaction leaf having the molecule root as reagent"""
        return next(
            (l for l in reaction_leaves if mol_root.uid in l.role_map["reagents"]), None
        )

    def handle_issue(
        self, route: MonopartiteReacSynGraph, nodes_to_remove: List
    ) -> MonopartiteReacSynGraph:
        """To remove the nodes in the isolated sequences of nodes"""
        [route.graph.pop(r, None) for r in nodes_to_remove]
        return route


def route_checker(
    route: Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph],
    checker: str,
) -> MonopartiteReacSynGraph:
    """
    To perform a sanity check on the input route

    Parameters:
    -------------
    route: Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
        the input route
    checker: str
        The sanity check to be performed

    Returns:
    ---------
    checked_route: MonopartiteReacSyngraph
        the checked route, possibly different from the input one

    Raises:
    --------
    TypeError: if the route is not of type MonopartiteReacSynGraph or BipartiteSynGraph

    Examples:
    ----------
    >>> graph = json.loads(open('ibm_file.json').read())
    >>> syngraph = translator('ibm_retro', graph[1], 'syngraph', out_data_model='monopartite_reactions')
    >>> checked_route = route_checker(syngraph, 'cycle_check')

    """
    factory = CheckerFactory()
    if not isinstance(
        route, (MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph)
    ):
        logger.error(
            f"""Object {type(route)} not supported.
        Only SynGraph objects are supported."""
        )
        raise TypeError

    if isinstance(route, (BipartiteSynGraph, MonopartiteMolSynGraph)):
        route = converter(route, "monopartite_reactions")
    checker_class = factory.get_route_checker(checker)
    return checker_class.check_route(route)


def get_available_route_sanity_checks():
    checkers = CheckerFactory.list_route_checkers()
    return {f: additional_info["info"] for f, additional_info in checkers.items()}
