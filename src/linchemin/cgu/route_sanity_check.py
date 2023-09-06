import copy
import itertools
from abc import ABC, abstractmethod
from typing import List, Type, Union

import networkx as nx

from linchemin import settings
from linchemin.cgu.convert import converter
from linchemin.cgu.syngraph import (
    BipartiteSynGraph,
    MonopartiteMolSynGraph,
    MonopartiteReacSynGraph,
)
from linchemin.cgu.syngraph_operations import find_path, handle_dangling_nodes
from linchemin.cgu.translate import translator
from linchemin.cheminfo.models import ChemicalEquation, Molecule
from linchemin.utilities import console_logger

""" Module containing functions and classes to check the presence of issues in routes and handle them """
logger = console_logger(__name__)


class RouteSanityCheckError(Exception):
    """Custom"exception for route sanity check errors"""

    pass


class CyclicRouteError(RouteSanityCheckError):
    """Error te be raised when a route contains a cycle"""

    pass


class IsolatedNodesError(RouteSanityCheckError):
    """Error te be raised when a route contains an isolated sequences of nodes"""

    pass


class RouteChecker(ABC):
    """Abstract class for route checkers"""

    @abstractmethod
    def check_route(
        self, route: MonopartiteReacSynGraph, fix_issues: bool
    ) -> MonopartiteReacSynGraph:
        pass

    @abstractmethod
    def handle_issue(
        self,
        route: Union[MonopartiteReacSynGraph, nx.DiGraph],
        nodes_to_remove: Union[List, ChemicalEquation],
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
@CheckerFactory.register_checker("cycle_check", "To identify any cycle in the route")
class CyclesChecker(RouteChecker):
    """Concrete class to check and handle cycles in route"""

    def check_route(self, route: MonopartiteReacSynGraph, fix_issues: bool):
        nx_route = translator("syngraph", route, "networkx", "monopartite_reactions")
        try:
            if cycle_info := nx.find_cycle(nx_route, orientation="original"):
                # a cycle is found
                if fix_issues:
                    # if fix_issues is True, the problematic nodes are removed
                    cycle_nodes = self.identify_cycle_nodes(cycle_info)
                    route = self.handle_issue(nx_route, cycle_nodes)
                else:
                    # otherwise an error is raised
                    logger.error("The route contains a cycle")
                    raise CyclicRouteError
        except nx.NetworkXNoCycle:
            # no cycles: route is valid
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
    "isolated_nodes_check", "To identify any isolated sequence of nodes in the route"
)
class IsolatedNodesChecker(RouteChecker):
    """Concrete class to check and handle isolated sequences of nodes in a route"""

    def check_route(
        self, route: MonopartiteReacSynGraph, fix_issues: bool
    ) -> MonopartiteReacSynGraph:
        """To check if the input route could contain isolated sequences of nodes"""
        # if there is only one molecule root, there are no isolated sequences of nodes
        mol_roots = route.get_molecule_roots()
        if len(mol_roots) == 1:
            return route

        return self.search_isolated_nodes(route, mol_roots, fix_issues)

    def search_isolated_nodes(
        self,
        route: MonopartiteReacSynGraph,
        mol_roots: List[Molecule],
        fix_issues: bool,
    ) -> MonopartiteReacSynGraph:
        """To search any isolated sequences of nodes"""
        reaction_roots = route.get_roots()
        all_reactions = route.get_unique_nodes()
        route_copy = copy.deepcopy(route)
        for mol_root in mol_roots:
            # if a reaction having the molecule root as reagents exists
            if reaction_with_reagents := next(
                (
                    reaction
                    for reaction in all_reactions
                    if mol_root.uid in reaction.role_map["reagents"]
                ),
                None,
            ):
                reaction_root = self.get_parent_reaction(reaction_roots, mol_root)
                # if there is a path between the corresponding reaction root and the reaction with reagents,
                # the route is ok
                if find_path(route_copy, reaction_with_reagents, reaction_root):
                    continue
                # otherwise,the reaction root is generating an isolated sequence
                if fix_issues:
                    # if fix_issues is True, the problematic nodes are removed
                    self.handle_issue(route_copy, reaction_root)
                else:
                    # otherwise an error is raised
                    logger.error("The route contains an isolated sequence of nodes")
                    raise IsolatedNodesError
        return route_copy

    @staticmethod
    def get_parent_reaction(
        reaction_roots: List[ChemicalEquation], mol_root: Molecule
    ) -> ChemicalEquation:
        """To get the reaction root having the molecule root as product"""
        return next(r for r in reaction_roots if mol_root.uid in r.role_map["products"])

    def handle_issue(
        self, route: MonopartiteReacSynGraph, node_to_remove: ChemicalEquation
    ) -> MonopartiteReacSynGraph:
        """To remove the nodes in the isolated sequences of nodes"""
        return handle_dangling_nodes(route, node_to_remove)


def route_checker(
    route: Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph],
    checker: str,
    fix_issue: bool = settings.ROUTE_SANITY_CHECKS.fix_issue,
) -> MonopartiteReacSynGraph:
    """
    To perform a sanity check on the input route

    Parameters:
    -------------
    route: Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
        the input route
    checker: str
        The sanity check to be performed
    fix_issue: bool (default: True)
        Whether to solve the issue and return a fixed route (True) or to raise an error is an issue is found (False)

    Returns:
    ---------
    checked_route: MonopartiteReacSyngraph
        the checked route, possibly different from the input one

    Raises:
    --------
    TypeError: if the route is not of type MonopartiteReacSynGraph or BipartiteSynGraph

    CyclicRouteError: if the route contains a cycle

    IsolatedNodesError: if the route contains an isolated sequence of nodes

    Examples:
    ----------
    >>> graph = json.loads(open('ibm_file.json').read())
    >>> syngraph = translator('ibm_retro', graph[1], 'syngraph', out_data_model='monopartite_reactions')
    >>> checked_route = route_checker(syngraph, 'cycle_check', fix_issue=False)

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
    return checker_class.check_route(route, fix_issue)


def get_available_route_sanity_checks():
    checkers = CheckerFactory.list_route_checkers()
    return {f: additional_info["info"] for f, additional_info in checkers.items()}
