import itertools
from abc import ABC, abstractmethod
from typing import List, Union, Type

import networkx as nx

from linchemin.cgu.convert import converter
from linchemin.cgu.syngraph import MonopartiteReacSynGraph, BipartiteSynGraph
from linchemin.cgu.translate import translator
from linchemin.utilities import console_logger

""" Module containing functions and classes to check the presence of issues in routes and handle them """
logger = console_logger(__name__)


class RouteChecker(ABC):
    """ Abstract class for route checkers """

    @abstractmethod
    def check_route(self,
                    route: MonopartiteReacSynGraph) -> MonopartiteReacSynGraph:
        pass

    @abstractmethod
    def handle_issue(self,
                     route: Union[MonopartiteReacSynGraph,
                                  nx.DiGraph],
                     nodes_to_remove: List) -> Union[MonopartiteReacSynGraph, nx.DiGraph]:
        pass


# Checkers factory definition
class CheckerFactory:
    """ Route checker Factory to give access to concrete route checkers.

        Attributes:
            _route_checkers: a dictionary
                It maps the strings representing the 'name' of a checker to the correct RouteChecker subclass

    """
    _route_checkers = {}

    @classmethod
    def register_checker(cls, name: str,
                         info: str):
        """Decorator for registering a new route checker.

        :param:
            name (str): The name of the route checker to be used as a key in the registry

            info (str): A brief description of the checker

        :return:
            function: The decorator function.
        """

        def decorator(checker_class: Type[RouteChecker]):
            cls._route_checkers[name.lower()] = {'class': checker_class,
                                                 'info': info}
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
        """Get an instance of the specified RouteChecker.

        :param:
            name (str): The name of the route checker.

        :return:
            RouteChecker: An instance of the specified route checker

        :raise:
            KeyError: If the specified checker is not registered.
        """
        route_checker = cls._route_checkers.get(name.lower())
        if route_checker is None:
            logger.error(f"Route checker '{name}' not found")
            raise KeyError
        return route_checker['class']()


# Concrete checkers definitions
@CheckerFactory.register_checker("cycle_check",
                                 "To handle any cycle in the route")
class CyclesChecker(RouteChecker):
    """ Concrete class to check and handle cycles in route"""

    def check_route(self,
                    route: MonopartiteReacSynGraph):
        nx_route = translator('syngraph', route, 'networkx', 'monopartite_reactions')
        try:
            if cycle_info := nx.find_cycle(nx_route, orientation="original"):
                print('cycles!')
                cycle_nodes = self.identify_cycle_nodes(cycle_info)
                route = self.handle_issue(nx_route, cycle_nodes)
        except nx.NetworkXNoCycle:
            return route
        d = [{'query_id': n, 'output_string': reaction} for n, reaction in enumerate(list(nx_route.nodes()))]
        return MonopartiteReacSynGraph(d)

    @staticmethod
    def identify_cycle_nodes(cycle_info: List) -> List:
        """ To identify the nodes belonging to a cycle """
        cycle_nodes = []
        for tup in cycle_info:
            cycle_nodes.extend((tup[0], tup[1]))
        return cycle_nodes

    def handle_issue(self,
                     route: nx.DiGraph,
                     problematic_nodes: List) -> nx.DiGraph:
        """ To remove the redundant nodes from a cycle  """
        nodes_to_remove = []
        for (i, n1), (j, n2) in itertools.product(enumerate(problematic_nodes), enumerate(problematic_nodes)):
            if j >= i and route.number_of_edges(u=n1, v=n2) + route.number_of_edges(u=n2, v=n1) > 1:
                nodes_to_remove.extend([n1, n2])
        route.remove_nodes_from(nodes_to_remove)
        return route


def route_checker(route: Union[MonopartiteReacSynGraph,
                               BipartiteSynGraph],
                  checker: str):
    """ To perform the selected sanity check on the input route

        :param:
            route: the input route

        :return:
            the checked route, possibly different from the input one

        :raises: TypeError if the route is not of type MonopartiteReacSynGraph or BipartiteSynGraph
    """
    factory = CheckerFactory()
    if not isinstance(route, (MonopartiteReacSynGraph, BipartiteSynGraph)):
        logger.error(f"""Object {type(route)} not supported. 
        Only MonopartiteReacSynGraph and BipartiteSynGraph are supported.""")
        raise TypeError

    if isinstance(route, BipartiteSynGraph):
        route = converter(route, 'monopartite_reactions')
    checker_class = factory.get_route_checker(checker)
    return checker_class.check_route(route)
