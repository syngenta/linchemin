import copy
from typing import List, Tuple, Union

from linchemin import settings
from linchemin.cgu.convert import converter
from linchemin.cgu.syngraph import (
    BipartiteSynGraph,
    MonopartiteMolSynGraph,
    MonopartiteReacSynGraph,
)
from linchemin.cgu.syngraph_operations import merge_syngraph
from linchemin.cgu.translate import nx, translator
from linchemin.cheminfo.constructors import MoleculeConstructor
from linchemin.cheminfo.models import ChemicalEquation, Molecule
from linchemin.rem.route_descriptors import find_path
from linchemin.utilities import console_logger

""" Module containing functions and classes to identify, extract and mine routes """

logger = console_logger(__name__)


class RouteMiner:
    """A class for extracting routes from a list of SynGraph objects."""

    def __init__(
        self,
        route_list: List[
            Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
        ],
        root: Union[str, None] = settings.ROUTE_MINING.root,
    ):
        """To initialize a RouteMiner object"""
        if all(isinstance(r, MonopartiteReacSynGraph) for r in route_list):
            self.route_list = route_list
        elif any(
            isinstance(r, (BipartiteSynGraph, MonopartiteMolSynGraph))
            for r in route_list
        ):
            self.route_list = [
                converter(route, "monopartite_reactions") for route in route_list
            ]
        else:
            logger.error("Only syngraph objects can be used.")
            raise TypeError
        if isinstance(root, str) or root is None:
            self.root = root
        else:
            logger.error("The input target molecule should be in smiles (string) form.")
            raise TypeError

    def mine_routes(self) -> List[MonopartiteReacSynGraph]:
        """To extract a list of MonopartiteSynGraph routes from a tree."""
        tree = merge_syngraph(self.route_list)
        return TreeMiner(tree, self.root).mine_tree()


class TreeMiner:
    def __init__(
        self,
        tree: Union[MonopartiteReacSynGraph, BipartiteSynGraph],
        root: Union[str, None] = settings.ROUTE_MINING.root,
    ):
        """To initialize a TreeMiner object"""
        self.tree = self.set_tree(tree)
        self.root = self.set_root(root)

    def set_root(self, root: Union[str, None]) -> Union[Molecule, None]:
        """To set the root attribute"""
        if root is None:
            if isinstance(self.tree, MonopartiteReacSynGraph):
                extracted_roots = list(set(self.tree.get_molecule_roots()))
            else:
                extracted_roots = list(set(self.tree.get_roots()))
            return extracted_roots[0]
        else:
            mol_root = MoleculeConstructor().build_from_molecule_string(root, "smiles")
            tree_molecules = set(converter(self.tree, "monopartite_molecules").graph)
            if mol_root in tree_molecules:
                return mol_root
            logger.error("The selected root does not appear in the tree")
            raise KeyError

    @staticmethod
    def set_tree(
        tree: Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
    ):
        """To set the tree attribute."""
        if isinstance(
            tree, (MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph)
        ):
            return tree
        logger.error("Only syngraph objects can be used.")
        raise TypeError

    def mine_tree(self) -> List[MonopartiteReacSynGraph]:
        """To mine routes from a tree."""
        tree_nx = translator("syngraph", self.tree, "networkx", "bipartite")
        routes_nx = RouteFinder(tree_nx, self.root.smiles).find_routes()
        return [
            translator("networkx", route_nx, "syngraph", "monopartite_reactions")
            for route_nx in routes_nx
        ]


class RouteFinder:
    def __init__(
        self,
        nx_tree: nx.DiGraph,
        root: str,
        product_edge_label: str = settings.ROUTE_MINING.product_edge_label,
        reactant_edge_label: str = settings.ROUTE_MINING.reactant_edge_label,
    ):
        """To initialize a new RouteFinder object."""
        self.nx_tree = nx_tree
        self.root = root
        self.routes: List = []
        self.product_edge = product_edge_label
        self.reactant_edge = reactant_edge_label

    def find_routes(self) -> List[nx.DiGraph]:
        """To find routes from a tree."""
        initial_route = nx.DiGraph()
        initial_seen = set()
        stack = {}
        self.traverse_route(initial_route, self.root, initial_seen, stack)
        return self.routes

    def traverse_route(
        self,
        route_graph: nx.DiGraph,
        current_node,
        seen: set,
        stack: dict,
    ):
        """To extract routes from a Tree."""
        # Add the current node with its attributes to the route_graph
        route_graph.add_node(current_node, **self.nx_tree.nodes[current_node])
        seen.add(current_node)
        current_label = route_graph.nodes[current_node]["label"]
        # handle node differently depending on its type
        if current_label == settings.ROUTE_MINING.molecule_node_label:
            self.handle_M_node(route_graph, current_node, seen, stack)

        elif current_label == settings.ROUTE_MINING.chemicalequation_node_label:
            self.handle_CE_node(route_graph, current_node, seen, stack)

    def handle_M_node(
        self, route_graph: nx.DiGraph, current_node: str, seen: set, stack: dict
    ):
        """To handle Molecule nodes during the tree traversal"""
        # Find P edges connected to the current M node
        p_neighbors = self.get_P_neighbors(current_node)
        # If the current M node has more than one P edge, it is an OR node
        if len(p_neighbors) > 1:
            self.handle_OR_node(route_graph, p_neighbors, seen, stack)
        elif p_neighbors:
            self.handle_AND_node(route_graph, p_neighbors, seen, stack)

    def handle_CE_node(
        self, route_graph: nx.DiGraph, current_node: str, seen: set, stack: dict
    ) -> None:
        """To handle the ChemicalEquation nodes during the tree traversal."""
        # Find R edges connected to the current CE node
        r_neighbors = self.get_R_neighbors(current_node)

        stack[current_node] = [u for u, v, d in r_neighbors if u not in seen]
        while stack[current_node]:
            n = stack[current_node].pop()
            (u, v, edge_data) = next(tup for tup in r_neighbors if tup[0] == n)
            route_graph.add_edge(u, v, **edge_data)
            self.traverse_route(route_graph, u, seen, stack)

        stack.pop(current_node)
        if stack:
            self.handle_leaf_nodes(route_graph, seen, stack)
        elif route_graph not in self.routes:
            self.routes.append(route_graph)

    def handle_OR_node(
        self, route_graph: nx.DiGraph, p_neighbors: List[tuple], seen: set, stack: dict
    ) -> None:
        """To handle 'OR' Molecule nodes --> Molecule nodes that are product of more than one ChemicalEquation node."""
        for i, (u, v, edge_data) in enumerate(p_neighbors):
            if self.is_loop(u, seen):
                continue

            if i < len(p_neighbors) - 1:  # Only create new routes for non-last P edges
                new_route, new_seen, new_stack = self.create_new_route(
                    route_graph, seen, stack
                )
                new_route.add_edge(u, v, **edge_data)
                self.traverse_route(
                    new_route,
                    u,
                    new_seen,
                    new_stack,
                )
            else:
                route_graph.add_edge(u, v, **edge_data)
                self.traverse_route(
                    route_graph,
                    u,
                    seen,
                    stack,
                )

    @staticmethod
    def create_new_route(
        route_graph: nx.DiGraph, seen: set, stack: dict
    ) -> Tuple[nx.DiGraph, set, dict]:
        """To create a new route and the corresponding set of visited nodes."""
        return copy.deepcopy(route_graph), copy.deepcopy(seen), copy.deepcopy(stack)

    def handle_AND_node(
        self, route_graph: nx.DiGraph, p_neighbors: List[tuple], seen: set, stack: dict
    ) -> None:
        u, v, edge_data = p_neighbors[0]
        if self.is_loop(u, seen):
            return
        route_graph.add_edge(u, v, **edge_data)
        self.traverse_route(
            route_graph,
            u,
            seen,
            stack,
        )

    def is_loop(self, node: str, seen: set) -> bool:
        """To check if the next node is involved in a loop."""

        r_edge_list = self.get_R_neighbors(node)
        r_neighbors = {m for (m, ce, d) in r_edge_list}
        return r_neighbors.issubset(seen)

    def get_P_neighbors(self, node: str) -> List[tuple]:
        """To get the ChemicalEquation nodes, parents of a Molecule node, following the 'PRODUCT' edges."""
        return [
            (u, v, d)
            for u, v, d in self.nx_tree.in_edges(node, data=True)
            if d["label"] == self.product_edge
        ]

    def get_R_neighbors(self, node: str) -> List[tuple]:
        """To get the Molecule nodes, parents of a ChemicalEquation node, following the 'REACTANT' edges."""
        return [
            (u, v, d)
            for u, v, d in self.nx_tree.in_edges(node, data=True)
            if d["label"] == self.reactant_edge
        ]

    def handle_leaf_nodes(
        self, route_graph: nx.DiGraph, seen: set, stack: dict
    ) -> None:
        if ce_nodes := [ce for ce, val in stack.items() if len(val) > 0]:
            for ce in ce_nodes:
                while stack[ce]:
                    n = stack[ce].pop()
                    r_neighbors = self.get_R_neighbors(ce)
                    (u, v, edge_data) = next(tup for tup in r_neighbors if tup[0] == n)
                    route_graph.add_edge(u, v, **edge_data)

                    self.traverse_route(
                        route_graph,
                        u,
                        seen,
                        stack,
                    )
        if route_graph not in self.routes:
            self.routes.append(route_graph)


def mine_routes(
    input_list: Union[
        List[Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]]
    ],
    root: Union[str, None] = settings.ROUTE_MINING.root,
    new_reaction_list: Union[List[str], None] = settings.ROUTE_MINING.new_reaction_list,
) -> List[MonopartiteReacSynGraph]:
    """
    To mine all the routes that can be found in tree obtained by merging the input list of routes.

    Parameters:
    ----------
    input_list: Union[List[Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]]]
                A list of SynGraph routes.
    root : Optional[Union[str, None]]
            The smiles of the target molecule for which routes should be searched.
            If not provided, the root node will be determined automatically (default None)
    new_reaction_list : Optional[Union[List[str], None]]
        The list of smiles of the chemical reactions to be added.
        If not provided, only the input graph objects are considered (default None)

    Returns:
    --------
    extracted routes : List[MonopartiteReacSynGraph]
        A list of MonopartiteReacSynGraph objects corresponding to ALL the mined routes (including the input ones)

    Raises:
    -------
    TypeError: If the input data is not a list of SynGraph objects.

    Example:
    --------
    >>> input_list = [route1, route2]
    >>> root = 'CCC(=O)Nc1ccc(cc1)C(=O)N[C@@H](CO)C(=O)O'
    >>> new_reaction_list = ['CC(=O)Nc1ccccc1C(=O)O.[O-]S(=O)(=O)C(F)(F)F>>CC(=O)Nc1ccccc1C(=O)OS(=O)(=O)C(F)(F)F']
    >>> mine_routes(input_list,root,new_reaction_list)
    """
    if isinstance(input_list, list) and all(
        isinstance(
            r, (MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph)
        )
        for r in input_list
    ):
        graphs_list = copy.deepcopy(input_list)
        if new_reaction_list is not None:
            new_graph = build_graph_from_node_sequence(new_reaction_list)
            graphs_list.append(new_graph)

        return RouteMiner(graphs_list, root).mine_routes()
    logger.error("Only a list of syngraph objects can be used.")
    raise TypeError


def build_graph_from_node_sequence(new_nodes: List[str]) -> MonopartiteReacSynGraph:
    """To build a MonopartiteReacSynGraph from a list of reaction smiles"""
    new_nodes_d = [{"query_id": n, "output_string": s} for n, s in enumerate(new_nodes)]
    return MonopartiteReacSynGraph(new_nodes_d)
