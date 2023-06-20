import abc
from collections import defaultdict
from typing import Union, List

from linchemin.cgu.convert import converter
from linchemin.cgu.syngraph import (
    BipartiteSynGraph,
    MonopartiteReacSynGraph,
    MonopartiteMolSynGraph,
)
from linchemin.cgu.syngraph_operations import find_path
from linchemin.cheminfo.models import ChemicalEquation, Molecule
from linchemin.rem.node_descriptors import node_descriptor_calculator
from linchemin.utilities import console_logger

"""
Module containing functions and classes for computing SynGraph descriptors

"""

logger = console_logger(__name__)


class DescriptorError(Exception):
    """Base class for exceptions leading to unsuccessful descriptor calculation."""

    pass


class UnavailableDescriptor(DescriptorError):
    """Raised if the selected descriptor is not among the available ones."""

    pass


class WrongGraphType(DescriptorError):
    """Raised if the input graph object is not of the required type."""

    pass


class InvalidInput(DescriptorError):
    """Raised if the input route is None"""

    pass


class MismatchingGraphType(DescriptorError):
    """Raised when graph of the same type are expected."""

    pass


class DescriptorCalculator(metaclass=abc.ABCMeta):
    """Abstract class for DescriptorCalculator."""

    @abc.abstractmethod
    def compute_descriptor(
        self,
        graph: Union[
            BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph
        ],
    ) -> Union[int, float, list]:
        """
        Calculates the descriptor for the given graph.

        Parameters:
        -----------
        graph: Union[BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph]
            The graph for which the descriptor should be computed

        Returns:
        --------
        dscriptor: Union[int, float, list]
            the value of the descriptor
        """
        pass


class NrBranches(DescriptorCalculator):
    """Subclass of DescriptorCalculator representing the number of "AND" branches in a SynRoute."""

    info = "Computes the number of branches in the input SynGraph"

    def compute_descriptor(
        self,
        graph: Union[
            BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph
        ],
    ) -> int:
        """Takes a SynGraph and returns the number of ChemicalEquation nodes that are "parents" of more than one
        node. 0 corresponds to a linear route."""

        if graph is None:
            logger.error("The input route is None.")
            raise InvalidInput

        if isinstance(graph, (BipartiteSynGraph, MonopartiteMolSynGraph)):
            mp_graph = converter(graph, "monopartite_reactions")
        elif isinstance(graph, MonopartiteReacSynGraph):
            mp_graph = graph
        else:
            logger.error(
                f"{type(graph)} is not supported. Only SynGraph objects are accepted."
            )
            raise WrongGraphType

        branching_nodes = set()
        for reac, connections in mp_graph.graph.items():
            for c in connections:
                source_reactions = [
                    r for r, products_set in mp_graph.graph.items() if c in products_set
                ]
                if len(source_reactions) > 1:
                    for reaction in source_reactions:
                        branching_nodes.add(reaction)

        return len(branching_nodes)


class Branchedness(DescriptorCalculator):
    """Subclass of DescriptorCalculator representing the "branchedness" of a SynGraph"""

    info = (
        'Computes the "branchedness" of the input SynGraph, weighting the number of branching nodes with their '
        "distance from the root "
    )

    def compute_descriptor(
        self,
        graph: Union[
            BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph
        ],
    ) -> float:
        """Takes a SynGraph and returns the "branchedness" computed as the number of branching nodes weighted by their
        distance from the root (the closer to the root, the better). 0 indicates a linear SynGraph
        """
        if isinstance(graph, (BipartiteSynGraph, MonopartiteMolSynGraph)):
            mp_graph = converter(graph, "monopartite_reactions")
        elif isinstance(graph, MonopartiteReacSynGraph):
            mp_graph = graph
        else:
            logger.error(
                f"{type(graph)} is not supported. SynGraph objects are accepted."
            )
            raise WrongGraphType

        branching_nodes = set()
        for reac, connections in mp_graph.graph.items():
            for c in connections:
                source_reactions = [
                    r for r, products_set in mp_graph.graph.items() if c in products_set
                ]
                if len(source_reactions) > 1:
                    for reaction in source_reactions:
                        branching_nodes.add(reaction)

        root = mp_graph.get_roots()[0]
        levels = defaultdict(set)
        for node in branching_nodes:
            path = find_path(graph, node, root)
            level = len(path) - 1
            levels[level].add(node)
        branchedness = 0.0
        for lv, s in levels.items():
            f = 1.0 / lv
            branchedness += f * len(s)
        return branchedness


class LongestSequence(DescriptorCalculator):
    """Subclass of DescriptorCalculator representing the longest linear sequence in a SynGraph."""

    info = "Computes the longest linear sequence in the input SynGraph"

    def compute_descriptor(
        self,
        graph: Union[
            BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph
        ],
    ) -> int:
        """Takes a SynGraph and returns the length of the longest sequence of ChemicalEquation between the SynRoot
        and the SynLeaves."""
        if isinstance(graph, (BipartiteSynGraph, MonopartiteMolSynGraph)):
            mp_graph = converter(graph, "monopartite_reactions")
        elif isinstance(graph, MonopartiteReacSynGraph):
            mp_graph = graph
        else:
            logger.error(
                f"{type(graph)} is not supported. Only SynGraph objects are accepted."
            )
            raise WrongGraphType

        root = mp_graph.get_roots()
        leaves = mp_graph.get_leaves()
        longest_sequence: list = []
        for leaf in leaves:
            path = find_path(mp_graph, leaf, root[0])
            reaction_path = list(path)
            if len(reaction_path) > len(longest_sequence):
                longest_sequence = reaction_path
        return len(longest_sequence)


class NrReactionSteps(DescriptorCalculator):
    """Subclass of DescriptorCalculator representing the number of ReactionStep nodes in a SynGraph."""

    info = "Computes the number of chemical reactions in the input SynGraph"

    def compute_descriptor(
        self,
        graph: Union[
            BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph
        ],
    ) -> int:
        """Takes a SynGraph and returns the number of ReactionStep nodes in it."""
        if isinstance(graph, (BipartiteSynGraph, MonopartiteMolSynGraph)):
            mp_graph = converter(graph, "monopartite_reactions")
        elif isinstance(graph, MonopartiteReacSynGraph):
            mp_graph = graph
        else:
            logger.error(
                f"{type(graph)} is not supported. Only Syngraph objects are accepted."
            )
            raise WrongGraphType

        return len(mp_graph.graph)


class PathFinder(DescriptorCalculator):
    """Subclass of DescriptorCalculator representing the list of paths (ReactionStep nodes only) in a SynGraph."""

    info = "Computes all the paths between the SynRoots and the SynLeaves in the input SynGraph"

    def compute_descriptor(
        self,
        graph: Union[
            BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph
        ],
    ) -> list:
        """Takes a SynGraph/MonopartiteSynGraph and returns all the paths between the SynRoot and the SynLeaves
        (only ReactionStep nodes)."""
        if isinstance(graph, MonopartiteMolSynGraph):
            graph = converter(graph, "monopartite_reactions")
        elif isinstance(graph, (BipartiteSynGraph, MonopartiteReacSynGraph)):
            graph = graph
        else:
            logger.error(
                f"{type(graph)} is not supported. Only Syngraph objects are accepted."
            )
            raise WrongGraphType

        root = graph.get_roots()
        leaves = graph.get_leaves()
        all_paths = []
        for leaf in leaves:
            path = find_path(graph, leaf, root[0])
            reaction_path = [
                step for step in path if isinstance(step, ChemicalEquation)
            ]
            if reaction_path not in all_paths:
                all_paths.append(reaction_path)

        return all_paths


class Convergence(DescriptorCalculator):
    """Subclass of DescriptorCalculator representing the convergence of a SynGraph."""

    info = (
        'Computes the "convergence" of the input SynGraph, as the ratio between the longest linear sequence and '
        "the number of steps "
    )

    def compute_descriptor(
        self,
        graph: Union[
            BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph
        ],
    ) -> float:
        """Takes a SynGraph and returns its convergence as the ratio between the longest linear sequence and the
        number of steps computed in the monopartite representation."""

        longest_lin_seq = descriptor_calculator(graph, "longest_seq")
        n_steps = descriptor_calculator(graph, "nr_steps")

        return longest_lin_seq / n_steps


class AvgBranchingFactor(DescriptorCalculator):
    """Subclass of DescriptorCalculator representing the average branching factor of a SynGraph."""

    info = "Computes the average branching factor of the input SynGraph"

    def compute_descriptor(
        self,
        graph: Union[
            BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph
        ],
    ) -> float:
        """Takes a SynGraph and returns the average branching factor as the ratio between the number of non-root
        reaction nodes and the number of non-leaf reaction nodes."""
        if isinstance(graph, (BipartiteSynGraph, MonopartiteMolSynGraph)):
            mp_graph = converter(graph, "monopartite_reactions")
        elif isinstance(graph, MonopartiteReacSynGraph):
            mp_graph = graph
        else:
            logger.error(
                f"{type(graph)} is not supported. Only SynGraph objects are accepted."
            )
            raise WrongGraphType

        root_reactions = mp_graph.get_roots()
        nr_non_root_nodes = len(mp_graph.graph) - len(root_reactions)

        reaction_leaves = mp_graph.get_leaves()
        nr_non_leaf_nodes = len(mp_graph.graph) - len(reaction_leaves)

        return float(nr_non_root_nodes / nr_non_leaf_nodes)


class CDScore(DescriptorCalculator):
    """Subclass of DescriptorCalculator representing the Convergent Disconnection Score of a SynGraph.
    https://pubs.acs.org/doi/10.1021/acs.jcim.1c01074
    """

    info = "Computes the Convergent Disconnection Score of the input SynGraph"

    def compute_descriptor(
        self,
        graph: Union[
            BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph
        ],
    ) -> float:
        """Takes a SynGraph and returns the average CDScore computing the score for each reaction involved."""

        if isinstance(graph, (BipartiteSynGraph, MonopartiteMolSynGraph)):
            mp_graph = converter(graph, "monopartite_reactions")
        elif isinstance(graph, MonopartiteReacSynGraph):
            mp_graph = graph
        else:
            logger.error(
                f"{type(graph)} is not supported. Only SynGraph objects are accepted."
            )
            raise WrongGraphType

        # Collect all unique reaction invovled in the route
        unique_reactions = set()
        for parent, children in mp_graph.graph.items():
            unique_reactions.add(parent)
            for child in children:
                unique_reactions.add(child)

        route_score = 0
        for reaction in unique_reactions:
            score = node_descriptor_calculator(reaction, "cdscore")
            route_score += score

        return route_score / len(unique_reactions)


class AtomEfficiency(DescriptorCalculator):
    """Subclass of DescriptorCalculator representing the atom efficiency of a SynGraph."""

    info = (
        "Computes the atom efficiency of the input SynGraph, as the ratio between the number of atoms in the "
        "target and the number of atoms in the starting materials "
    )

    def compute_descriptor(
        self,
        graph: Union[
            BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph
        ],
    ) -> float:
        """Takes a SynGraph and returns its atom efficiency"""
        if isinstance(graph, (BipartiteSynGraph, MonopartiteMolSynGraph)):
            root = graph.get_roots()[0]
            leaves = graph.get_leaves()
        elif isinstance(graph, MonopartiteReacSynGraph):
            root = graph.get_molecule_roots()[0]
            leaves = graph.get_molecule_leaves()
        else:
            logger.error(f"{type(graph)} is not supported. Only Syngraph are accepted.")
            raise WrongGraphType
        target_n_atoms = root.rdmol.GetNumAtoms()
        all_atoms_leaves = sum(leaf.rdmol.GetNumAtoms() for leaf in leaves)
        return target_n_atoms / all_atoms_leaves


class DescriptorsCalculatorFactory:
    """DescriptorCalculator Factory to give access to the descriptors.

    Attributes:
    ------------
    route_descriptors: a dictionary
        It maps the strings representing the 'name' of a descriptor to the correct DescriptorCalculator subclass
    """

    route_descriptors = {
        "longest_seq": {"value": LongestSequence, "info": LongestSequence.info},
        "nr_steps": {"value": NrReactionSteps, "info": NrReactionSteps.info},
        "all_paths": {"value": PathFinder, "info": PathFinder.info},
        "nr_branches": {"value": NrBranches, "info": NrBranches.info},
        "branchedness": {"value": Branchedness, "info": Branchedness.info},
        "branching_factor": {
            "value": AvgBranchingFactor,
            "info": AvgBranchingFactor.info,
        },
        "convergence": {"value": Convergence, "info": Convergence.info},
        "cdscore": {"value": CDScore, "info": CDScore.info},
        "atom_efficiency": {"value": AtomEfficiency, "info": AtomEfficiency.info},
    }

    def select_route_descriptor(self, graph, descriptor: str):
        """Takes a string indicating a descriptor and a SynGraph and returns the value of the descriptor"""
        if descriptor not in self.route_descriptors:
            logger.error(f"'{descriptor}' is not a valid descriptor.")
            raise UnavailableDescriptor

        calculator = self.route_descriptors[descriptor]["value"]
        return calculator().compute_descriptor(graph)


def descriptor_calculator(
    graph: Union[BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph],
    descriptor: str,
):
    """
    To compute a route descriptor.

    Parameters:
    ------------
    graph: Union[BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph]
        The route in SynGraph format for which the descriptor must be computed
    descriptor: str
        The descriptor to be computed

    Returns:
    ---------
    The value of the selected descriptor for the input graph

    Raises:
    -------
    UnavailableDescriptor: if the selected descriptor is not available

    WrongGraphType: if the input graph is not a SynGraph object

    Example:
    --------
    >>> graph = json.loads(open(az_path).read())
    >>> syngraph = translator('az_retro', graph[4], 'syngraph', out_data_model='bipartite')
    >>> n_steps = descriptor_calculator(syngraph, 'nr_steps')
    """
    descriptor_selector = DescriptorsCalculatorFactory()
    return descriptor_selector.select_route_descriptor(graph, descriptor)


def get_available_descriptors():
    """
    Returns the available options for the 'descriptor_calculator' function.

    Returns:
    --------
    available options: dict
        The dictionary listing arguments, options and default values of the 'descriptor_calculator' function

    Example:
    --------
    >>> options = get_available_descriptors()

    """
    return {
        f: additional_info["info"]
        for f, additional_info in DescriptorsCalculatorFactory.route_descriptors.items()
    }


# def find_path(
#     graph: Union[MonopartiteReacSynGraph, BipartiteSynGraph],
#     leaf: Union[Molecule, ChemicalEquation],
#     root: Union[Molecule, ChemicalEquation],
#     path: Union[list, None] = None,
# ) -> list:
#     """
#     To find a path between two nodes in a SynGraph.
#
#     Parameters:
#     ------------
#     graph: Union[MonopartiteReacSynGraph, BipartiteSynGraph]
#         The graph of interest
#     leaf:  Union[Molecule, ChemicalEquation]
#         The node at which the path should end
#     root: Union[Molecule, ChemicalEquation]
#         The node at which the path should start
#     path: Optional[Union[list, None]]
#         The list of Molecule/ChemicalEquation instances already discovered along the path (default None)
#
#     Returns:
#     --------
#     path: list
#         The path as list of Molecule and/or ChemicalEquation
#
#     Example:
#     ---------
#     >>> path = find_path(syngraph, leaf_mol, root_mol)
#     """
#     if path is None:
#         path = []
#     path += [leaf]
#     if leaf == root:
#         return path
#     for node in graph.graph[leaf]:
#         if node not in path:
#             if newpath := find_path(graph, node, root, path):
#                 return newpath


def is_subset(
    syngraph1: Union[
        BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph
    ],
    syngraph2: Union[
        BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph
    ],
) -> bool:
    """
    To check whether a graph is subset of another. A route R1 is subset of another route
    R2 if (i) the dictionary of R1 SynGraph instace is subset of the dictionary of R2, (ii) R1 and R2 have the
    same roots, (iii) R1 and R2 have different leaves.

    Parameters:
    ------------
    syngraph1: Union[BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph]
        The graph that might be subset
     syngraph2: Union[BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph]
        The graph that might be superset

    Returns:
    ---------
    bool
        True if syngraph1 is subset of syngraph2; False otherwise

    Raises:
    --------
    TypeError: if the input graph are not SynGraph objects
    """
    if isinstance(syngraph1, (BipartiteSynGraph, MonopartiteMolSynGraph)):
        mp_graph1 = converter(syngraph1, "monopartite_reactions")
    elif isinstance(syngraph1, MonopartiteReacSynGraph):
        mp_graph1 = syngraph1
    else:
        logger.error("Only SynGraph objects are accepted")
        raise TypeError
    if isinstance(syngraph2, (BipartiteSynGraph, MonopartiteMolSynGraph)):
        mp_graph2 = converter(syngraph2, "monopartite_reactions")
    elif isinstance(syngraph2, MonopartiteReacSynGraph):
        mp_graph2 = syngraph2
    else:
        logger.error("Only SynGraph objects are accepted")
        raise TypeError
    return (
        mp_graph2.get_leaves() != mp_graph1.get_leaves()
        and mp_graph1.get_roots() == mp_graph2.get_roots()
        and mp_graph1.graph.items() <= mp_graph2.graph.items()
    )


def find_duplicates(syngraphs1: list, syngraphs2: list) -> Union[List[tuple], None]:
    """Returns a list of tuples containing the common elements in the two input lists.

    Parameters:
    ------------
    syngraphs1: list
        A list of SynGraph obejcts
    syngraphs2: list
        The second list of SynGraph objects

    Returns:
    -------
    duplicates: Union[List[tuple], None]
        It contains the id/source of identical routes; if there are no duplicates, None is returned and a
        message is written to the screen

    Raises:
    --------
    MismatchingGraphType: if the input list contains different types of graph
    """
    if {type(s) for s in syngraphs1} != {type(s) for s in syngraphs2}:
        logger.error("The two input lists should contain graphs of the same type")
        raise MismatchingGraphType

    duplicates = []
    for g1 in syngraphs1:
        if g1 in syngraphs2:
            g2 = [g.source for g in syngraphs2 if g == g1]
            duplicates.append((g1.source, *g2))
    if duplicates:
        return duplicates
    else:
        print("No common routes were found")


def get_nodes_consensus(syngraphs: list) -> dict:
    """
    To get a dictionary of sets with the ChemicalEquation/Molecule instances as keys and the set of route ids
    involving the reaction/chemical as value.

    Parameters:
    ------------
    syngraphs: list
        The list of SynGraph for which node consensus should be computed

    Returns:
    ---------
    node_consensus: dict
        It contains the nodes and the ids of the routes that contain them in the form {nodes: {set of route ids}}
    """
    node_consensus = defaultdict(set)
    for graph in syngraphs:
        for reac, connections in graph.graph.items():
            node_consensus[reac].add(graph.source)
            for c in connections:
                node_consensus[c].add(graph.source)
    node_consensus = dict(
        sorted(node_consensus.items(), reverse=True, key=lambda item: len(item[1]))
    )

    return node_consensus
