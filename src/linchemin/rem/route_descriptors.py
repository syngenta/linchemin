import abc
from collections import defaultdict
from typing import List, Type, Union

from linchemin.cgu.convert import converter
from linchemin.cgu.syngraph import (
    BipartiteSynGraph,
    MonopartiteMolSynGraph,
    MonopartiteReacSynGraph,
)
from linchemin.cgu.syngraph_operations import find_path
from linchemin.rem.node_descriptors import chemical_equation_descriptor_calculator
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


class RouteDescriptor(metaclass=abc.ABCMeta):
    """Abstract class for DescriptorCalculator.

    Attributes:
    ----------
        info: A string describing the descriptor

        title: A string that can be used as title of a column of a dataframe containing the descriptor

        type: A string indicating the type of the descriptor (e.g., "number" for single values, "ratio" for fractions)

        fields: A list of string indicating the names of elements contributing to the descriptor (name of the
                descriptor for single values, names of the elements for fractions)

        order: An integer used to order the columns of the descriptors' dataframe
    """

    info: str
    title: str
    type: str
    fields: List[str]
    order: int

    @abc.abstractmethod
    def compute_descriptor(
        self, graph: MonopartiteReacSynGraph
    ) -> Union[int, float, list]:
        """
        To calculate the descriptor for the given graph.

        Parameters:
        -----------
        graph: MonopartiteReacSynGraph
            The graph for which the descriptor should be computed

        Returns:
        --------
        descriptor: Union[int, float, list]
            The value of the descriptor
        """
        pass

    def get_configuration(self) -> dict:
        return {
            "title": self.title,
            "type": self.type,
            "fields": self.fields,
            "order": self.order,
        }


class DescriptorsCalculatorFactory:
    """DescriptorCalculator Factory to give access to the descriptors.

    Attributes:
    ------------
    route_descriptors: a dictionary
        It maps the strings representing the 'name' of a descriptor to the correct DescriptorCalculator subclass
    """

    _registered_descriptors = {}

    @classmethod
    def register_descriptors(cls, name: str):
        """
        Decorator for registering a new descriptor.

        Parameters:
        ------------
        name: str
            The name of the descriptor to be used as a key in the registry

        Returns:
        ---------
        function: The decorator function.
        """

        def decorator(descriptor_class: Type[RouteDescriptor]):
            cls._registered_descriptors[name.lower()] = descriptor_class
            return descriptor_class

        return decorator

    @classmethod
    def get_descriptor(cls, name: str) -> RouteDescriptor:
        """
        To get an instance of the specified RouteDescriptor.

        Parameters:
        ------------
        name: str
            The name of the RouteDescriptor

        Returns:
        ---------
        RouteDescriptor: An instance of the specified RouteDescriptor

        Raises:
        -------
        UnavailableDescriptor: If the specified descriptor is not registered.
        """
        descriptor = cls._registered_descriptors.get(name.lower())
        if descriptor is None:
            logger.error(f"Descriptor '{name}' not found")
            raise UnavailableDescriptor
        return descriptor()

    @classmethod
    def list_route_descriptors(cls):
        """List the names of all available RouteDescriptors.

        Returns:
        ---------
        list: The names of the available descriptors.
        """
        return list(cls._registered_descriptors.keys())

    def get_descriptor_configuration(self, descriptor: str) -> dict:
        """To get the configuration dictionary of the selected descriptor"""
        descriptor_instance = self.get_descriptor(descriptor)
        return descriptor_instance.get_configuration()


@DescriptorsCalculatorFactory.register_descriptors("nr_branches")
class NrBranches(RouteDescriptor):
    """Subclass of DescriptorCalculator representing the number of "AND" branches in a SynRoute."""

    info = "Computes the number of branches in the input SynGraph"
    title = "N of Branches"
    type = "number"
    fields = ["nr_branches"]
    order = 30

    def compute_descriptor(
        self,
        graph: Union[
            BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph
        ],
    ) -> int:
        """Takes a SynGraph and returns the number of ChemicalEquation nodes that are "parents" of more than one
        node. 0 corresponds to a linear route."""

        branching_nodes = set()
        for reac, connections in graph:
            for c in connections:
                source_reactions = [r for r, products_set in graph if c in products_set]
                if len(source_reactions) > 1:
                    for reaction in source_reactions:
                        branching_nodes.add(reaction)

        return len(branching_nodes)


@DescriptorsCalculatorFactory.register_descriptors("branchedness")
class Branchedness(RouteDescriptor):
    """Subclass of DescriptorCalculator representing the "branchedness" of a SynGraph"""

    info = (
        'Computes the "branchedness" of the input SynGraph, weighting the number of branching nodes with their '
        "distance from the root "
    )
    title = "Branchedness"
    type = "number"
    fields = ["branchedness"]
    order = 40

    def compute_descriptor(self, graph: MonopartiteReacSynGraph) -> float:
        """
        To compute the input graph's "branchedness", as the number of branching nodes weighted by their
        distance from the root (the closer to the root, the better). 0 indicates a linear SynGraph
        """
        branching_nodes = self.find_branching_nodes(graph)
        root = graph.get_roots()[0]
        levels = defaultdict(set)
        for node in branching_nodes:
            path = find_path(graph, node, root)
            level = len(path) - 1
            levels[level].add(node)
        branchedness = 0.0
        for lv, s in levels.items():
            f = 1.0 / lv
            branchedness += f * len(s)
        return round(branchedness, 2)

    @staticmethod
    def find_branching_nodes(graph) -> set:
        """To identify the branching nodes in the graph"""
        branching_nodes = set()
        for parent, children in graph:
            for child in children:
                source_reactions = [
                    r for r, products_set in graph if child in products_set
                ]
                if len(source_reactions) > 1:
                    for reaction in source_reactions:
                        branching_nodes.add(reaction)
        return branching_nodes


@DescriptorsCalculatorFactory.register_descriptors("longest_seq")
class LongestSequence(RouteDescriptor):
    """Subclass of DescriptorCalculator representing the longest linear sequence in a SynGraph."""

    info = "Computes the longest linear sequence in the input SynGraph"
    title = "Longest Linear Sequence"
    type = "number"
    fields = ["longest_seq"]
    order = 20

    def compute_descriptor(self, graph: MonopartiteReacSynGraph) -> int:
        """
        To compute the length of the longest sequence of ChemicalEquation between the SynRoot
        and the SynLeaves of the input graph.
        """
        if len(graph.graph) == 1:
            return 1

        root = graph.get_roots()[0]
        leaves = graph.get_leaves()
        longest_sequence: list = []
        for leaf in leaves:
            reaction_path = find_path(graph, leaf, root)
            if len(reaction_path) > len(longest_sequence):
                longest_sequence = reaction_path
        return len(longest_sequence)


@DescriptorsCalculatorFactory.register_descriptors("nr_steps")
class NrReactionSteps(RouteDescriptor):
    """Subclass of DescriptorCalculator representing the number of ReactionStep nodes in a SynGraph."""

    info = "Computes the number of chemical reactions in the input SynGraph"
    title = "Total N of Steps"
    type = "number"
    fields = ["nr_steps"]
    order = 10

    def compute_descriptor(self, graph: MonopartiteReacSynGraph) -> int:
        """Takes a SynGraph and returns the number of ReactionStep nodes in it."""
        return len(graph.graph)


@DescriptorsCalculatorFactory.register_descriptors("convergence")
class Convergence(RouteDescriptor):
    """Subclass of DescriptorCalculator representing the convergence of a SynGraph."""

    info = (
        'Computes the "convergence" of the input SynGraph, as the ratio between the longest linear sequence and '
        "the number of steps "
    )
    title = "Convergence"
    type = "number"
    fields = ["convergence"]
    order = 50

    def compute_descriptor(self, graph: MonopartiteReacSynGraph) -> float:
        """
        To compute the input graph's convergence as the ratio between the longest linear sequence and the
        number of steps computed in the monopartite representation.
        """

        longest_lin_seq = descriptor_calculator(graph, "longest_seq")
        n_steps = descriptor_calculator(graph, "nr_steps")

        return round(longest_lin_seq / n_steps, 2)


@DescriptorsCalculatorFactory.register_descriptors("branching_factor")
class AvgBranchingFactor(RouteDescriptor):
    """Subclass of DescriptorCalculator representing the average branching factor of a SynGraph."""

    info = "Computes the average branching factor of the input SynGraph"
    title = "Avg Branching Factor"
    type = "number"
    fields = ["branching_factor"]
    order = 80

    def compute_descriptor(self, graph: MonopartiteReacSynGraph) -> float:
        """
        To compute the average branching factor as the ratio between the number of non-root
        reaction nodes and the number of non-leaf reaction nodes.
        """
        root_reactions = graph.get_roots()
        nr_non_root_nodes = len(graph.graph) - len(root_reactions)

        reaction_leaves = graph.get_leaves()
        nr_non_leaf_nodes = len(graph.graph) - len(reaction_leaves)

        return round(float(nr_non_root_nodes / nr_non_leaf_nodes), 2)


@DescriptorsCalculatorFactory.register_descriptors("cdscore")
class CDScore(RouteDescriptor):
    """Subclass of DescriptorCalculator representing the Convergent Disconnection Score of a SynGraph.
    https://pubs.acs.org/doi/10.1021/acs.jcim.1c01074
    """

    info = "Computes the Convergent Disconnection Score of the input SynGraph"
    title = "Convergent Disconnection Score"
    type = "number"
    fields = ["cdscore"]
    order = 60

    def compute_descriptor(self, graph: MonopartiteReacSynGraph) -> float:
        """Takes a SynGraph and returns the average CDScore computing the score for each reaction involved."""
        # Collect all unique reaction involved in the route
        unique_reactions = graph.get_unique_nodes()

        route_score = 0
        for reaction in unique_reactions:
            score = chemical_equation_descriptor_calculator(reaction, "ce_convergence")
            route_score += score

        return round(route_score / len(unique_reactions), 2)


@DescriptorsCalculatorFactory.register_descriptors("simplified_atom_effectiveness")
class SimplifiedAtomEffectiveness(RouteDescriptor):
    """Subclass of DescriptorCalculator representing the simplified atom effectiveness of a SynGraph."""

    info = (
        "Computes the simplified atom effectiveness of the input SynGraph, as the ratio between the number "
        "of atoms in the target and the number of atoms in the starting materials "
    )
    title = "Simplified Atom Effectiveness"
    type = "number"
    fields = ["simplified_atom_effectiveness"]
    order = 70

    def compute_descriptor(self, graph: MonopartiteReacSynGraph) -> float:
        """Takes a SynGraph and returns its simplified atom effectiveness"""
        root = graph.get_molecule_roots()[0]
        leaves = graph.get_molecule_leaves()
        target_n_atoms = root.rdmol.GetNumAtoms()
        all_atoms_leaves = sum(leaf.rdmol.GetNumAtoms() for leaf in leaves)
        return round(target_n_atoms / all_atoms_leaves, 2)


def descriptor_calculator(
    graph: Union[BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph],
    descriptor: str,
) -> Union[int, float, list]:
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
    Union[int, float, list]
        The value of the selected descriptor for the input graph

    Example:
    --------
    >>> graph = json.loads(open(az_path).read())
    >>> syngraph = translator('az_retro', graph[4], 'syngraph', out_data_model='bipartite')
    >>> n_steps = descriptor_calculator(syngraph, 'nr_steps')
    """
    graph = validate_input_graph(graph)
    descriptor = DescriptorsCalculatorFactory.get_descriptor(descriptor)
    return descriptor.compute_descriptor(graph)


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
        name: d_class.info
        for name, d_class in DescriptorsCalculatorFactory._registered_descriptors.items()
    }


def get_configuration(descriptor: str) -> dict:
    """To get the configuration dictionary for a given descriptor"""
    factory = DescriptorsCalculatorFactory()
    return factory.get_descriptor_configuration(descriptor)


def validate_input_graph(
    graph: Union[BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph]
) -> MonopartiteReacSynGraph:
    """
    To validate the input graph and converts it to a MonopartiteReacSynGraph if necessary.

    Parameters:
        graph: An instance of BipartiteSynGraph, MonopartiteReacSynGraph, or MonopartiteMolSynGraph.

    Returns:
        An instance of MonopartiteReacSynGraph.

    Raises:
        InvalidInput: If the input graph is None
        WrongGraphType: If the input graph type is not supported.
    """
    if graph is None:
        logger.error("The input route is None.")
        raise InvalidInput
    if isinstance(graph, MonopartiteReacSynGraph):
        return graph
    elif isinstance(graph, (BipartiteSynGraph, MonopartiteMolSynGraph)):
        return converter(graph, "monopartite_reactions")
    else:
        raise WrongGraphType(type(graph))


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
            g2 = [g.name for g in syngraphs2 if g == g1]
            duplicates.append((g1.name, *g2))
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
        for reac, connections in graph:
            node_consensus[reac].add(graph.name)
            for c in connections:
                node_consensus[c].add(graph.name)
    node_consensus = dict(
        sorted(node_consensus.items(), reverse=True, key=lambda item: len(item[1]))
    )

    return node_consensus
