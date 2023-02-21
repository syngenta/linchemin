import abc
from collections import defaultdict
from typing import Union

from linchemin.cgu.convert import converter
from linchemin.cgu.syngraph import (BipartiteSynGraph, MonopartiteReacSynGraph,
                                    SynGraph)
from linchemin.cheminfo.models import ChemicalEquation
from linchemin.rem.node_descriptors import node_descriptor_calculator
from linchemin.utilities import console_logger

"""
Module containing functions and classes for computing SynGraph descriptors

"""

logger = console_logger(__name__)


class DescriptorError(Exception):
    """ Base class for exceptions leading to unsuccessful descriptor calculation. """
    pass


class UnavailableDescriptor(DescriptorError):
    """ Raised if the selected descriptor is not among the available ones. """
    pass


class WrongGraphType(DescriptorError):
    """ Raised if the input gaph object is not of the required type. """
    pass


class InvalidInput(DescriptorError):
    """ Raised if the input route is None"""
    pass


class MismatchingGraphType(DescriptorError):
    """ Raised when graph of the same type are expected. """
    pass


class DescriptorCalculator(metaclass=abc.ABCMeta):
    """ Abstract class for DescriptorCalculator. """

    @abc.abstractmethod
    def compute_descriptor(self, graph: Union[BipartiteSynGraph, MonopartiteReacSynGraph]):
        """ Calculates the descriptor for the given graph.

            :param:
                graph: a graph object
                    It is the graph for which the descriptor should be computed

            :return:
                the value of the descriptor
        """
        pass


class NrBranches(DescriptorCalculator):
    """ Subclass of DescriptorCalculator representing the number of "AND" branches in a SynRoute. """
    info = 'Computes the number of branches in the input SynGraph'
    def compute_descriptor(self,
                           graph: Union[BipartiteSynGraph, MonopartiteReacSynGraph]) -> int:
        """ Takes a SynGraph and returns the number of ChemicalEquation nodes that are "parents" of more than one
            node. 0 corresponds to a linear route. """

        if graph is None:
            logger.error('The input route is None.')
            raise InvalidInput

        if isinstance(graph, BipartiteSynGraph):
            mp_graph = converter(graph, 'monopartite_reactions')
        elif isinstance(graph, MonopartiteReacSynGraph):
            mp_graph = graph
        else:
            logger.error(
                f'{type(graph)} is not supported. Only BipartiteSynGraph and MonopartiteReacSynGraph are accepted.')
            raise WrongGraphType

        branching_nodes = set()
        for reac, connections in mp_graph.graph.items():
            for c in connections:
                source_reactions = [r for r, products_set in mp_graph.graph.items() if c in products_set]
                if len(source_reactions) > 1:
                    for reaction in source_reactions:
                        branching_nodes.add(reaction)

        return len(branching_nodes)


class Branchedness(DescriptorCalculator):
    """ Subclass of DescriptorCalculator representing the "branchedness" of a SynGraph """
    info = 'Computes the "branchedness" of the input SynGraph, weighting the number of branching nodes with their ' \
           'distance from the root '

    def compute_descriptor(self, graph: Union[BipartiteSynGraph, MonopartiteReacSynGraph]):
        """ Takes a SynGraph and returns the "branchedness" computed as the number of branching nodes weighted by their
            distance from the root (the closer to the root, the better). 0 indicates a linear SynGraph"""
        if isinstance(graph, BipartiteSynGraph):
            mp_graph = converter(graph, 'monopartite_reactions')
        elif isinstance(graph, MonopartiteReacSynGraph):
            mp_graph = graph
        else:
            logger.error(
                f'{type(graph)} is not supported. Only BipartiteSynGraph and MonopartiteReacSynGraph are accepted.')
            raise WrongGraphType

        branching_nodes = set()
        for reac, connections in mp_graph.graph.items():
            for c in connections:
                source_reactions = [r for r, products_set in mp_graph.graph.items() if c in products_set]
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
    """ Subclass of DescriptorCalculator representing the longest linear sequence in a SynGraph. """
    info = 'Computes the longest linear sequence in the input SynGraph'

    def compute_descriptor(self,
                           graph: Union[BipartiteSynGraph, MonopartiteReacSynGraph]) -> int:
        """ Takes a SynGraph and returns the length of the longest sequence of ChemicalEquation between the SynRoot
            and the SynLeaves. """
        if isinstance(graph, BipartiteSynGraph):
            mp_graph = converter(graph, 'monopartite_reactions')
        elif isinstance(graph, MonopartiteReacSynGraph):
            mp_graph = graph
        else:
            logger.error(
                f'{type(graph)} is not supported. Only BipartiteSynGraph and MonopartiteReacSynGraph are accepted.')
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
    """ Subclass of DescriptorCalculator representing the number of ReactionStep nodes in a SynGraph. """
    info = 'Computes the number of chemical reactions in the input SynGraph'
    def compute_descriptor(self,
                           graph: Union[BipartiteSynGraph, MonopartiteReacSynGraph]) -> int:
        """ Takes a SynGraph and returns the number of ReactionStep nodes in it. """
        if isinstance(graph, BipartiteSynGraph):
            mp_graph = converter(graph, 'monopartite_reactions')
        elif isinstance(graph, MonopartiteReacSynGraph):
            mp_graph = graph
        else:
            logger.error(
                f'{type(graph)} is not supported. Only BipartiteSynGraph and MonopartiteReacSynGraph are accepted.')
            raise WrongGraphType

        return len(mp_graph.graph)


class PathFinder(DescriptorCalculator):
    """ Subclass of DescriptorCalculator representing the list of paths (ReactionStep nodes only) in a SynGraph. """
    info = 'Computes all the paths between the SynRoots and the SynLeaves in the input SynGraph'
    def compute_descriptor(self,
                           graph: Union[BipartiteSynGraph, MonopartiteReacSynGraph]) -> list:
        """ Takes a SynGraph/MonopartiteSynGraph and returns all the paths between the SynRoot and the SynLeaves
            (only ReactionStep nodes). """
        if not isinstance(graph, (BipartiteSynGraph, MonopartiteReacSynGraph)):
            logger.error(
                f'{type(graph)} is not supported. Only BipartiteSynGraph and MonopartiteReacSynGraph are accepted.')
            raise WrongGraphType

        root = graph.get_roots()
        leaves = graph.get_leaves()
        all_paths = []
        for leaf in leaves:
            path = find_path(graph, leaf, root[0])
            reaction_path = [step for step in path if isinstance(step, ChemicalEquation)]
            if reaction_path not in all_paths:
                all_paths.append(reaction_path)

        return all_paths


class Convergence(DescriptorCalculator):
    """ Subclass of DescriptorCalculator representing the convergence of a SynGraph. """
    info = 'Computes the "convergence" of the input SynGraph, as the ratio between the longest linear sequence and ' \
           'the number of steps '

    def compute_descriptor(self,
                           graph: Union[BipartiteSynGraph, MonopartiteReacSynGraph]) -> float:
        """ Takes a SynGraph and returns its convergence as the ratio between the longest linear sequence and the
            number of steps computed in the monopartite representation. """

        longest_lin_seq = descriptor_calculator(graph, 'longest_seq')
        n_steps = descriptor_calculator(graph, 'nr_steps')

        return longest_lin_seq / n_steps


class AvgBranchingFactor(DescriptorCalculator):
    """ Subclass of DescriptorCalculator representing the average branching factor of a SynGraph. """
    info = 'Computes the average branching factor of the input SynGraph'
    def compute_descriptor(self,
                           graph: Union[BipartiteSynGraph, MonopartiteReacSynGraph]) -> float:
        """ Takes a SynGraph and returns the average branching factor as the ratio between the number of non-root
            reaction nodes and the number of non-leaf reaction nodes. """
        if isinstance(graph, BipartiteSynGraph):
            mp_graph = converter(graph, 'monopartite_reactions')
        elif isinstance(graph, MonopartiteReacSynGraph):
            mp_graph = graph
        else:
            logger.error(
                f'{type(graph)} is not supported. Only BipartiteSynGraph and MonopartiteReacSynGraph are accepted.')
            raise WrongGraphType

        root_reactions = mp_graph.get_roots()
        nr_non_root_nodes = len(mp_graph.graph) - len(root_reactions)

        reaction_leaves = mp_graph.get_leaves()
        nr_non_leaf_nodes = len(mp_graph.graph) - len(reaction_leaves)

        return float(nr_non_root_nodes / nr_non_leaf_nodes)


class CDScore(DescriptorCalculator):
    """ Subclass of DescriptorCalculator representing the Convergent Disconnection Score of a SynGraph.
        https://pubs.acs.org/doi/10.1021/acs.jcim.1c01074
    """
    info = 'Computes the Convergent Disconnection Score of the input SynGraph'

    def compute_descriptor(self,
                           graph: Union[BipartiteSynGraph, MonopartiteReacSynGraph]) -> float:
        """ Takes a SynGraph and returns the average CDScore computing the score for each reaction involved. """

        if isinstance(graph, BipartiteSynGraph):
            mp_graph = converter(graph, 'monopartite_reactions')
        elif isinstance(graph, MonopartiteReacSynGraph):
            mp_graph = graph
        else:
            logger.error(
                f'{type(graph)} is not supported. Only BipartiteSynGraph and MonopartiteReacSynGraph are accepted.')
            raise WrongGraphType

        # Collect all unique reaction invovled in the route
        unique_reactions = set()
        for parent, children in mp_graph.graph.items():
            unique_reactions.add(parent)
            for child in children:
                unique_reactions.add(child)

        route_score = 0
        for reaction in unique_reactions:
            score = node_descriptor_calculator(reaction, 'cdscore')
            route_score += score

        return route_score / len(unique_reactions)


class AtomEfficiency(DescriptorCalculator):
    """ Subclass of DescriptorCalculator representing the atom efficiency of a SynGraph. """
    info = 'Computes the atom efficiency of the input SynGraph, as the ratio between the number of atoms in the ' \
           'target and the number of atoms in the starting materials '

    def compute_descriptor(self, graph: Union[BipartiteSynGraph, MonopartiteReacSynGraph]) -> float:
        """ Takes a SynGraph and returns its atom efficiency """
        if isinstance(graph, BipartiteSynGraph):
            root = graph.get_roots()[0]
            leaves = graph.get_leaves()
        elif isinstance(graph, MonopartiteReacSynGraph):
            root = graph.get_molecule_roots()[0]
            leaves = graph.get_molecule_leaves()
        else:
            logger.error(
                f'{type(graph)} is not supported. Only BipartiteSynGraph and MonopartiteReacSynGraph are accepted.')
            raise WrongGraphType
        target_n_atoms = root.rdmol.GetNumAtoms()
        all_atoms_leaves = sum(leaf.rdmol.GetNumAtoms() for leaf in leaves)
        return target_n_atoms / all_atoms_leaves


class DescriptorsCalculatorFactory:
    """ DescriptorCalculator Factory to give access to the descriptors.

        Attributes:
            route_descriptors: a dictionary
                It maps the strings representing the 'name' of a descriptor to the correct DescriptorCalculator subclass
    """
    route_descriptors = {
        'longest_seq': {'value': LongestSequence,
                        'info': LongestSequence.info},
        'nr_steps': {'value': NrReactionSteps,
                     'info': NrReactionSteps.info},
        'all_paths': {'value': PathFinder,
                      'info': PathFinder.info},
        'nr_branches': {'value': NrBranches,
                        'info': NrBranches.info},
        'branchedness': {'value': Branchedness,
                         'info': Branchedness.info},
        'branching_factor': {'value': AvgBranchingFactor,
                             'info': AvgBranchingFactor.info},
        'convergence': {'value': Convergence,
                        'info': Convergence.info},
        'cdscore': {'value': CDScore,
                    'info': CDScore.info},
        'atom_efficiency': {'value': AtomEfficiency,
                            'info': AtomEfficiency.info}
    }

    def select_route_descriptor(self, graph, descriptor: str):
        """ Takes a string indicating a descriptor and a SynGraph and returns the value of the descriptor """
        if descriptor not in self.route_descriptors:
            logger.error(f"'{descriptor}' is not a valid descriptor.")
            raise UnavailableDescriptor

        calculator = self.route_descriptors[descriptor]['value']
        return calculator().compute_descriptor(graph)


def descriptor_calculator(graph, descriptor: str):
    """ Gives access to the routes descriptors factory.

            :param:
                graph: a graph object
                    The single route for which the descriptor must be computed
                descriptor: a string
                    It indicates the descriptor to be computed

            :return:
                The value of the selected descriptor for the input graph
    """
    descriptor_selector = DescriptorsCalculatorFactory()
    return descriptor_selector.select_route_descriptor(graph, descriptor)


def get_available_descriptors():
    """ Returns the list of the available descriptors """
    return {f: additional_info['info'] for f, additional_info in DescriptorsCalculatorFactory.route_descriptors.items()}


def find_path(graph: SynGraph, leaf: str, root: str, path: Union[list, None] = None) -> list:
    """ Returns the path between two nodes in a SynGraph.

            :param:
                graph: a SynGraph
                leaf: the smiles of one SynLeaf
                root: the smiles of the SynRoot
                path: a list of smiles (default: empty list)

            :return:
                path/newpath: a list of smiles
    """
    if path is None:
        path = []
    path += [leaf]
    if leaf == root:
        return path
    for node in graph.graph[leaf]:
        if node not in path:
            if newpath := find_path(graph, node, root, path):
                return newpath


def is_subset(syngraph1, syngraph2) -> bool:
    """ Returns a boolean indicating whether syngraph1 is a subset of syngraph2. A route R1 is subset of another route
        R2 if (i) the dictionary of R1 SynGraph instace is subset of the dictionary of R2, (ii) R1 and R2 have the
        same roots, (iii) R1 and R2 have different leaves.

        :param:
            syngraph1, syngraph2: two instances of SynGraph or MonopartiteSynGraph

        :return:
            a boolean: True if syngraph1 is subset of syngraph2; False otherwise
    """
    if type(syngraph1) == BipartiteSynGraph:
        mp_graph1 = converter(syngraph1, 'monopartite_reactions')
    else:
        mp_graph1 = syngraph1
    if type(syngraph2) == BipartiteSynGraph:
        mp_graph2 = converter(syngraph2, 'monopartite_reactions')
    else:
        mp_graph2 = syngraph2
    return mp_graph2.get_leaves() != mp_graph1.get_leaves() and mp_graph1.get_roots() == mp_graph2.get_roots() and mp_graph1.graph.items() <= mp_graph2.graph.items()


def find_duplicates(syngraphs1: list, syngraphs2: list):
    """ Returns a list of tuples containing the common elements in the two input lists.

        :param:
            syngraphs1, syngraphs2: two lists of SynGraph objects

        :return:
            duplicates: a list of tuples
                It contains the id/source of identical routes;
                if there are no duplicates, nothing is returned and a message appears
    """
    if {type(s) for s in syngraphs1} != {type(s) for s in syngraphs2}:
        logger.error('The two input lists should contain graphs of the same type')
        raise MismatchingGraphType

    duplicates = []
    for g1 in syngraphs1:
        if g1 in syngraphs2:
            g2 = [g.source for g in syngraphs2 if g == g1]
            duplicates.append((g1.source, *g2))
    if duplicates:
        return duplicates
    else:
        print('No common routes were found')


def get_nodes_consensus(syngraphs: list) -> dict:
    """ Returns a dictionary of sets with the ChemicalEquation/Molecule instances as keys and the set of route ids
        involving the reaction/chemical as value.

        :param:
              syngraphs: a list of SynGraph objects

        :return:
            node_consensus: a dictionary of sets
                It contains the nodes and the ids of the routes that contain them in the form {nodes: {set of route ids}}
    """
    node_consensus = defaultdict(set)
    for graph in syngraphs:
        for reac, connections in graph.graph.items():
            node_consensus[reac].add(graph.source)
            for c in connections:
                node_consensus[c].add(graph.source)
    node_consensus = dict(sorted(node_consensus.items(), reverse=True, key=lambda item: len(item[1])))

    return node_consensus
