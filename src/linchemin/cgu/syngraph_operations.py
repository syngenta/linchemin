import copy
from abc import ABC, abstractmethod
from typing import List, Union

from linchemin.cgu.syngraph import (
    MonopartiteReacSynGraph,
    BipartiteSynGraph,
    MonopartiteMolSynGraph,
)
from linchemin.cheminfo.models import Molecule, ChemicalEquation
from linchemin.cheminfo.constructors import (
    ChemicalEquationConstructor,
    MoleculeConstructor,
)
import linchemin.utilities as utilities

"""Module containing functions and classes to perform operations on SynGraph instances"""

logger = utilities.console_logger(__name__)


class SmilesTypeError(TypeError):
    """Error raised if the provided smiles string is of the wrong type"""

    pass


def merge_syngraph(
    list_syngraph: List[
        Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
    ],
) -> Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]:
    """
    To merge a list od SynGraph objects in a single graph instance.

    Parameters:
    -----------
    list_syngraph: List[Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]]
        The list of the input SynGraph objects to be merged

    Returns:
    ---------
    merged: Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
        The new SynGraph object resulting from the merging of the input graphs; the SynGraph type is the same as the
        input graphs

    Raises:
    ---------
    TypeError: if the input list contains non SynGraph objects

    Example:
    --------
    >>> graph_ibm = json.loads(open('ibm_file.json').read())
    >>> all_routes_ibm = [translator('ibm_retro', g, 'syngraph', out_data_model='bipartite') for g in graph_ibm]
    >>> merged_graph = merge_syngraph(all_routes_ibm)
    """
    merged: Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
    if all(isinstance(x, MonopartiteReacSynGraph) for x in list_syngraph):
        merged = MonopartiteReacSynGraph()
    elif all(isinstance(x, BipartiteSynGraph) for x in list_syngraph):
        merged = BipartiteSynGraph()
    elif all(isinstance(x, MonopartiteMolSynGraph) for x in list_syngraph):
        merged = MonopartiteMolSynGraph()
    else:
        raise TypeError(
            "Invalid type. Only SynGraph objects can be merged. All routes must "
            "be in the same data model"
        )
    merged.source = "tree"
    for syngraph in list_syngraph:
        [merged.add_node(step) for step in syngraph]
    return merged


def add_reaction_to_syngraph(
    syngraph: Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph],
    reaction_to_add: str,
) -> Union[BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph]:
    """
    To add a chemical reaction to a SynGraph object

    Parameters:
    -------------
    syngraph: Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
        The SynGraph object to be modified

    reaction_to_add: str
        The smiles of the reaction to be added to the graph

    Returns:
    ----------
    new_graph: Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
        The SynGraph object with the addition of the new node and of the same type as the input graph

    Raises:
    -------
    TypeError: if the input graph is not in SynGraph format

    SmilesTypeError: if the provided smiles is not a valid reaction smiles
    """
    if not isinstance(
        syngraph, (BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph)
    ):
        logger.error("Only Syngraph objects are supported")
        raise TypeError

    if reaction_to_add.count(">") != 2:
        logger.error("Please insert a valid reaction smiles")
        raise SmilesTypeError
    reaction_graph = BipartiteSynGraph(
        [{"query_id": 0, "output_string": reaction_to_add}]
    )
    reactions = build_list_of_reactions(syngraph)
    bp_graph = BipartiteSynGraph(reactions)

    new_graph = merge_syngraph([bp_graph, reaction_graph])
    all_reactions = build_list_of_reactions(new_graph)
    return type(syngraph)(all_reactions)


def remove_nodes_from_syngraph(
    syngraph: Union[BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph],
    node_to_remove: str,
) -> Union[BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph]:
    """
    To remove a node and the paths form it to any leaf from a SynGraph object

    Parameters:
    -----------
    syngraph: Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
        The SynGraph object from which the nodes should be removed
    node_to_remove: str
        The smiles of the molecule/reaction to be removed

    Returns:
    ---------
    new_graph: Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
        A new SynGraph object from which the selected node and all its "parent" nodes are removed

    Raises:
    ---------
    TypeError: if the input objectis not a SynGraph

    KeyError: if the node to be removed is not present in the input SynGraph

    Example
    -------
    >>> new_graph = remove_nodes_from_syngraph(syngraph, 'CCC(=O)Nc1ccc(cc1)C(=O)N[C@@H](CO)C(=O)O')
    """
    if not isinstance(
        syngraph, (BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph)
    ):
        logger.error("Only Syngraph objects are supported")
        raise TypeError
    if node_to_remove.count(">") == 2:
        node = ChemicalEquationConstructor().build_from_reaction_string(
            node_to_remove, "smiles"
        )
    else:
        node = MoleculeConstructor().build_from_molecule_string(
            node_to_remove, "smiles"
        )
    if syngraph.graph.get(node, None) is None:
        logger.warning(
            "The selected node is not present in the input graph. The input graph is returned unchanged"
        )
        return syngraph
    nodes_to_remove = set()
    for leaf in syngraph.get_leaves():
        if path := find_path(syngraph, leaf, node):
            [nodes_to_remove.add(n) for n in path]
    new_graph = copy.deepcopy(syngraph)
    for n in nodes_to_remove:
        new_graph.remove_node(n)
    return new_graph


# Factory to extract reaction strings from a syngraph object
class ReactionsExtractor(ABC):
    """Abstract class for extracting a list of dictionary of reaction strings from a SynGraph object"""

    @abstractmethod
    def extract(
        self,
        syngraph: Union[
            MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph
        ],
    ) -> list:
        pass


class ReactionsExtractorFromBipartite(ReactionsExtractor):
    """ReactionsExtractor subclass to handle BipartiteSynGraph objects"""

    def extract(self, syngraph: BipartiteSynGraph) -> list:
        unique_reactions = set()
        for parent, children in syngraph.graph.items():
            if type(parent) == ChemicalEquation:
                unique_reactions.add(parent)
            for child in children:
                if type(child) == ChemicalEquation:
                    unique_reactions.add(child)
        sorted_reactions = sorted([reaction.smiles for reaction in unique_reactions])
        return [
            {"query_id": n, "input_string": reaction}
            for n, reaction in enumerate(sorted_reactions)
        ]


class ReactionsExtractorFromMonopartiteReaction(ReactionsExtractor):
    """ReactionsExtractor subclass to handle MonopartiteReacSynGraph objects"""

    def extract(self, syngraph: MonopartiteReacSynGraph) -> list:
        unique_reactions = set()
        for parent, children in syngraph.graph.items():
            unique_reactions.add(parent)
            for child in children:
                unique_reactions.add(child)
        sorted_reactions = sorted([reaction.smiles for reaction in unique_reactions])
        return [
            {"query_id": n, "input_string": reaction}
            for n, reaction in enumerate(sorted_reactions)
        ]


class ReactionsExtractorFromMonopartiteMolecules(ReactionsExtractor):
    """ReactionsExtractor subclass to handle MonopartiteMolSynGraph objects"""

    # TODO: check behavior when the same reactant appears twice (should be solved once the stoichiometry attribute is developed)
    def extract(self, syngraph: MonopartiteMolSynGraph) -> list:
        unique_reactions = set()
        for parent, children in syngraph.graph.items():
            for child in children:
                reactants = [
                    r.smiles
                    for r, products_set in syngraph.graph.items()
                    if child in products_set
                ]
                reaction_string = ">".join(
                    [".".join(reactants), ".".join([]), ".".join([child.smiles])]
                )
                chemical_equation_constructor = ChemicalEquationConstructor(
                    molecular_identity_property_name="smiles"
                )
                chemical_equation = (
                    chemical_equation_constructor.build_from_reaction_string(
                        reaction_string=reaction_string, inp_fmt="smiles"
                    )
                )
                unique_reactions.add(chemical_equation)
        sorted_reactions = sorted([reaction.smiles for reaction in unique_reactions])
        return [
            {"query_id": n, "input_string": reaction}
            for n, reaction in enumerate(sorted_reactions)
        ]


class ExtractorFactory:
    syngraph_types = {
        BipartiteSynGraph: ReactionsExtractorFromBipartite,
        MonopartiteReacSynGraph: ReactionsExtractorFromMonopartiteReaction,
        MonopartiteMolSynGraph: ReactionsExtractorFromMonopartiteMolecules,
    }

    def extract_reactions(self, syngraph):
        if type(syngraph) not in self.syngraph_types:
            raise TypeError(
                "Invalid graph type. Available graph objects are:",
                list(self.syngraph_types.keys()),
            )

        extractor = self.syngraph_types[type(syngraph)]
        return extractor().extract(syngraph)


def extract_reactions_from_syngraph(
    syngraph: Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
) -> List[dict]:
    """
    To extract the smiles of the chemical reaction included in a SynGraph object in a format suitable for atom mapping

    Parameters:
    -----------
    syngraph: Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
        The input graph for which the reaction smiles should be extracted

    Returns:
    ---------
    reactions: List[dict]
        The list of dictionary with the reaction smiles in the form [{'query_id': n, 'input_string': reaction}]

    Raises:
    -------
    TypeError: if the input graph is not a SynGraph object

    Example:
    --------
    >>> graph_az = json.loads(open('az_file.json').read())
    >>> syngraph = translator('az_retro', graph_az[0], 'syngraph', 'monopartite_reactions')
    >>> reactions = extract_reactions_from_syngraph(syngraph)
    """
    factory = ExtractorFactory()

    return factory.extract_reactions(syngraph)


def find_path(
    graph: Union[MonopartiteReacSynGraph, BipartiteSynGraph],
    leaf: Union[Molecule, ChemicalEquation],
    root: Union[Molecule, ChemicalEquation],
    path: Union[list, None] = None,
) -> list:
    """
    To find a path between two nodes in a SynGraph.

    Parameters:
    ------------
    graph: Union[MonopartiteReacSynGraph, BipartiteSynGraph]
        The graph of interest
    leaf:  Union[Molecule, ChemicalEquation]
        The node at which the path should end
    root: Union[Molecule, ChemicalEquation]
        The node at which the path should start
    path: Optional[Union[list, None]]
        The list of Molecule/ChemicalEquation instances already discovered along the path (default None)

    Returns:
    --------
    path: list
        The path as list of Molecule and/or ChemicalEquation

    Example:
    ---------
    >>> path = find_path(syngraph, leaf_mol, root_mol)
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


def build_list_of_reactions(syngraph):
    """To extract a list of reaction smiles from a syngraph and return it suitable to be used as input for syngraph building"""
    reactions = extract_reactions_from_syngraph(syngraph)
    return [
        {"query_id": d["query_id"], "output_string": d["input_string"]}
        for d in reactions
    ]


if __name__ == "__main__":
    print("main")
