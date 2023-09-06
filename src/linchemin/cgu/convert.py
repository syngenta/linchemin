from abc import ABC, abstractmethod
from typing import Union

from linchemin.cgu.syngraph import (BipartiteSynGraph, MonopartiteMolSynGraph,
                                    MonopartiteReacSynGraph, SynGraph)
from linchemin.cgu.syngraph_operations import extract_reactions_from_syngraph

"""
Module containing all the functions to convert from and into different SynGraph data models:
-bipartite molecules and reactions
-monopartite molecules
-monopartite reactions
"""


# Abstract strategy class
class Converter(ABC):
    """Abstract class for data model converters"""

    input_data_model: Union[
        BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph
    ]
    output_data_model: Union[
        BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph
    ]

    out_datamodels = {
        "bipartite": BipartiteSynGraph,
        "monopartite_reactions": MonopartiteReacSynGraph,
        "monopartite_molecules": MonopartiteMolSynGraph,
    }

    @abstractmethod
    def convert(
        self,
        graph: Union[
            BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph
        ],
    ):
        pass


# Concrete strategy classes
class BipartiteToMonopartiteReactions(Converter):
    """Converter subclass to convert a bipartite SynGraph in a monopartite reactions SynGraph"""

    input_data_model = BipartiteSynGraph
    output_data_model = MonopartiteReacSynGraph

    def convert(self, graph: BipartiteSynGraph):
        """To build a MonopartiteSynGraph with only ReactionStep nodes from a bipartite SynGraph."""
        out_reaction_list = extract_reactions_from_syngraph(graph)
        in_reaction_list = [
            {"query_id": d["query_id"], "output_string": d["input_string"]}
            for d in out_reaction_list
        ]
        mp_graph = MonopartiteReacSynGraph(in_reaction_list)
        mp_graph.source = graph.source
        return mp_graph


class BipartiteToMonopartiteMolecules(Converter):
    """Converter subclass to convert a bipartite SynGraph in a monopartite molecules SynGraph"""

    input_data_model = BipartiteSynGraph
    output_data_model = MonopartiteMolSynGraph

    def convert(self, graph: BipartiteSynGraph):
        """To build a MonopartiteSynGraph with only Molecules nodes from a bipartite SynGraph."""
        out_reaction_list = extract_reactions_from_syngraph(graph)
        in_reaction_list = [
            {"query_id": d["query_id"], "output_string": d["input_string"]}
            for d in out_reaction_list
        ]
        mp_graph = MonopartiteMolSynGraph(in_reaction_list)
        mp_graph.source = graph.source
        return mp_graph


class MonopartiteMoleculesToMonopartiteReactions(Converter):
    """Converter subclass to convert a monopartite reactions SynGraph in a monopartite molecules SynGraph"""

    input_data_model = MonopartiteMolSynGraph
    output_data_model = MonopartiteReacSynGraph

    def convert(self, graph: MonopartiteMolSynGraph) -> MonopartiteReacSynGraph:
        out_reaction_list = extract_reactions_from_syngraph(graph)
        in_reaction_list = [
            {"query_id": d["query_id"], "output_string": d["input_string"]}
            for d in out_reaction_list
        ]
        mp_graph = MonopartiteReacSynGraph(in_reaction_list)
        mp_graph.source = graph.source
        return mp_graph


class MonopartiteMoleculesToBiparite(Converter):
    """Converter subclass to convert a monopartite reactions SynGraph in a bipartite SynGraph"""

    input_data_model = MonopartiteMolSynGraph
    output_data_model = BipartiteSynGraph

    def convert(self, graph: MonopartiteMolSynGraph) -> BipartiteSynGraph:
        out_reaction_list = extract_reactions_from_syngraph(graph)
        in_reaction_list = [
            {"query_id": d["query_id"], "output_string": d["input_string"]}
            for d in out_reaction_list
        ]
        bp_graph = BipartiteSynGraph(in_reaction_list)
        bp_graph.source = graph.source
        return bp_graph


class MonopartiteReactionsToMonopartiteMolecules(Converter):
    """Converter subclass to convert a monopartite reactions SynGraph in a monopartite molecules SynGraph"""

    input_data_model = MonopartiteReacSynGraph
    output_data_model = MonopartiteMolSynGraph

    def convert(self, graph: MonopartiteReacSynGraph) -> MonopartiteMolSynGraph:
        out_reaction_list = extract_reactions_from_syngraph(graph)
        in_reaction_list = [
            {"query_id": d["query_id"], "output_string": d["input_string"]}
            for d in out_reaction_list
        ]
        bp_graph = MonopartiteMolSynGraph(in_reaction_list)
        bp_graph.source = graph.source
        return bp_graph


class MonopartiteReactionsToBipartite(Converter):
    """Converter subclass to convert a monopartite reactions SynGraph in a bipartite SynGraph"""

    input_data_model = MonopartiteReacSynGraph
    output_data_model = BipartiteSynGraph

    def convert(self, graph: MonopartiteReacSynGraph) -> BipartiteSynGraph:
        out_reaction_list = extract_reactions_from_syngraph(graph)
        in_reaction_list = [
            {"query_id": d["query_id"], "output_string": d["input_string"]}
            for d in out_reaction_list
        ]
        bp_graph = BipartiteSynGraph(in_reaction_list)
        bp_graph.source = graph.source
        return bp_graph


# Context class
class Conversion:
    def __init__(self, input_data_model, output_data_model):
        c = next(
            subclass
            for subclass in Converter.__subclasses__()
            if subclass.input_data_model == input_data_model
            and subclass.output_data_model == output_data_model
        )
        self.converter = c()

    def apply_conversion(self, graph):
        return self.converter.convert(graph)


def converter(
    graph: Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph],
    out_data_model: str,
) -> Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]:
    """
    To convert a SynGraph object in other types of SynGraph

    Parameters:
    ------------
    graph: Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
        The input graph as instance of one of the available SynGraph subclasses
    out_data_model: str
        The desired output data model.

    Returns:
    ---------
    converted graph: Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
        The converted graph

    Raises:
    --------
    TypeError: if the input graph is of the wrong type

    KeyError: if the selected output data model is not available

    Example:
    --------
    >>> mp_syngraph = converter(bp_syngraph, 'monopartite_molecules')

    """

    if type(graph) not in list(SynGraph.__subclasses__()):
        raise TypeError("Invalid input type. Only SynGraph objects con be converted.")
    if out_data_model not in Converter.out_datamodels:
        raise KeyError(
            f"Invalid output data model. Available data models are: {Converter.out_datamodels.keys()}"
        )
    if type(graph) == Converter.out_datamodels[out_data_model]:
        # print('The input SynGraph is already in the required data model')
        return graph

    in_data_model = type(graph)
    out_dm = Converter.out_datamodels[out_data_model]

    c = Conversion(in_data_model, out_dm)
    return c.apply_conversion(graph)
