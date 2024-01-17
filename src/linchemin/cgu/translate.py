from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union

import networkx as nx
import pydot

from linchemin.cgu.iron import Iron
from linchemin.cgu.syngraph import (
    BipartiteSynGraph,
    MonopartiteMolSynGraph,
    MonopartiteReacSynGraph,
)
from linchemin.cgu.graph_transformations.data_model_converters import (
    DataModelCatalog,
    DataModelConverter,
)
from linchemin.cgu.graph_transformations.exceptions import UnavailableTranslation
from linchemin.cgu.graph_transformations.format_translators import (
    GraphFormatCatalog,
    get_formats,
)
from linchemin.utilities import console_logger

"""
Module containing functions and classes to transform a graph from an input format to a different output format.

"""

logger = console_logger(__name__)


@dataclass
class TranslationParameters:
    """Class to store translation parameters"""

    input_format: str = field(default_factory=str)
    output_format: str = field(default_factory=str)
    data_model_converter: Union[DataModelConverter, None] = None


class Translation:
    @staticmethod
    def start_translation(
        input_format: str, input_graph, output_format: str, out_data_model: str
    ):
        data_model_converter = DataModelCatalog.get_data_model(out_data_model)
        parameters = TranslationParameters(
            input_format=input_format,
            output_format=output_format,
            data_model_converter=data_model_converter,
        )
        return InputToIron().translate(input_graph, parameters)


# Chain of responsibility structure to handle the translation input -> iron -> syngraph -> iron -> output
class Handler(ABC):
    """Abstract handler for the concrete translators taking care of the single steps in the workflow"""

    @abstractmethod
    def translate(
        self,
        graph: Union[
            MonopartiteReacSynGraph,
            MonopartiteMolSynGraph,
            BipartiteSynGraph,
            nx.classes.digraph.DiGraph,
            pydot.Dot,
            list,
            dict,
            Iron,
        ],
        parameters: TranslationParameters,
    ):
        """To perform the translation/conversion of a specific step in the chain"""
        pass


class InputToIron(Handler):
    """Handler taking care of the first step: input to iron"""

    def translate(self, graph, parameters):
        """To perform the first step: input to iron"""
        # If the input graph is a syngraph
        if parameters.input_format == "syngraph":
            syngraph = parameters.data_model_converter.convert_syngraph(graph)
            graph = SynGraphToIron().translate(syngraph, parameters)
            return graph
        input_to_iron_translator = GraphFormatCatalog.get_format(
            parameters.input_format
        )
        graph = input_to_iron_translator.to_iron(graph)
        if parameters.output_format == "iron":
            graph = graph
        elif graph is None:
            return None
        else:
            graph = IronToSynGraph().translate(graph, parameters)
        return graph


class IronToSynGraph(Handler):
    """Handler taking care of the second step: iron to syngraph in the desired data model"""

    def translate(self, graph, parameters):
        """To perform the second step: iron to syngraph"""
        graph = parameters.data_model_converter.iron_to_syngraph(graph)
        if parameters.output_format == "syngraph" or graph is None:
            return graph
        else:
            graph = SynGraphToIron().translate(graph, parameters)
        return graph


class SynGraphToIron(Handler):
    """Handler taking care of the third step: syngraph to iron"""

    def translate(self, graph, parameters):
        """To perform the third step: syngraph to iron"""
        graph = parameters.data_model_converter.syngraph_to_iron(graph)
        graph = IronToOutput().translate(graph, parameters)
        return graph


class IronToOutput(Handler):
    """Handler taking care of the fourth step: iron to output"""

    def translate(self, graph, parameters):
        """To perform the fourth step: iron to output"""
        if parameters.output_format == "iron":
            return graph
        iron_to_output_translator = GraphFormatCatalog.get_format(
            parameters.output_format
        )
        graph = iron_to_output_translator.from_iron(graph)
        return graph


# Facade function
def translator(
    input_format: str,
    original_graph: Union[
        MonopartiteReacSynGraph,
        MonopartiteMolSynGraph,
        BipartiteSynGraph,
        nx.classes.digraph.DiGraph,
        pydot.Dot,
        list,
        dict,
        Iron,
    ],
    output_format: str,
    out_data_model: str,
) -> Union[
    MonopartiteReacSynGraph,
    BipartiteSynGraph,
    MonopartiteMolSynGraph,
    nx.DiGraph,
    pydot.Dot,
    Iron,
    None,
]:
    """
    To translate an input graph from a format/data model to another one

    Parameters:
    ------------
    input_format: str
        The format of the input graph object
    original_graph: Union[MonopartiteReacSynGraph, MonopartiteMolSynGraph, BipartiteSynGraph,
    nx.classes.digraph.DiGraph, pydot.Dot, list, dict, Iron]
        The input graph
    output_format: str
        The desired output format
    out_data_model: str
        The desired type of output graph (monopartite, bipartite...)

    Returns:
    ----------
    out_graph: Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph,
    nx.DiGraph, pydot.Dot, Iron, None]
        The output graph

    Raises:
    -------
    UnavailableTranslation: if the selected translation is not available

    Example:
    ---------
    >>> graph = json.loads(open('ibm_file.json').read())
    >>> nx_graph = translator('ibm_retro', graph[1], 'networkx', out_data_model='monopartite_molecules')

    """
    if "syngraph" in input_format and "syngraph" in output_format:
        logger.error(
            'To convert between data models, please use the "converter" function.'
        )
        raise UnavailableTranslation

    translation = Translation()

    return translation.start_translation(
        input_format, original_graph, output_format, out_data_model
    )


# Helper functions
def get_available_formats() -> dict:
    """
    To retrieve all the available formats names and brief descriptions

    Returns:
    --------
    d: dict
        The information about the available formats

    Example:
    >>> av_formats = get_available_formats()
    """
    return get_formats()


def get_input_formats() -> dict:
    """To retrieve graph formats available as input"""
    d = get_formats()
    return d["as_input"]


def get_output_formats() -> dict:
    """To retrieve graph formats available as output"""
    d = get_formats()
    return d["as_output"]


def get_available_data_models() -> dict:
    """
    To retrieve all the available data models names and brief descriptions

    Returns:
    --------
    d: dict
        The information about the available data models

    Example:
    >>> av_datamodels = get_available_data_models()
    """
    all_datamodels = DataModelCatalog.list_data_models()
    return {f: additional_info["info"] for f, additional_info in all_datamodels.items()}
