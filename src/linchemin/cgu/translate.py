import datetime
import os
from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Type

import networkx as nx
import pydot

import linchemin.cheminfo.depiction as cid
import linchemin.cheminfo.functions as cif
from linchemin.cgu.convert import converter
from linchemin.cgu.iron import Direction, Edge, Iron, Node
from linchemin.cgu.syngraph import (BipartiteSynGraph, MonopartiteMolSynGraph,
                                    MonopartiteReacSynGraph)
from linchemin.cheminfo.models import ChemicalEquation, Molecule
from linchemin.IO import io as lio
from linchemin.utilities import console_logger

"""
Module containing functions and classes to transform a graph from an input format to a different output format.
"""

logger = console_logger(__name__)


# Translation Errors definitions
class TranslationError(Exception):
    """ Base class for exceptions leading to unsuccessful translation."""
    pass


class EmptyRoute(TranslationError):
    """ Raised if an empty route is found """
    pass


class InvalidRoute(TranslationError):
    """ Raised if the route does not contain at least two molecules connected by an edge"""


class UnavailableFormat(TranslationError):
    """ Raised if the selected format is not among the available ones"""
    pass


class UnavailableDataModel(TranslationError):
    """ Raised if the selected output format is not among the available ones """
    pass


class UnavailableTranslation(TranslationError):
    """ Raised if the required translation cannot be performed """
    pass


# Abstract class definition for translators
class AbsTranslator(ABC):
    """ Abstract class for format translators.

        Attributes:
              as_input: a string indicating if the 'to_iron' method has been implemented
                        If implemented: as_input = 'implemented'
                        If not implemented: as_input = None

              as_output: a string indicating if the 'from_iron' method has been implemented
                         If implemented: as_output = 'implemented'
                         If not implemented: as_output = None
    """

    as_input: Union[str, None] = None
    as_output: Union[str, None] = None

    @abstractmethod
    def from_iron(self, graph: Iron) -> Union[MonopartiteReacSynGraph,
                                              MonopartiteMolSynGraph,
                                              BipartiteSynGraph,
                                              nx.classes.digraph.DiGraph,
                                              pydot.Dot,
                                              list,
                                              dict,
                                              None]:
        """ Translates an Iron instance into a graph object of another type.

            :param:
                graph: an Iron instance
                    It is the graph of interest as an Ion instance

            :return:
                The same graph in the selected format
        """
        pass

    @abstractmethod
    def to_iron(self, graph: Union[MonopartiteReacSynGraph,
                                   MonopartiteMolSynGraph,
                                   BipartiteSynGraph,
                                   nx.classes.digraph.DiGraph,
                                   pydot.Dot,
                                   list,
                                   dict,
                                   None]) -> Union[Iron, None]:
        """ Translates a graph object of a specific type into an Iron instance.

            :param:
                graph: a graph object of a specific type

            :return:
                The same graph as Iron instance

        """
        pass

    def get_node_info(self,
                      node: ChemicalEquation,
                      iron: Iron) -> Tuple[int, Node]:
        """ To create or identify a specif node in an Iron instance """
        # if the node is not yet in the Iron instance, it is created and added to Iron
        if node not in [n.properties['node_type'] for n in iron.nodes.values()]:
            id_n = iron.i_node_number()
            id_n1, node1 = self.build_iron_node(node, id_n)
            iron.add_node(str(id_n), node1)
        else:
            # if the node is already included in Iron, the relative information is retrieved
            id_n1, node1 = next((i, n) for i, n in iron.nodes.items() if n.properties['node_type'] == node)
        return id_n1, node1

    @staticmethod
    def build_iron_node(node: ChemicalEquation,
                        id_n: int) -> Tuple[int, Node]:
        """ To build an Iron's node instance from a ChemicalEquation """
        # the unmapped smiles is built so that the route is suitable to be correctly displayed in a png file
        if type(node) == ChemicalEquation:
            unmapped_smiles = cif.rdrxn_to_string(node.rdrxn, out_fmt='smiles', use_atom_mapping=False)
        else:
            unmapped_smiles = node.smiles

        prop = {
            'node_unmapped_smiles': unmapped_smiles,
            'node_smiles': node.smiles,
            'node_type': node}
        return id_n, Node(iid=str(id_n), properties=prop, labels=[])


# Translator factory definition
class TranslatorFactory:
    """ Translator Factory to give access to the translations.

        Attributes:
            _translators: a dictionary
                It maps the strings representing the 'name' of a format to the correct AbsTranslator subclass

            _data_models: a dictionary
                It maps the strings representing the 'name' of a data model to the correct SynGraph subclass

    """
    _formats = {}
    _data_models = {}

    @classmethod
    def register_datamodel_translator(cls, name: str,
                                      info: str):
        """Decorator for registering a new data model translator.

        :param:
            name (str): The name of the data model to be used as a key in the registry

            info (str): A brief description of the data model

        :return:
            function: The decorator function.
        """

        def decorator(datamodel_translator_class: Type[AbsTranslator]):
            cls._data_models[name.lower()] = {'class': datamodel_translator_class,
                                              'info': info}
            return datamodel_translator_class

        return decorator

    @classmethod
    def register_format_translator(cls, name: str,
                                   info: str):
        """Decorator for registering a new format translator.

        :param:
            name (str): The name of the format to be used as a key in the registry.

            info (str): a brief description of the format

        :return:
            function: The decorator function.
        """

        def decorator(format_translator_class: Type[AbsTranslator]):
            cls._formats[name.lower()] = {'class': format_translator_class,
                                          'info': info}
            return format_translator_class

        return decorator

    @classmethod
    def list_format_translators(cls):
        """List the names of all available formats.

        :return:
            list: The names and information of the available formats.
        """
        return cls._formats

    @classmethod
    def list_datamodels_translators(cls):
        """List the names of all available formats.

        :return:
            list: The names and information of the available data models.
        """
        return cls._data_models

    @classmethod
    def get_format_translator(cls, name: str) -> AbsTranslator:
        """Get an instance of the specified AbsTranslator.

        :param:
            name (str): The name of the format translator.

        :return:
            AbsTranslator: An instance of the specified format translator.

        :raise:
            KeyError: If the specified format is not registered.
        """
        format_translator = cls._formats.get(name.lower())
        if format_translator is None:
            logger.error(f"Format '{name}' not found")
            raise UnavailableFormat
        return format_translator['class']()

    @classmethod
    def get_datamodel_translator(cls, name: str) -> AbsTranslator:
        """Get an instance of the specified AbsTranslator.

        :param:
            name (str): The name of the data model translator.

        :return:
            AbsTranslator: An instance of the specified format translator.

        :raise:
            KeyError: If the specified format is not registered.
        """
        datamodel_translator = cls._data_models.get(name.lower())
        if datamodel_translator is None:
            logger.error(f"Data model '{name}' not found")
            raise UnavailableDataModel
        return datamodel_translator['class']()

    def select_translation_to_iron(self, input_format: str,
                                   graph,
                                   out_data_model: str) -> Iron:
        """ Calls the correct translator to transform the selected input graph object into an Iron instance """
        if input_format == "iron":
            return graph

        if input_format == "syngraph":
            in_translator = self.get_datamodel_translator(out_data_model)
        else:
            in_translator = self.get_format_translator(input_format)
        return in_translator.to_iron(graph)

    def select_translation_from_iron(self, out_format: str,
                                     iron_graph: Iron,
                                     out_data_model: str):
        """ Calls the correct translator to transform an Iron instance into the selected output graph object """
        if out_format == 'iron':
            return iron_graph
        if out_format == 'syngraph':
            out_translator = self.get_datamodel_translator(out_data_model)
        else:
            out_translator = self.get_format_translator(out_format)
        return out_translator.from_iron(iron_graph)


# Concrete translators definitions
@TranslatorFactory.register_datamodel_translator("monopartite_reactions",
                                                 "A graph with reaction nodes only")
class TranslatorMonopartiteReacSynGraph(AbsTranslator):
    """ Translator subclass to handle translations into and from MonopartiteReacSynGraph instances"""
    as_input = 'implemented'
    as_output = 'implemented'

    def from_iron(self, iron_route: Iron) -> Union[MonopartiteReacSynGraph, None]:
        """ Translates an Iron instance into a MonopartiteReacSynGraph instance """
        try:
            if iron_route is None:
                raise EmptyRoute
            return MonopartiteReacSynGraph(iron_route)
        except EmptyRoute:
            logger.warning(
                'While translating from Iron to monopartite-reactions SynGraph object an empty route was found: '
                '"None" returned')
            return None

    def to_iron(self, mp_syngraph: MonopartiteReacSynGraph) -> Union[Iron, None]:
        """ Translates a MonopartiteReacSynGraph instance into an Iron instance """
        try:
            if mp_syngraph is None:
                raise EmptyRoute
            iron = Iron()
            id_e = 0
            for parent, children in mp_syngraph.graph.items():
                id_n1, node1 = self.get_node_info(parent, iron)

                for c in children:
                    id_n2, node2 = self.get_node_info(c, iron)

                    e = build_iron_edge(id_n1, id_n2, id_e)
                    iron.add_edge(str(id_e), e)
                    id_e += 1

            iron.source = mp_syngraph.source
            return iron

        except EmptyRoute:
            logger.warning(
                'While translating from a monopartite-reactions SynGraph to Iron an empty route was found: "None" '
                'returned')
            return None


@TranslatorFactory.register_datamodel_translator("bipartite",
                                                 "A graph with reaction and molecule nodes")
class TranslatorBipartiteSynGraph(AbsTranslator):
    """ Translator subclass to handle translations into and from BipartiteSynGraph instances """
    as_input = 'implemented'
    as_output = 'implemented'

    def from_iron(self, iron_route: Iron) -> Union[BipartiteSynGraph, None]:
        """ Translates an Iron instance into a BipartiteSynGraph instance """
        try:
            if iron_route is None:
                raise EmptyRoute
            return BipartiteSynGraph(iron_route)
        except EmptyRoute:
            logger.warning(
                'While translating from Iron to bipartite SynGraph object an empty route was found: "None" returned')
            return None

    def to_iron(self, syngraph: BipartiteSynGraph) -> Union[Iron, None]:
        """ Translates a BipartiteSynGraph instance into an Iron instance """
        try:
            if syngraph is None:
                raise EmptyRoute
            iron = Iron()
            id_e = 0
            for parent, children in syngraph.graph.items():
                id_n1, node1 = self.get_node_info(parent, iron)

                for c in children:
                    id_n2, node2 = self.get_node_info(c, iron)

                    e = build_iron_edge(id_n1, id_n2, id_e)
                    iron.add_edge(str(id_e), e)
                    id_e += 1

            iron.source = syngraph.source
            return iron
        except EmptyRoute:
            logger.warning(
                'While translating from a bipartite SynGraph to Iron an empty route was found: "None" returned')
            return None


@TranslatorFactory.register_datamodel_translator("monopartite_molecules",
                                                 "A graph with molecule nodes only")
class TranslatorMonopartiteMolSynGraph(AbsTranslator):
    """ Translator subclass to handle translations into and from MonopartiteMolSynGraph instances"""
    as_input = 'implemented'
    as_output = 'implemented'

    def from_iron(self, iron_route: Iron) -> Union[MonopartiteMolSynGraph, None]:
        """ Translates an Iron instance into a MonopartiteMolSynGraph instance """
        try:
            if iron_route is None:
                raise EmptyRoute
            return MonopartiteMolSynGraph(iron_route)
        except EmptyRoute:
            logger.warning(
                'While translating from Iron to a monopartite-molecules SynGraph object an empty route was found: '
                '"None" returned')
            return None

    def to_iron(self, mp_syngraph: MonopartiteMolSynGraph) -> Union[Iron, None]:
        """ Translates a MonopartiteReacSynGraph instance into an Iron instance """
        try:
            if mp_syngraph is None:
                raise EmptyRoute
            iron = Iron()
            id_e = 0
            for parent, children in mp_syngraph.graph.items():
                id_n1, node1 = self.get_node_info(parent, iron)

                for c in children:
                    id_n2, node2 = self.get_node_info(c, iron)

                    e = build_iron_edge(id_n1, id_n2, id_e)
                    iron.add_edge(str(id_e), e)
                    id_e += 1

            iron.source = mp_syngraph.source
            return iron
        except EmptyRoute:
            logger.warning(
                'While translating from a monopartite-molecules SynGraph to Iron an empty route was found: "None" '
                'returned')
            return None


@TranslatorFactory.register_format_translator("networkx",
                                              "Networkx DiGraph object")
class TranslatorNetworkx(AbsTranslator):
    """ Translator subclass to handle translations into and from Networkx objects """
    as_input = 'implemented'
    as_output = 'implemented'

    def from_iron(self, route_iron: Iron) -> Union[nx.classes.digraph.DiGraph, None]:
        """ Translates an Iron instance into a Networkx object """
        try:
            if route_iron is None:
                raise EmptyRoute
            elif route_iron.i_edge_number() == 0:
                # graph with no edges
                nx_graph = self.build_single_node_nx(route_iron)

            else:
                # graph with at least one edge
                nx_graph = self.build_multi_nodes_nx(route_iron)

            return nx_graph
        except EmptyRoute:
            logger.warning('While translating from Iron to NetworkX object an empty route was found: "None" returned')
            return None

    def build_multi_nodes_nx(self, route_iron: Iron) -> nx.classes.digraph.DiGraph:
        """ To build a networkx GiGraph with multiple nodes connected by edges """
        nx_graph = nx.DiGraph()
        nx_graph.graph['source'] = route_iron.source
        self.add_nx_nodes_edges(nx_graph, route_iron)
        self.add_nx_nodes_attributes(nx_graph, route_iron)

        return nx_graph

    @staticmethod
    def add_nx_nodes_edges(nx_graph: nx.classes.digraph.DiGraph,
                           route_iron: Iron) -> None:
        """ To add nodes and edges to a networky DiGraph instance """
        for id_e, edge in route_iron.edges.items():
            a = next(n.properties['node_smiles'] for i, n in route_iron.nodes.items() if i == edge.a_iid)
            b = next(n.properties['node_smiles'] for i, n in route_iron.nodes.items() if i == edge.b_iid)
            nx_graph.add_edge(a, b)
            attrs_e = {
                (a, b):
                    {'direction': edge.direction.string, 'properties': edge.properties,
                     'labels': edge.labels}
            }
            nx.set_edge_attributes(nx_graph, attrs_e, 'attributes')

    @staticmethod
    def add_nx_nodes_attributes(nx_graph: nx.classes.digraph.DiGraph,
                                route_iron: Iron) -> None:
        """ To add attributes to the nodes in the networkx DiGraph instance """
        for node in route_iron.nodes.values():
            attrs_n = {node.properties['node_smiles']: {'properties': node.properties, 'labels': node.labels,
                                                        'source': route_iron.source}}

            nx.set_node_attributes(nx_graph, attrs_n, 'attributes')

    @staticmethod
    def build_single_node_nx(route_iron: Iron) -> nx.classes.digraph.DiGraph:
        """ To create a networky network with only isolated nodes"""
        nx_graph = nx.DiGraph()
        nx_graph.graph['source'] = route_iron.source
        for node in route_iron.nodes.values():
            nx_graph.add_node(node.properties['node_smiles'])
            attrs_n = {node.properties['node_smiles']: {'properties': node.properties, 'labels': node.labels,
                                                        'source': route_iron.source}}

            nx.set_node_attributes(nx_graph, attrs_n, 'attributes')
        return nx_graph

    def to_iron(self, route: nx.classes.digraph.DiGraph) -> Union[Iron, None]:
        """ Translates a Networkx object into an Iron instance """
        iron = Iron()

        for id_n, (node, data) in enumerate(route.nodes.items()):
            iron_node = Node(iid=str(id_n), properties={'node_smiles': node}, labels=[])
            iron.add_node(str(id_n), iron_node)
        id_e = 0
        for edge, data in route.edges.items():
            iron_edge = self.nx_edge_to_iron_edge(edge, id_e, iron)
            iron.add_edge(id_e, iron_edge)
            id_e = id_e + 1

        return iron

    @staticmethod
    def nx_edge_to_iron_edge(edge,
                             id_e: int,
                             iron: Iron) -> Edge:
        """ To transform a nx edge into an Iron edge """
        a_id = next(n.iid for n in iron.nodes.values() if n.properties['node_smiles'] == edge[0])
        b_id = next(n.iid for n in iron.nodes.values() if n.properties['node_smiles'] == edge[1])
        return build_iron_edge(a_id, b_id, id_e)


@TranslatorFactory.register_format_translator("pydot",
                                              "pydot.Dot object")
class TranslatorDot(AbsTranslator):
    """ Translator subclass to handle translations into and from Dot objects """
    as_input = 'implemented'
    as_output = 'implemented'

    def from_iron(self, route_iron: Iron) -> Union[pydot.Dot, None]:
        """ Translates an Iron instance into a Pydot object """
        try:
            if route_iron is None:
                raise EmptyRoute
            dot_graph = pydot.Dot(route_iron.source, graph_type='digraph')
            # Translating iron nodes into dot nodes
            for id_n, node in route_iron.nodes.items():
                # NB Sometimes Pydot adds unwanted double quotes to the string name of the node
                # related issue: https://github.com/pydot/pydot/issues/72
                dot_graph.add_node(pydot.Node(node.properties['node_smiles']))

            # Translating iron edges into dot edges
            for id_e, edge in route_iron.edges.items():
                a = next(n.properties['node_smiles'] for i, n in route_iron.nodes.items() if i == edge.a_iid)
                b = next(n.properties['node_smiles'] for i, n in route_iron.nodes.items() if i == edge.b_iid)
                dot_graph.add_edge(pydot.Edge(a, b))
            return dot_graph
        except EmptyRoute:
            logger.warning('While translating from Iron to PyDot object an empty route was found: "None" returned')
            return None

    def to_iron(self, route: pydot.Dot) -> Union[Iron, None]:
        """ Translates a Pydot object into an Iron instance """
        iron = Iron()

        for id_n, node in enumerate(route.get_nodes()):
            # Since pydot has some issues in treating quotes in string, the stripping step is to make sure
            # that no extra, unwanted double quotes remain in the node_smiles string
            iron_node = Node(properties={'node_smiles': node.get_name().strip('\"')}, iid=str(id_n), labels=[])
            iron.add_node(str(id_n), iron_node)
        id_e = 0
        for edge in route.get_edges():
            iron_edge = self.pydot_edges_to_iron_edge(edge, id_e, iron)
            iron.add_edge(id_e, iron_edge)
            id_e = id_e + 1
        return iron

    @staticmethod
    def pydot_edges_to_iron_edge(edge,
                                 id_e: int,
                                 iron: Iron) -> Edge:
        """ To transform pydot edge to iron edge """
        a = edge.get_source().strip('\"')
        a_id = next(n.iid for n in iron.nodes.values() if n.properties['node_smiles'] == a)
        b = edge.get_destination().strip('\"')
        b_id = next(n.iid for n in iron.nodes.values() if n.properties['node_smiles'] == b)
        return build_iron_edge(a_id, b_id, id_e)


@TranslatorFactory.register_format_translator("ibm_retro",
                                              "IBMRXN output (dictionary)")
class TranslatorIbm(AbsTranslator):
    """ Translator subclass to handle translations of outputs generated by IBM CASP tool """
    as_input = 'implemented'
    as_output = None

    def from_iron(self, graph: Iron) -> Union[list, None]:
        pass

    def to_iron(self, route: dict) -> Union[Iron, None]:
        """ Translates a dictionary generated by the IBM CASP tool into an Iron instance"""
        try:
            if not route:
                raise EmptyRoute
            iron_graph = ibm_dict_to_iron(route, iron=None, parent=None)

            if iron_graph.i_edge_number() == 0:
                raise InvalidRoute

            iron_graph.source = 'ibm_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            return iron_graph

        except EmptyRoute:
            logger.warning('While translating from IBM to Iron an empty route was found: "None" returned')
            return None
        except InvalidRoute:
            logger.warning(
                'While translating from IBM to Iron an invalid route containing only one molecule was found: '
                '"None" returned')
            return None


@TranslatorFactory.register_format_translator("az_retro",
                                              "AiZynthFinder output (dictionary)")
class TranslatorAz(AbsTranslator):
    """ Translator subclass to handle translations of outputs generated by AZ CASP tool """
    as_input = 'implemented'
    as_output = None

    def from_iron(self, graph: Iron):
        pass

    def to_iron(self, route: dict) -> Union[Iron, None]:
        """ Translates a dictionary generated by the AZ CASP tool into an Iron instance """
        try:
            if not route:
                raise EmptyRoute
            iron_graph = az_dict_to_iron(route, iron=None, parent=None)
            iron_graph.source = 'az_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            return iron_graph
        except EmptyRoute:
            logger.warning('While translating from AZ to Iron an empty route was found: "None" returned')
            return None


@TranslatorFactory.register_format_translator("mit_retro",
                                              "Askcos output (dictionary)")
class TranslatorMit(AbsTranslator):
    """ Translator subclass to handle translations of outputs generated by Askcos CASP tool """
    as_input = 'implemented'
    as_output = None

    def from_iron(self, graph: Iron):
        pass

    def to_iron(self, route: dict) -> Union[Iron, None]:
        """ Translates a dictionary generated by the Askcos CASP tool into an Iron instance """
        try:
            if not route:
                raise EmptyRoute
            iron_graph = mit_dict_to_iron(route, iron=None, parent=None)
            iron_graph.source = 'mit_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            return iron_graph
        except EmptyRoute:
            logger.warning('While translating from Asckos to Iron an empty route was found: "None" returned')
            return None


@TranslatorFactory.register_format_translator("noc",
                                              "NOC format (dictionary)")
class TranslatorNOC(AbsTranslator):
    """ Translator subclass to handle translations of NOC-compatible documents """
    as_input = None
    as_output = 'implemented'

    def from_iron(self, route_iron: Iron) -> Union[dict, None]:
        """ Translates an Iron instance into an NOC-compatible document """
        try:
            if route_iron is None:
                raise EmptyRoute
            nodes = route_iron.nodes
            node_documents: List = []
            edge_documents: List = []
            for iid, n in nodes.items():
                node_instance = n.properties.get('node_type')
                node_documents = self.build_node_documents(node_instance, node_documents)

                if type(node_instance) is ChemicalEquation:
                    edge_documents = self.build_edges_documents(node_instance, edge_documents)

            return {'nodes': node_documents, 'edges': edge_documents}
        except EmptyRoute:
            logger.warning('While translating from Iron to NOC format an empty route was found: "None" returned')
            return None

    @staticmethod
    def build_edges_documents(node_instance: ChemicalEquation,
                              edge_documents) -> List[dict]:
        """ To build the list of dictionaries with edges information """
        for item in node_instance.role_map.get('reactants'):
            edge_doc = {'type': 'REACTANT', 'source': item, 'destination': node_instance.uid,
                        'properties': {}}
            edge_documents.append(edge_doc)
        for item in node_instance.role_map.get('reagents'):
            edge_doc = {'type': 'REAGENT', 'source': item, 'destination': node_instance.uid,
                        'properties': {}}
            edge_documents.append(edge_doc)
        for item in node_instance.role_map.get('products'):
            edge_doc = {'type': 'PRODUCT', 'source': node_instance.uid, 'destination': item,
                        'properties': {}}
            edge_documents.append(edge_doc)
        # append Template node
        # append Pattern nodes
        # append Template - ChemicalEquation relationships
        # append Template - Patterns relationships
        # append Patterns - Molecules relationships (to be calculated while constructing chemical equation!!!!)
        return edge_documents

    @staticmethod
    def build_node_documents(node_instance: Union[Molecule, ChemicalEquation],
                             node_documents: list) -> List[dict]:
        """ To build the list of dictionaries with nodes information """
        node_dict = node_instance.to_dict()
        node_document = {'type': node_dict.get('type'), 'uid': node_dict.get('uid'),
                         'properties': {**{'smiles': node_dict.get('smiles')}, **node_dict.get('hash_map')}}
        if type(node_instance) == ChemicalEquation and node_instance.disconnection is not None:
            node_document['disconnection_uid'] = node_instance.disconnection.uid
        node_documents.append(node_document)
        return node_documents

    def to_iron(self, graph: dict) -> Union[Iron, None]:
        pass


@TranslatorFactory.register_format_translator("pydot_visualization",
                                              "PNG file")
class TranslatorDotVisualization(AbsTranslator):
    """ Translator subclass to generate pictures """
    as_input = None
    as_output = 'implemented'

    def from_iron(self, route_iron: Iron) -> None:
        """ Translates an Iron instance into a PNG picture """
        try:
            if route_iron is None:
                raise EmptyRoute
            dot_graph = pydot.Dot(route_iron.source, graph_type='digraph')
            # Translating iron nodes into dot nodes
            for id_n, node in route_iron.nodes.items():
                self.create_dot_node(node, id_n, dot_graph)

            # Translating iron edges into dot edges
            for edge in route_iron.edges.values():
                self.create_dot_edge(edge, route_iron, dot_graph)

            dot_graph.write(f'route_{route_iron.source}.dot')
            os.system(f'dot -Tpng route_{route_iron.source}.dot > route_{route_iron.source}.png')

            for id_n in route_iron.nodes.keys():
                os.remove(f'{id_n}.png')

        except EmptyRoute:
            logger.warning(
                'While translating from Iron to PyDot-visualization format- an empty route was found: "None" returned')
            return None

    def create_dot_node(self, node: Node,
                        id_n: int,
                        dot_graph: pydot.Dot) -> None:
        """ To add a dot node for visualization"""
        node_instance = node.properties['node_type']
        self.create_node_picture(node_instance, id_n)
        dot_graph.add_node(pydot.Node(node.properties['node_unmapped_smiles'], image=f'{id_n}.png', label=''))

    @staticmethod
    def create_node_picture(node_instance: Union[Molecule, ChemicalEquation],
                            id_n: int) -> None:
        """ To generate the node picture """
        if type(node_instance) == ChemicalEquation:
            depiction_data = cid.draw_reaction(node_instance.rdrxn)

        else:
            depiction_data = cid.draw_molecule(node_instance.rdmol)

        lio.write_rdkit_depict(data=depiction_data, file_path=f'{id_n}.png')

    @staticmethod
    def create_dot_edge(edge: Edge,
                        route_iron: Iron,
                        dot_graph: pydot.Dot) -> None:
        """ To add an edge for visualization"""
        # the unmapped smiles is used to avoid the pydot issues in dealing with nodes' name with numbers
        a = next(n.properties['node_unmapped_smiles'] for i, n in route_iron.nodes.items() if i == edge.a_iid)
        b = next(n.properties['node_unmapped_smiles'] for i, n in route_iron.nodes.items() if i == edge.b_iid)
        dot_graph.add_edge(pydot.Edge(a, b))

    def to_iron(self, graph):
        pass


# Chain of responsibility structure to handle the translation input -> iron -> syngraph -> iron -> output
class Handler(ABC):
    """ Abstract handler for the concrete translators taking care of the single steps in the workflow """

    @abstractmethod
    def translate(self, input_format: str,
                  graph,
                  output_format: str,
                  out_data_model: str):
        pass


class InputToIron(Handler):
    """ Handler taking care of the first step: input to iron """

    def translate(self, input_format: str,
                  graph,
                  output_format: str,
                  out_data_model: str):
        factory = TranslatorFactory()
        # If the input graph is a syngraph
        if 'syngraph' in input_format:
            syngraphs = converter(graph, out_data_model)
            graph = SynGraphToIron().translate(input_format, syngraphs, output_format, out_data_model)
            return graph
        graph = factory.select_translation_to_iron(input_format, graph, out_data_model)
        if output_format == 'iron':
            graph = graph
        elif graph is None:
            return None
        else:
            graph = IronToSynGraph().translate(input_format, graph, output_format, out_data_model)
        return graph


class IronToSynGraph(Handler):
    """ Handler taking care of the second step: iron to syngraph in the desired data model """

    def translate(self, input_format, graph, output_format, out_data_model):
        factory = TranslatorFactory()
        graph = factory.select_translation_from_iron('syngraph', graph, out_data_model)
        if output_format == 'syngraph' or graph is None:
            return graph
        else:
            graph = SynGraphToIron().translate(input_format, graph, output_format, out_data_model)
        return graph


class SynGraphToIron(Handler):
    """ Handler taking care of the third step: syngraph to iron """

    def translate(self, input_format, graph, output_format, out_data_model):
        factory = TranslatorFactory()
        graph = factory.select_translation_to_iron('syngraph', graph, out_data_model)
        graph = IronToOutput().translate(input_format, graph, output_format, out_data_model)
        return graph


class IronToOutput(Handler):
    """ Handler taking care of the fourth step: iron to output """

    def translate(self, input_format, graph, output_format, out_data_model):
        factory = TranslatorFactory()
        graph = factory.select_translation_from_iron(output_format, graph, out_data_model)
        return graph


class Translation:
    """ Class to start the chain calling the handler of the first step """

    @staticmethod
    def start_translation(input_format, graph, output_format, out_data_model):
        return InputToIron().translate(input_format, graph, output_format, out_data_model)


# Facade function
def translator(input_format: str,
               original_graph: Union[MonopartiteReacSynGraph,
                                     MonopartiteMolSynGraph,
                                     BipartiteSynGraph,
                                     nx.classes.digraph.DiGraph,
                                     pydot.Dot,
                                     list,
                                     dict,
                                     Iron],
               output_format: str,
               out_data_model: str):
    """ Takes a graph object in an input format and translate it into a graph object in the desired output format
        and data model.

            :param:

                input_format: a string
                    It indicates the format of the input graph object

                original_graph: a graph object
                    It is the input graph

                output_format: a string
                    It indicates the desired output format

                out_data_model: a string
                    It indicates the desired type of output graph (monopartite, bipartite...)

            :return:
                out_graph: a graph object in the specified output format
    """
    if 'syngraph' in input_format and 'syngraph' in output_format:
        logger.error('To convert between data models, please use the "converter" function.')
        raise UnavailableTranslation

    translation = Translation()

    return translation.start_translation(input_format, original_graph, output_format, out_data_model)


# Helper functions
def get_available_formats():
    """ To retrieve all the available formats names and brief descriptions"""
    all_formats = TranslatorFactory.list_format_translators()
    d = {f: additional_info['info'] for f, additional_info in all_formats.items()}
    d.update({'iron': 'Iron object',
              'syngraph': 'Syngraph object'})
    return d


def get_available_data_models():
    """ To retrieve all the available data models names and brief descriptions"""
    all_datamodels = TranslatorFactory.list_datamodels_translators()
    return {f: additional_info['info'] for f, additional_info in all_datamodels.items()}


def get_output_formats():
    """ To retrieve names and information of the formats available as output formats """
    all_formats = TranslatorFactory.list_format_translators()
    available_output = {}
    for c in list(AbsTranslator.__subclasses__()):
        if c.as_output is not None:
            if name := next((f for f, info in all_formats.items() if info['class'] == c), None):
                available_output[name] = all_formats[name]['info']
    available_output['iron'] = 'Iron object'
    available_output['syngraph'] = 'SynGraph object'
    return available_output


def get_input_formats():
    """ To retrieve names and information of the formats available as input formats """
    all_formats = TranslatorFactory.list_format_translators()
    available_input = {}
    for c in list(AbsTranslator.__subclasses__()):
        if c.as_input is not None:
            if name := next((f for f, info in all_formats.items() if info['class'] == c), None):
                available_input[name] = all_formats[name]['info']
    available_input['iron'] = 'Iron object'
    available_input['syngraph'] = 'SynGraph object'
    return available_input


# Functions to handle json-like routes
def az_dict_to_iron(route: dict,
                    iron: Union[Iron, None],
                    parent: Union[int, None]) -> Iron:
    """ Takes a nested dictionary and returns an Iron instance. Recursive function.

            :param:
                route: a nested dictionary
                    The nested dictionary representing the input route

                properties: a dictionary
                    It contains the properties of the node to be stored

                iron: an Iron instance or None
                    The Iron instance that is built

                parent: an integer
                    It represents the id of the parent node

            :return:
                iron: an Iron instance
                    The output Iron instance
    """
    # During the first iteration, create Iron instance
    if not iron:
        iron = Iron()

    # if the node is a molecule node, go on and add node and edge to the Iron instance
    if route['type'] == 'mol':

        iron, id_n = populate_iron(parent, route['smiles'], iron)

        # If the node has 'children' nodes, iterate over them and call iteratively this function
        if 'children' in route:
            for child in route['children']:
                az_dict_to_iron(child, iron, id_n)

    # if the node is a reaction node, ignore it and proceed with the next level
    elif 'children' in route:
        for child in route['children']:
            az_dict_to_iron(child, iron, parent)

    return iron


def mit_dict_to_iron(route: dict,
                     iron: Union[Iron, None],
                     parent: Union[int, None]) -> Iron:
    """ Takes a nested dictionary and returns an Iron instance. Recursive function.

                :param:
                    route: a nested dictionary
                        The nested dictionary representing the input route

                    properties: a dictionary
                        It contains the properties of the node to be stored

                    iron: an Iron instance or None
                        The Iron instance that is built

                    parent: an integer
                        It represents the id of the parent node

                :return:
                    iron: an Iron instance
                        The output Iron instance
        """
    # During the first iteration, create Iron instance
    if not iron:
        iron = Iron()

    # if the node is a molecule node, go on and add node and edge to the Iron instance
    if 'is_chemical' in route:

        molecules = route['smiles'].split('.')

        for mol in molecules:

            iron, id_n = populate_iron(parent, mol, iron)

            # If the node has 'children' nodes, iterate over them and call iteratively this function
            if 'children' in route:
                for child in route['children']:
                    mit_dict_to_iron(child, iron, id_n)

    # if the node is a reaction node, ignore it and proceed with the next level
    elif 'children' in route:
        for child in route['children']:
            mit_dict_to_iron(child, iron, parent)

    return iron


def ibm_dict_to_iron(route: dict,
                     iron: Union[Iron, None],
                     parent: Union[int, None]) -> Iron:
    """ Takes a nested dictionary and returns an Iron instance. Recursive function.

            :param:
                route: a nested dictionary
                    The nested dictionary representing the input route

                properties: a dictionary
                    It contains the properties of the node to be stored

                iron: an Iron instance or None
                    The Iron instance that is built

                parent: an integer
                    It represents the id of the parent node

            :return:
                iron: an Iron instance
                    The output Iron instance
    """
    # During the first iteration, create Iron instance
    if not iron:
        iron = Iron()

    if 'artifact' not in route['metaData']:

        molecules = route['smiles'].split('.')

        for mol in molecules:

            iron, id_n = populate_iron(parent, mol, iron)

            # If the node has 'children' nodes, iterate over them and call iteratively this function
            if 'children' in route:
                for child in route['children']:
                    ibm_dict_to_iron(child, iron, id_n)
    return iron


def populate_iron(parent: int,
                  mol: str,
                  iron: Iron) -> Tuple[Iron, int]:
    """ To add new entities to an Iron instance

        :param:
            parent: an integer representing the id of the parent node

            mol: a string representing the smiles of the new node

            iron: the Iron object to which the new node should be added

        :return:
            a tuple containing the Iron object with the new node and the integer representing the id of the new node
    """
    # Count the number of nodes already present in the Iron structure and use it as id for the node
    id_n = iron.i_node_number()

    # Store additional information related to the node into the Iron node properties
    prop = {'node_smiles': mol}

    # Create node and add it to the Iron structure
    node = Node(properties=prop, iid=str(id_n), labels=[])
    iron.add_node(str(id_n), node)
    # If the node is a "child" node, create edge from it to the parent node
    # The edge direction follows the chemical reaction direction: from reagent to product
    if parent is not None:
        id_e = iron.i_edge_number()
        edge = build_iron_edge(id_n, parent, id_e)
        iron.add_edge(str(id_e), edge)
    return iron, id_n


def build_iron_edge(id_n1: int,
                    id_n2: int,
                    id_e: int) -> Edge:
    """ To build an edge object for an Iron instance from the ids of the involved nodes

        :param:
            id_n1: an integer representing the id of the source node

            id_n2: an integer representing the id of the target node

            id_e: an integer representing the id of the output edge

        :return:
            an iron.Edge object
    """
    d = Direction(f'{id_n1}>{id_n2}')
    return Edge(iid=str(id_e),
                a_iid=str(id_n1),
                b_iid=str(id_n2),
                direction=d,
                properties={},
                labels=[])
