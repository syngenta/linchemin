import datetime
import os
from abc import ABC, abstractmethod
from typing import List, Union

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


class TranslationError(Exception):
    """ Base class for exceptions leading to unsuccessful translation."""
    pass


class EmptyRoute(TranslationError):
    """ Raised if an empty route is found """
    pass


class InvalidRoute(TranslationError):
    """ Raised if the route does not contain at least two molecules connected by an edge"""


class UnavailableInputFormat(TranslationError):
    """ Raised if the selected input format is not among the available ones"""
    pass


class UnavailableOutputFormat(TranslationError):
    """ Raised if the selected output format is not among the available ones """
    pass


class UnavailableTranslation(TranslationError):
    """ Raised if the required translation cannot be performed """
    pass


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
    def from_iron(self, graph: Iron):
        """ Translates an Iron instance into a graph object of another type.

            :param:
                graph: an Iron instance
                    It is the graph of interest as an Ion instance

            :return:
                The same graph as another object type
        """
        pass

    @abstractmethod
    def to_iron(self, graph) -> Union[Iron, None]:
        """ Translates a graph object of a specific type into an Iron instance.

            :param:
                graph: a graph object os a specific type

            :return:
                The same graph as Iron instance

        """
        pass


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
            id_n = 0
            id_e = 0
            for reac, connections in mp_syngraph.graph.items():
                # if the smiles 'reac' is not yet among the nodes of the Iron instance,
                # the corresponding node is created and added to Iron
                if reac not in [n.properties['node_class'] for id, n in iron.nodes.items()]:
                    # the unmapped smiles is built so that the route is suitable to be correctly displayed in a png file
                    unmapped_smiles = cif.rdrxn_to_string(reac.rdrxn, out_fmt='smiles', use_atom_mapping=False)
                    prop = {
                        'node_unmapped_smiles': unmapped_smiles,
                        'node_smiles': reac.smiles,
                        'node_class': reac}
                    node1 = [(id_n, Node(iid=str(id_n), properties=prop, labels=[]))]
                    iron.add_node(str(node1[0][0]), node1[0][1])
                    id_n += 1
                else:
                    # if the smiles 'reac' is already included in Iron,
                    # the relative information are retrieved
                    node1 = [(id, n) for id, n in iron.nodes.items() if n.properties['node_class'] == reac]
                for c in connections:
                    if c.smiles not in [n.properties['node_smiles'] for id, n in iron.nodes.items()]:
                        unmapped_smiles = cif.rdrxn_to_string(c.rdrxn, out_fmt='smiles', use_atom_mapping=False)
                        prop = {
                            'node_unmapped_smiles': unmapped_smiles,
                            'node_smiles': c.smiles,
                            'node_class': c}
                        node2 = [(id_n, Node(iid=str(id_n), properties=prop, labels=[]))]
                        iron.add_node(str(node2[0][0]), node2[0][1])
                        id_n += 1
                    else:
                        node2 = [(id, n) for id, n in iron.nodes.items() if n.properties['node_class'] == c]

                    d = Direction(f'{str(node1[0][0])}>{str(node2[0][0])}')
                    e = Edge(iid=str(id_e), a_iid=str(node1[0][0]), b_iid=str(node2[0][0]), direction=d,
                             properties={},
                             labels=[])
                    iron.add_edge(str(id_e), e)
                    id_e += 1
            iron.source = mp_syngraph.source
            return iron
        except EmptyRoute:
            logger.warning(
                'While translating from a monopartite-reactions SynGraph to Iron an empty route was found: "None" '
                'returned')
            return None


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
            id_n = 0
            id_e = 0
            for reac, connections in syngraph.graph.items():
                # if the smiles 'reac' is not yet among the nodes of the Iron instance,
                # the corresponding node is created and added to Iron
                if reac.smiles not in [n.properties['node_smiles'] for id, n in iron.nodes.items()]:
                    if type(reac) == ChemicalEquation:
                        unmapped_smiles = cif.rdrxn_to_string(reac.rdrxn, out_fmt='smiles', use_atom_mapping=False)
                    elif type(reac) == Molecule:
                        unmapped_smiles = reac.smiles

                    prop = {
                        'node_unmapped_smiles': unmapped_smiles,
                        'node_smiles': reac.smiles,
                        'node_class': reac
                    }

                    node1 = [(id_n, Node(iid=str(id_n), properties=prop, labels=[]))]
                    iron.add_node(str(node1[0][0]), node1[0][1])
                    id_n += 1
                else:
                    # if the smiles 'reac' is already included in Iron,
                    # the relative information are retrieved
                    node1 = [(id, n) for id, n in iron.nodes.items() if n.properties['node_smiles'] == reac.smiles]
                for c in connections:
                    if c.smiles not in [n.properties['node_smiles'] for id, n in iron.nodes.items()]:
                        if type(c) == ChemicalEquation:
                            unmapped_smiles = cif.rdrxn_to_string(c.rdrxn, out_fmt='smiles', use_atom_mapping=False)
                        elif type(c) == Molecule:
                            unmapped_smiles = c.smiles

                        prop = {
                            'node_unmapped_smiles': unmapped_smiles,
                            'node_smiles': c.smiles,
                            'node_class': c
                        }
                        node2 = [(id_n, Node(iid=str(id_n), properties=prop, labels=[]))]
                        iron.add_node(str(node2[0][0]), node2[0][1])
                        id_n += 1
                    else:
                        node2 = [(id, n) for id, n in iron.nodes.items() if n.properties['node_smiles'] == c.smiles]

                    d = Direction(f'{str(node1[0][0])}>{str(node2[0][0])}')
                    e = Edge(iid=str(id_e), a_iid=str(node1[0][0]), b_iid=str(node2[0][0]), direction=d,
                             properties={}, labels=[])
                    iron.add_edge(str(id_e), e)
                    id_e += 1

            iron.source = syngraph.source
            return iron
        except EmptyRoute:
            logger.warning(
                'While translating from a bipartite SynGraph to Iron an empty route was found: "None" returned')
            return None


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
            id_n = 0
            id_e = 0
            for reac, connections in mp_syngraph.graph.items():
                if reac not in [n.properties['node_class'] for id, n in iron.nodes.items()]:
                    # if the smiles 'reac' is not yet among the nodes of the Iron instance,
                    # the corresponding node is created and added to Iron
                    prop = {'node_smiles': reac.smiles,
                            'node_class': reac}
                    node1 = [(id_n, Node(iid=str(id_n), properties=prop, labels=[]))]
                    iron.add_node(str(node1[0][0]), node1[0][1])
                    id_n += 1
                else:
                    # if the smiles 'reac' is already included in Iron,
                    # the relative information are retrieved
                    node1 = [(id, n) for id, n in iron.nodes.items() if n.properties['node_class'] == reac]
                for c in connections:
                    if c.smiles not in [n.properties['node_smiles'] for id, n in iron.nodes.items()]:
                        prop = {'node_smiles': c.smiles,
                                'node_class': c}
                        node2 = [(id_n, Node(iid=str(id_n), properties=prop, labels=[]))]
                        iron.add_node(str(node2[0][0]), node2[0][1])
                        id_n += 1
                    else:
                        node2 = [(id, n) for id, n in iron.nodes.items() if n.properties['node_class'] == c]

                    d = Direction(f'{str(node1[0][0])}>{str(node2[0][0])}')
                    e = Edge(iid=str(id_e), a_iid=str(node1[0][0]), b_iid=str(node2[0][0]), direction=d,
                             properties={},
                             labels=[])
                    iron.add_edge(str(id_e), e)
                    id_e += 1
            iron.source = mp_syngraph.source
            return iron
        except EmptyRoute:
            logger.warning(
                'While translating from a monopartite-molecules SynGraph to Iron an empty route was found: "None" '
                'returned')
            return None


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
                nx_graph = nx.DiGraph()
                nx_graph.graph['source'] = route_iron.source
                for id, node in route_iron.nodes.items():
                    nx_graph.add_node(node.properties['node_smiles'])
                    attrs_n = {node.properties['node_smiles']: {'properties': node.properties, 'labels': node.labels,
                                                                'source': route_iron.source}}

                    nx.set_node_attributes(nx_graph, attrs_n, 'attributes')
            else:
                nx_graph = nx.DiGraph()
                nx_graph.graph['source'] = route_iron.source
                # Translating iron edges in networkx edges; nodes are automatically added
                for id_e, edge in route_iron.edges.items():
                    a = [n.properties['node_smiles'] for id, n in route_iron.nodes.items() if id == edge.a_iid]
                    b = [n.properties['node_smiles'] for id, n in route_iron.nodes.items() if id == edge.b_iid]
                    nx_graph.add_edge(a[0], b[0])
                    attrs_e = {
                        (a[0], b[0]):
                            {'direction': edge.direction.string, 'properties': edge.properties,
                             'labels': edge.labels}
                    }
                    nx.set_edge_attributes(nx_graph, attrs_e, 'attributes')
                for id_n, node in route_iron.nodes.items():
                    attrs_n = {node.properties['node_smiles']: {'properties': node.properties, 'labels': node.labels,
                                                                'source': route_iron.source}}

                    nx.set_node_attributes(nx_graph, attrs_n, 'attributes')
            return nx_graph
        except EmptyRoute:
            logger.warning('While translating from Iron to NetworkX object an empty route was found: "None" returned')
            return None

    def to_iron(self, route: nx.classes.digraph.DiGraph) -> Union[Iron, None]:
        """ Translates a Networkx object into an Iron instance """
        iron = Iron()

        for id_n, (node, data) in enumerate(route.nodes.items()):
            iron_node = Node(iid=str(id_n), properties={'node_smiles': node}, labels=[])
            iron.add_node(str(id_n), iron_node)
        id_e = 0
        for edge, data in route.edges.items():
            a = edge[0]
            a_id = [n.iid for n in iron.nodes.values() if n.properties['node_smiles'] == a][0]
            b = edge[1]
            b_id = [n.iid for n in iron.nodes.values() if n.properties['node_smiles'] == b][0]
            direc = Direction(f'{str(a_id)}>{str(b_id)}')
            iron_edge = Edge(iid=str(id_e), a_iid=str(a_id), b_iid=str(b_id), direction=direc, properties={}, labels=[])
            iron.add_edge(id_e, iron_edge)
            id_e = id_e + 1

        return iron


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
                a = [n.properties['node_smiles'] for id, n in route_iron.nodes.items() if id == edge.a_iid]
                b = [n.properties['node_smiles'] for id, n in route_iron.nodes.items() if id == edge.b_iid]
                dot_graph.add_edge(pydot.Edge(a[0], b[0]))
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
            a = edge.get_source().strip('\"')
            a_id = [n.iid for n in iron.nodes.values() if n.properties['node_smiles'] == a][0]
            b = edge.get_destination().strip('\"')
            b_id = [n.iid for n in iron.nodes.values() if n.properties['node_smiles'] == b][0]
            direc = Direction(f'{a_id}>{b_id}')
            iron_edge = Edge(iid=str(id_e), a_iid=str(a_id), b_iid=str(b_id), direction=direc, properties={}, labels=[])
            iron.add_edge(id_e, iron_edge)
            id_e = id_e + 1
        return iron


class TranslatorIbm(AbsTranslator):
    """ Translator subclass to handle translations of outputs generated by IBM CASP tool """
    as_input = 'implemented'
    as_output = None

    def from_iron(self, graph: Iron) -> Union[list, None]:
        pass

    def to_iron(self, route: dict) -> Union[Iron, None]:
        """ Translates a dictionary generated by the IBM CASP tool into an Iron instance"""
        prop = {'node_smiles': 'smiles'}
        try:
            if not route:
                raise EmptyRoute
            iron_graph = ibm_dict_to_iron(route, prop, iron=None, parent=None)

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


class TranslatorAz(AbsTranslator):
    """ Translator subclass to handle translations of outputs generated by AZ CASP tool """
    as_input = 'implemented'
    as_output = None

    def from_iron(self, graph: Iron):
        pass

    def to_iron(self, route: dict) -> Union[Iron, None]:
        """ Translates a dictionary generated by the AZ CASP tool into an Iron instance """
        prop = {'node_smiles': 'smiles'}  # properties mapping: {iron_term: casp_term}
        try:
            if not route:
                raise EmptyRoute
            iron_graph = az_dict_to_iron(route, prop, iron=None, parent=None)
            iron_graph.source = 'az_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            return iron_graph
        except EmptyRoute:
            logger.warning('While translating from AZ to Iron an empty route was found: "None" returned')
            return None


class TranslatorMit(AbsTranslator):
    """ Translator subclass to handle translations of outputs generated by Askcos CASP tool """
    as_input = 'implemented'
    as_output = None

    def from_iron(self, graph: Iron):
        pass

    def to_iron(self, route: dict) -> Union[Iron, None]:
        """ Translates a dictionary generated by the Askcos CASP tool into an Iron instance """
        prop = {'node_smiles': 'smiles'}  # properties mapping: {iron_term: casp_term}
        try:
            if not route:
                raise EmptyRoute
            iron_graph = mit_dict_to_iron(route, prop, iron=None, parent=None)
            iron_graph.source = 'mit_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            return iron_graph
        except EmptyRoute:
            logger.warning('While translating from Asckos to Iron an empty route was found: "None" returned')
            return None


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
                node_instance = n.properties.get('node_class')
                node_dict = node_instance.to_dict()
                node_document = {'type': node_dict.get('type'), 'uid': node_dict.get('uid'),
                                 'properties': {**{'smiles': node_dict.get('smiles')}, **node_dict.get('hash_map')}}
                if type(node_instance) == ChemicalEquation and node_instance.disconnection is not None:
                    node_document['disconnection_uid'] = node_instance.disconnection.uid
                node_documents.append(node_document)

                if type(node_instance) is ChemicalEquation:
                    # append role edges
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
            return {'nodes': node_documents, 'edges': edge_documents}
        except EmptyRoute:
            logger.warning('While translating from Iron to NOC format an empty route was found: "None" returned')
            return None

    def to_iron(self, graph: dict) -> Union[Iron, None]:
        pass


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
                node_instance = node.properties['node_class']

                if type(node_instance) == ChemicalEquation:
                    depiction_data = cid.draw_reaction(node_instance.rdrxn)

                else:
                    depiction_data = cid.draw_molecule(node_instance.rdmol)

                lio.write_rdkit_depict(data=depiction_data, file_path=f'{id_n}.png')

                dot_graph.add_node(pydot.Node(node.properties['node_unmapped_smiles'], image=f'{id_n}.png', label=''))

            # Translating iron edges into dot edges
            for id_e, edge in route_iron.edges.items():
                # the unmapped smiles is used to avoid the pydot issues in dealing with nodes' name with numbers
                a = [n.properties['node_unmapped_smiles'] for id, n in route_iron.nodes.items() if id == edge.a_iid]
                b = [n.properties['node_unmapped_smiles'] for id, n in route_iron.nodes.items() if id == edge.b_iid]
                dot_graph.add_edge(pydot.Edge(a[0], b[0]))

            dot_graph.write(f'route_{route_iron.source}.dot')
            os.system(f'dot -Tpng route_{route_iron.source}.dot > route_{route_iron.source}.png')

            for id_n in route_iron.nodes.keys():
                os.remove(f'{id_n}.png')

        except EmptyRoute:
            logger.warning(
                'While translating from Iron to PyDot-visualization format- an empty route was found: "None" returned')
            return None

    def to_iron(self, graph):
        pass


class TranslatorFactory:
    """ Translator Factory to give access to the translations.

        Attributes:
            translators: a dictionary
                It maps the strings representing the 'name' of a format to the correct AbsTranslator subclass

            data_models: a dictionary
                It maps the strings representing the 'name' of a data model to the correct SynGraph subclass

    """
    translators = {
        'networkx': {'value': TranslatorNetworkx, 'info': 'Translating from/into Networkx objects'},
        'pydot': {'value': TranslatorDot, 'info': 'Translating from/into Pydot objects'},
        'ibm_retro': {'value': TranslatorIbm, 'info': 'Translating from outputs generated by the IBMRXN CASP tool'},
        'az_retro': {'value': TranslatorAz,
                     'info': 'Translating from outputs generated by the AiZynthFinder CASP tool'},
        'mit_retro': {'value': TranslatorMit,
                      'info': 'Translating from outputs generated by the Askos CASP tool'},
        'syngraph': {'value': None,
                     'info': 'Translating from/into SynGraph objects'},
        'pydot_visualization': {'value': TranslatorDotVisualization,
                                'info': 'Creates PNG pictures of the input routes'},
        'iron': {'value': None, 'info': 'Translating from/into a list of Iron objects'},
        'noc': {'value': TranslatorNOC,
                'info': 'Translating to a format compatible with a Network of Organic Chemistry'},
    }

    data_models = {
        'bipartite': {'value': TranslatorBipartiteSynGraph,
                      'info': 'A bipartite graph (molecule and chemical reaction nodes)'},
        'monopartite_reactions': {'value': TranslatorMonopartiteReacSynGraph,
                                  'info': 'A monopartite graph (chemical reaction nodes only)'},
        'monopartite_molecules': {'value': TranslatorMonopartiteMolSynGraph,
                                  'info': 'A monopartite graph (molecules nodes only)'},
    }

    def select_translation_to_iron(self, input_format: str, graph, out_data_model: str):
        """ Calls the correct translator to transform the selected input graph object into an Iron instance """
        if input_format not in self.translators:
            logger.error(
                f"'{input_format}' is not a valid input format. Possible formats are: {self.translators.keys()}")
            raise UnavailableInputFormat

        elif input_format == 'iron':
            return graph

        if input_format == 'syngraph':
            in_translator = self.data_models[out_data_model]['value']
        else:
            in_translator = self.translators[input_format]['value']
        return in_translator().to_iron(graph)

    def select_translation_from_iron(self, out_format: str, iron_graph, out_data_model: str):
        """ Calls the correct translator to transform an Iron instance into the selected output graph object """
        if out_format not in self.translators:
            logger.error(
                f"'{out_format}' is not a valid output format. Possible formats are: {self.translators.keys()}")
            raise UnavailableOutputFormat

        elif out_format == 'iron':
            # if the output format is iron, the same iron graph is returned
            return iron_graph

        if out_format == 'syngraph':
            out_translator = self.data_models[out_data_model]['value']
        else:
            out_translator = self.translators[out_format]['value']
        return out_translator().from_iron(iron_graph)


# Chain of responsibility structure to handle the translation input -> iron -> syngraph -> iron -> output
class Handler(ABC):
    """ Abstract handler for the concrete translators taking care of the single steps in the workflow """

    @abstractmethod
    def translate(self, input_format: str, graph, output_format: str, out_data_model: str):
        pass


class InputToIron(Handler):
    """ Handler taking care of the first step: input to iron """

    def translate(self, input_format: str, graph, output_format: str, out_data_model: str):
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
def translator(input_format: str, original_graph, output_format: str, out_data_model: str):
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


def get_available_formats():
    return {f: additional_info['info'] for f, additional_info in TranslatorFactory.translators.items()}


def get_available_data_models():
    return {f: additional_info['info'] for f, additional_info in TranslatorFactory.data_models.items()}


def get_output_formats():
    available_output = {}
    for c in list(AbsTranslator.__subclasses__()):
        if c.as_output is not None:
            if name := [f for f, info in TranslatorFactory.translators.items() if info['value'] == c]:
                available_output[name[0]] = TranslatorFactory.translators[name[0]]['info']
    available_output['iron'] = TranslatorFactory.translators['iron']['info']
    available_output['syngraph'] = TranslatorFactory.translators['syngraph']['info']
    return available_output


def get_input_formats():
    available_input = {}
    for c in list(AbsTranslator.__subclasses__()):
        if c.as_input is not None:
            if name := [f for f, info in TranslatorFactory.translators.items() if info['value'] == c]:
                available_input[name[0]] = TranslatorFactory.translators[name[0]]['info']
    available_input['iron'] = TranslatorFactory.translators['iron']['info']
    available_input['syngraph'] = TranslatorFactory.translators['syngraph']['info']
    return available_input


def az_dict_to_iron(route: dict, properties: dict, iron: Union[Iron, None], parent: Union[int, None]) -> Iron:
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

        # Count the number of nodes already present in the Iron structure and use it as id for the node
        id_n = iron.i_node_number()

        # Store additional information related to the node into the Iron node properties
        prop = {iron_term: route[casp_term] for iron_term, casp_term in properties.items()}

        # Create node and add it to the Iron structure
        node = Node(properties=prop, iid=str(id_n), labels=[])
        iron.add_node(str(id_n), node)
        # If the node is a "child" node, create edge from it to the parent node
        # The edge direction follows the chemical reaction direction: from reagent to product
        if parent is not None:
            id_e = iron.i_edge_number()
            direc = Direction(f'{str(id_n)}>{parent}')
            edge = Edge(iid=str(id_e), a_iid=node.iid, b_iid=str(parent), direction=direc, properties={})
            iron.add_edge(str(id_e), edge)

        # If the node has 'children' nodes, iterate over them and call iteratively this function
        if 'children' in route:
            for child in route['children']:
                az_dict_to_iron(child, properties, iron, id_n)

    # if the node is a reaction node, ignore it and proceed with the next level
    elif 'children' in route:
        for child in route['children']:
            az_dict_to_iron(child, properties, iron, parent)

    return iron


def mit_dict_to_iron(route: dict, properties: dict, iron: Union[Iron, None], parent: Union[int, None]) -> Iron:
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
                direc = Direction(f'{str(id_n)}>{parent}')
                edge = Edge(iid=str(id_e), a_iid=node.iid, b_iid=str(parent), direction=direc, properties={})
                iron.add_edge(str(id_e), edge)

            # If the node has 'children' nodes, iterate over them and call iteratively this function
            if 'children' in route:
                for child in route['children']:
                    mit_dict_to_iron(child, properties, iron, id_n)

    # if the node is a reaction node, ignore it and proceed with the next level
    elif 'children' in route:
        for child in route['children']:
            mit_dict_to_iron(child, properties, iron, parent)

    return iron


def ibm_dict_to_iron(route: dict, properties: dict, iron: Union[Iron, None], parent: Union[int, None]) -> Iron:
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
                direc = Direction(f'{str(id_n)}>{parent}')
                edge = Edge(iid=str(id_e), a_iid=node.iid, b_iid=str(parent), direction=direc, properties={})
                iron.add_edge(str(id_e), edge)

            # If the node has 'children' nodes, iterate over them and call iteratively this function
            if 'children' in route:
                for child in route['children']:
                    ibm_dict_to_iron(child, properties, iron, id_n)
    return iron
