import datetime
import os
from abc import ABC, abstractmethod
from typing import List, Type, Union
from linchemin.cgu.iron import Edge, Iron, Node
import networkx as nx
import pydot
from linchemin.utilities import console_logger
from linchemin.cgu.graph_transformations.supporting_functions import (
    az_dict_to_iron,
    ibm_dict_to_iron,
    mit_dict_to_iron,
    build_iron_edge,
)
from linchemin import settings
import linchemin.cheminfo.depiction as cid
from linchemin.IO import io as lio
from linchemin.cheminfo.models import Molecule, ChemicalEquation
import linchemin.cgu.graph_transformations.exceptions as exceptions

logger = console_logger(__name__)


class GraphFormatTranslator(ABC):
    """
    Abstract representation of translators handling graph formats.

    Attributes:
    ------------
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
    def from_iron(
        self, graph: Iron
    ) -> Union[nx.classes.digraph.DiGraph, pydot.Dot, list, dict, None]:
        """
        To translate an Iron instance into a graph object of another type.

        Parameters:
        -----------
        graph: Iron
            The input Iron instance

        Returns:
        --------
        output graph: Union[nx.classes.digraph.DiGraph, pydot.Dot, list, dict, None]
            The input graph in another format selected by the user
        """
        pass

    @abstractmethod
    def to_iron(
        self,
        graph: Union[
            nx.classes.digraph.DiGraph,
            pydot.Dot,
            list,
            dict,
            None,
        ],
    ) -> Union[Iron, None]:
        """
        To translate a graph object of a specific type into an Iron instance.

        Parameters:
        -----------
        graph: Union[nx.classes.digraph.DiGraph, pydot.Dot, list, dict, None]
            The input graph to be translated into an Iron object

        Returns:
        --------
        iron_graph: Iron
            The input graph as Iron object
        """
        pass


class GraphFormatCatalog:
    """Class to store the available GraphFormatTranslators"""

    _registered_graph_formats = {}

    @classmethod
    def register_format(cls, name: str, info: str):
        """
        Decorator for registering a new format translator.

        Parameters:
        ------------
        name: str
            The name of the translator to be used as a key in the registry
        info: str
            A brief description of the translator

        Returns:
        ---------
        function: The decorator function.
        """

        def decorator(format_translator_class: Type[GraphFormatTranslator]):
            cls._registered_graph_formats[name.lower()] = {
                "class": format_translator_class,
                "info": info,
            }
            return format_translator_class

        return decorator

    @classmethod
    def get_format(cls, name: str) -> GraphFormatTranslator:
        """
        To get an instance of the specified AbsTranslator.

        Parameters:
        ------------
        name: str
            The name of the format translator.

        Returns:
        ---------
        AbsTranslator: An instance of the specified format translator.

        Raises:
        -------
        KeyError: If the specified format is not registered.
        """
        format_translator = cls._registered_graph_formats.get(name.lower())
        if format_translator is None:
            logger.error(f"Format '{name}' not found")
            raise exceptions.UnavailableFormat
        return format_translator["class"]()

    @classmethod
    def list_formats(cls):
        """
        To list the names of all available data model factories.

        Returns:
        ---------
        formats: dict
            The names and information of the available data models.
        """
        return cls._registered_graph_formats


@GraphFormatCatalog.register_format("reaxys", "Reaxys Retrosynthesis output")
class ReaxysRT(GraphFormatTranslator):
    as_input = "implemented"
    as_output = None

    def from_iron(self, graph: Iron) -> Union[dict, None]:
        pass

    def to_iron(
        self,
        graph: Union[dict, None],
    ) -> Union[Iron, None]:
        """Translates the output of Reaxys' CASP into an Iron instance"""
        iron = Iron()
        steps = graph["rxspm:hasStep"]
        for step in steps:
            products_smiles = [
                mol["rxspm:hasSubstance"]["edm:smiles"]
                for mol in step["rxspm:hasReaction"]["rxspm:hasProduct"]
            ]
            products_ids = self.handle_molecules(products_smiles, iron)
            starting_materials = [
                mol["rxspm:hasSubstance"]["edm:smiles"]
                for mol in step["rxspm:hasReaction"]["rxspm:hasStartingMaterial"]
            ]
            reactants_ids = self.handle_molecules(starting_materials, iron)
            for i in products_ids:
                self.add_iron_edge(iron, i, reactants_ids)

        return iron

    def handle_molecules(self, molecules: list, iron: Iron) -> list:
        """To get the iron's id of the nodes corresponding to the input molecules' smiles"""
        iids = []
        for mol in molecules:
            if i := next(
                (
                    iid
                    for iid, node in iron.nodes.items()
                    if node.properties["node_smiles"] == mol
                ),
                None,
            ):
                iids.append(i)
            else:
                id_n = len(iron.nodes)
                self.add_iron_node(mol, id_n, iron)
                iids.append(id_n)
        return iids

    @staticmethod
    def add_iron_node(smiles: str, iid: int, iron: Iron) -> None:
        """To add a new node to an Iron instance"""
        iron_node = Node(iid=str(iid), properties={"node_smiles": smiles}, labels=[])
        iron.add_node(str(iid), iron_node)

    @staticmethod
    def add_iron_edge(iron: Iron, i: int, reactants_ids: List[int]) -> None:
        """To add a new edge to an Iron instance"""
        edges = []
        id_e = len(iron.edges)
        for sm in reactants_ids:
            edges.append(build_iron_edge(sm, i, id_e))
            id_e += 1
        [iron.add_edge(iron_edge.iid, iron_edge) for iron_edge in edges]


@GraphFormatCatalog.register_format("networkx", "Networkx DiGraph object")
class Networkx(GraphFormatTranslator):
    """Translator subclass to handle translations into and from Networkx objects"""

    as_input = "implemented"
    as_output = "implemented"

    def from_iron(self, route_iron: Iron) -> Union[nx.classes.digraph.DiGraph, None]:
        """Translates an Iron instance into a Networkx object"""
        try:
            if route_iron is None:
                raise exceptions.EmptyRoute
            elif route_iron.i_edge_number() == 0:
                # graph with no edges
                nx_graph = self.build_single_node_nx(route_iron)

            else:
                # graph with at least one edge
                nx_graph = self.build_multi_nodes_nx(route_iron)

            return nx_graph
        except exceptions.EmptyRoute:
            logger.warning(
                'While translating from Iron to NetworkX object an empty route was found: "None" returned'
            )
            return None

    def build_multi_nodes_nx(self, route_iron: Iron) -> nx.DiGraph:
        """To build a networkx GiGraph with multiple nodes connected by edges"""
        nx_graph = nx.DiGraph()
        nx_graph.graph["source"] = route_iron.source
        self.add_nx_nodes_edges(nx_graph, route_iron)
        self.add_nx_nodes_attributes(nx_graph, route_iron)

        return nx_graph

    @staticmethod
    def add_nx_nodes_edges(nx_graph: nx.DiGraph, route_iron: Iron) -> None:
        """To add nodes and edges to a networkx DiGraph instance"""
        for id_e, edge in route_iron.edges.items():
            a = next(
                n.properties["node_smiles"]
                for i, n in route_iron.nodes.items()
                if i == edge.a_iid
            )
            b = next(
                n.properties["node_smiles"]
                for i, n in route_iron.nodes.items()
                if i == edge.b_iid
            )
            nx_graph.add_edge(a, b)

    def add_nx_nodes_attributes(self, nx_graph: nx.DiGraph, route_iron: Iron) -> None:
        """To add attributes to the nodes in the networkx DiGraph instance"""

        for node in route_iron.nodes.values():
            if node_instance := node.properties.get("node_type", None):
                if isinstance(node_instance, Molecule):
                    attrs_n = {
                        node.properties["node_smiles"]: {
                            "properties": node.properties,
                            "source": route_iron.source,
                            "label": settings.ROUTE_MINING.molecule_node_label,
                            "uid": node_instance.uid,
                        }
                    }
                elif isinstance(node_instance, ChemicalEquation):
                    attrs_n = {
                        node.properties["node_smiles"]: {
                            "properties": node.properties,
                            "source": route_iron.source,
                            "label": settings.ROUTE_MINING.chemicalequation_node_label,
                            "uid": node_instance.uid,
                        }
                    }
                    self.set_nx_edges_labels(node_instance, nx_graph)
            else:
                attrs_n = {
                    node.properties["node_smiles"]: {
                        "properties": node.properties,
                        "source": route_iron.source,
                        "label": node.labels,
                    }
                }

            nx.set_node_attributes(
                nx_graph,
                attrs_n,
            )

    @staticmethod
    def set_nx_edges_labels(node_instance: ChemicalEquation, nx_graph: nx.DiGraph):
        """To assign labels to the networkx edges based on chemical roles"""
        role_label_map = {
            "reactants": settings.ROUTE_MINING.reactant_edge_label,
            "reagents": settings.ROUTE_MINING.reagent_edge_label,
            "products": settings.ROUTE_MINING.product_edge_label,
        }
        for role, label in role_label_map.items():
            for h in node_instance.role_map[role]:
                mol_smiles = [
                    m.smiles for uid, m in node_instance.catalog.items() if uid == h
                ]
                if role == "products":
                    nx.set_edge_attributes(
                        nx_graph,
                        {
                            (node_instance.smiles, smiles): {"label": label}
                            for smiles in mol_smiles
                        },
                    )
                else:
                    nx.set_edge_attributes(
                        nx_graph,
                        {
                            (smiles, node_instance.smiles): {"label": label}
                            for smiles in mol_smiles
                        },
                    )

    def build_single_node_nx(self, route_iron: Iron) -> nx.classes.digraph.DiGraph:
        """To create a networky network with only isolated nodes"""
        nx_graph = nx.DiGraph()
        nx_graph.graph["source"] = route_iron.source
        for node in route_iron.nodes.values():
            nx_graph.add_node(node.properties["node_smiles"])
        self.add_nx_nodes_attributes(nx_graph, route_iron)
        return nx_graph

    def to_iron(self, route: nx.classes.digraph.DiGraph) -> Union[Iron, None]:
        """Translates a Networkx object into an Iron instance"""
        iron = Iron()

        for id_n, (node, data) in enumerate(route.nodes.items()):
            iron_node = Node(iid=str(id_n), properties={"node_smiles": node}, labels=[])
            iron.add_node(str(id_n), iron_node)
        for id_e, (edge, data) in enumerate(route.edges.items()):
            iron_edge = self.nx_edge_to_iron_edge(edge, id_e, iron)
            iron.add_edge(str(id_e), iron_edge)

        return iron

    @staticmethod
    def nx_edge_to_iron_edge(edge, id_e: str, iron: Iron) -> Edge:
        """To transform a nx edge into an Iron edge"""
        a_id = next(
            n.iid for n in iron.nodes.values() if n.properties["node_smiles"] == edge[0]
        )
        b_id = next(
            n.iid for n in iron.nodes.values() if n.properties["node_smiles"] == edge[1]
        )
        return build_iron_edge(a_id, b_id, id_e)


@GraphFormatCatalog.register_format("pydot", "PyDot object")
class PyDot(GraphFormatTranslator):
    """Translator subclass to handle translations into and from Dot objects"""

    as_input = "implemented"
    as_output = "implemented"

    def from_iron(self, route_iron: Iron) -> Union[pydot.Dot, None]:
        """Translates an Iron instance into a Pydot object"""

        try:
            if route_iron is None:
                raise exceptions.EmptyRoute
            dot_graph = pydot.Dot(route_iron.source, graph_type="digraph")
            # Translating iron nodes into dot nodes
            for id_n, node in route_iron.nodes.items():
                # NB Sometimes Pydot adds unwanted double quotes to the string name of the node
                # related issue: https://github.com/pydot/pydot/issues/72
                dot_graph.add_node(pydot.Node(node.properties["node_smiles"]))

            # Translating iron edges into dot edges
            for id_e, edge in route_iron.edges.items():
                a = next(
                    n.properties["node_smiles"]
                    for i, n in route_iron.nodes.items()
                    if i == edge.a_iid
                )
                b = next(
                    n.properties["node_smiles"]
                    for i, n in route_iron.nodes.items()
                    if i == edge.b_iid
                )
                dot_graph.add_edge(pydot.Edge(a, b))
            return dot_graph
        except exceptions.EmptyRoute:
            logger.warning(
                'While translating from Iron to PyDot object an empty route was found: "None" returned'
            )
            return None

    def to_iron(self, route: pydot.Dot) -> Union[Iron, None]:
        """Translates a Pydot object into an Iron instance"""

        iron = Iron()

        for id_n, node in enumerate(route.get_nodes()):
            # Since pydot has some issues in treating quotes in string, the stripping step is to make sure
            # that no extra, unwanted double quotes remain in the node_smiles string
            iron_node = Node(
                properties={"node_smiles": node.get_name().strip('"')},
                iid=str(id_n),
                labels=[],
            )
            iron.add_node(str(id_n), iron_node)
        for id_e, edge in enumerate(route.get_edges()):
            iron_edge = self.pydot_edges_to_iron_edge(edge, id_e, iron)
            iron.add_edge(str(id_e), iron_edge)
        return iron

    @staticmethod
    def pydot_edges_to_iron_edge(edge, id_e: str, iron: Iron) -> Edge:
        """To transform pydot edge to iron edge"""

        a = edge.get_source().strip('"')
        a_id = next(
            n.iid for n in iron.nodes.values() if n.properties["node_smiles"] == a
        )
        b = edge.get_destination().strip('"')
        b_id = next(
            n.iid for n in iron.nodes.values() if n.properties["node_smiles"] == b
        )
        return build_iron_edge(a_id, b_id, id_e)


@GraphFormatCatalog.register_format("ibm_retro", "IBMRXN output (dictionary)")
class IbmRetro(GraphFormatTranslator):
    """Translator subclass to handle translations of outputs generated by IBM CASP tool"""

    as_input = "implemented"
    as_output = None

    def from_iron(self, graph: Iron) -> Union[list, None]:
        pass

    def to_iron(self, route: dict) -> Union[Iron, None]:
        """Translates a dictionary generated by the IBM CASP tool into an Iron instance"""

        try:
            if not route:
                raise exceptions.EmptyRoute
            iron_graph = ibm_dict_to_iron(route, iron=None, parent=None)

            if iron_graph.i_edge_number() == 0:
                raise exceptions.InvalidRoute

            iron_graph.source = "ibm_" + datetime.datetime.now().strftime(
                "%Y%m%d%H%M%S%f"
            )
            return iron_graph

        except exceptions.EmptyRoute:
            logger.warning(
                'While translating from IBM to Iron an empty route was found: "None" returned'
            )
            return None
        except exceptions.InvalidRoute:
            logger.warning(
                "While translating from IBM to Iron an invalid route containing only one molecule was found: "
                '"None" returned'
            )
            return None


@GraphFormatCatalog.register_format("az_retro", "AiZynthFinder output (dictionary)")
class AzRetro(GraphFormatTranslator):
    """Translator subclass to handle translations of outputs generated by AZ CASP tool"""

    as_input = "implemented"
    as_output = None

    def from_iron(self, graph: Iron):
        pass

    def to_iron(self, route: dict) -> Union[Iron, None]:
        """Translates a dictionary generated by the AZ CASP tool into an Iron instance"""

        try:
            if not route:
                raise exceptions.EmptyRoute
            iron_graph = az_dict_to_iron(route, iron=None, parent=None)
            iron_graph.source = "az_" + datetime.datetime.now().strftime(
                "%Y%m%d%H%M%S%f"
            )
            return iron_graph
        except exceptions.EmptyRoute:
            logger.warning(
                'While translating from AZ to Iron an empty route was found: "None" returned'
            )
            return None


@GraphFormatCatalog.register_format("mit_retro", "Askcos output (dictionary)")
class MitRetro(GraphFormatTranslator):
    """Translator subclass to handle translations of outputs generated by Askcos CASP tool"""

    as_input = "implemented"
    as_output = None

    def from_iron(self, graph: Iron):
        pass

    def to_iron(self, route: dict) -> Union[Iron, None]:
        """Translates a dictionary generated by the Askcos CASP tool into an Iron instance"""

        try:
            if not route:
                raise exceptions.EmptyRoute
            iron_graph = mit_dict_to_iron(route, iron=None, parent=None)
            iron_graph.source = "mit_" + datetime.datetime.now().strftime(
                "%Y%m%d%H%M%S%f"
            )
            return iron_graph
        except exceptions.EmptyRoute:
            logger.warning(
                'While translating from Asckos to Iron an empty route was found: "None" returned'
            )
            return None


@GraphFormatCatalog.register_format("noc", "NOC format (dictionary)")
class NOC(GraphFormatTranslator):
    """Translator subclass to handle translations of NOC-compatible documents"""

    as_input = None
    as_output = "implemented"

    def from_iron(self, route_iron: Iron) -> Union[dict, None]:
        """Translates an Iron instance into an NOC-compatible document"""

        try:
            if route_iron is None:
                raise exceptions.EmptyRoute
            nodes = route_iron.nodes
            node_documents: List = []
            edge_documents: List = []
            for iid, n in nodes.items():
                node_instance = n.properties.get("node_type")
                node_documents = self.build_node_documents(
                    node_instance, node_documents
                )

                if type(node_instance) is ChemicalEquation:
                    edge_documents = self.build_edges_documents(
                        node_instance, edge_documents
                    )

            return {"nodes": node_documents, "edges": edge_documents}
        except exceptions.EmptyRoute:
            logger.warning(
                'While translating from Iron to NOC format an empty route was found: "None" returned'
            )
            return None

    @staticmethod
    def build_edges_documents(
        node_instance: ChemicalEquation, edge_documents: list
    ) -> List[dict]:
        """To build the list of dictionaries with edges information"""
        current_node = "CE" + str(node_instance.uid)
        for item in node_instance.role_map.get("reactants"):
            edge_doc = {
                "type": "REACTANT",
                "source": "M" + str(item),
                "destination": current_node,
                "properties": {},
            }
            edge_documents.append(edge_doc)
        for item in node_instance.role_map.get("reagents"):
            edge_doc = {
                "type": "REAGENT",
                "source": "M" + str(item),
                "destination": current_node,
                "properties": {},
            }
            edge_documents.append(edge_doc)
        for item in node_instance.role_map.get("products"):
            edge_doc = {
                "type": "PRODUCT",
                "source": current_node,
                "destination": "M" + str(item),
                "properties": {},
            }
            edge_documents.append(edge_doc)
        # append Template node
        # append Pattern nodes
        # append Template - ChemicalEquation relationships
        # append Template - Patterns relationships
        # append Patterns - Molecules relationships (to be calculated while constructing chemical equation!!!!)
        return edge_documents

    @staticmethod
    def build_node_documents(
        node_instance: Union[Molecule, ChemicalEquation], node_documents: list
    ) -> List[dict]:
        """To build the list of dictionaries with nodes information"""

        node_dict = node_instance.to_dict()
        node_document = {
            "type": node_dict.get("type"),
            "properties": {"smiles": node_instance.smiles},
        }
        if isinstance(node_instance, ChemicalEquation):
            node_document["uid"] = "CE" + str(node_dict.get("uid"))
            if node_instance.disconnection is not None:
                node_document["disconnection_uid"] = node_instance.disconnection.uid
        else:
            node_document["uid"] = "M" + str(node_dict.get("uid"))

        node_documents.append(node_document)
        return node_documents

    def to_iron(self, graph: dict) -> Union[Iron, None]:
        pass


@GraphFormatCatalog.register_format("pydot_visualization", "PNG file")
class PydotVisualization(GraphFormatTranslator):
    """Translator subclass to generate pictures"""

    as_input = None
    as_output = "implemented"

    def from_iron(self, route_iron: Iron) -> None:
        """Translates an Iron instance into a PNG picture"""

        try:
            if route_iron is None:
                raise exceptions.EmptyRoute
            dot_graph = pydot.Dot(route_iron.source, graph_type="digraph")
            # Translating iron nodes into dot nodes
            for id_n, node in route_iron.nodes.items():
                self.create_dot_node(node, id_n, dot_graph)

            # Translating iron edges into dot edges
            for edge in route_iron.edges.values():
                self.create_dot_edge(edge, route_iron, dot_graph)

            dot_graph.write(f"route_{route_iron.source}.dot")
            os.system(
                f"dot -Tpng route_{route_iron.source}.dot > route_{route_iron.source}.png"
            )

            for id_n in route_iron.nodes.keys():
                os.remove(f"{id_n}.png")

        except exceptions.EmptyRoute:
            logger.warning(
                'While translating from Iron to PyDot-visualization format- an empty route was found: "None" returned'
            )
            return None

    def create_dot_node(self, node: Node, id_n: int, dot_graph: pydot.Dot) -> None:
        """To add a dot node for visualization"""

        node_instance = node.properties["node_type"]
        self.create_node_picture(node_instance, id_n)
        dot_graph.add_node(
            pydot.Node(
                node.properties["node_unmapped_smiles"], image=f"{id_n}.png", label=""
            )
        )

    @staticmethod
    def create_node_picture(
        node_instance: Union[Molecule, ChemicalEquation], id_n: int
    ) -> None:
        """To generate the node picture"""

        if type(node_instance) == ChemicalEquation:
            depiction_data = cid.draw_reaction(node_instance.rdrxn)

        else:
            depiction_data = cid.draw_molecule(node_instance.rdmol)

        lio.write_rdkit_depict(data=depiction_data, file_path=f"{id_n}.png")

    @staticmethod
    def create_dot_edge(edge: Edge, route_iron: Iron, dot_graph: pydot.Dot) -> None:
        """To add an edge for visualization"""

        # the unmapped smiles is used to avoid the pydot issues in dealing with nodes' name with numbers
        a = next(
            n.properties["node_unmapped_smiles"]
            for i, n in route_iron.nodes.items()
            if i == edge.a_iid
        )
        b = next(
            n.properties["node_unmapped_smiles"]
            for i, n in route_iron.nodes.items()
            if i == edge.b_iid
        )
        dot_graph.add_edge(pydot.Edge(a, b))

    def to_iron(self, graph):
        pass


@GraphFormatCatalog.register_format("sparrow", "file generated by the SPARROW software")
class Sparrow(GraphFormatTranslator):
    """Translator subclass to handle graphs generated by the SPARROW software"""

    as_input = "implemented"
    as_output = "implemented"

    def to_iron(self, graph) -> Iron:
        reactions = [
            reaction
            for target, data in graph.items()
            for reaction in data["Reactions"]
            if not reaction["smiles"].startswith(">>")
        ]
        iron = Iron()
        for n, reaction in enumerate(reactions):
            if reaction["conditions"]:
                reaction_smiles = self.handle_reagents(reaction)
            else:
                reaction_smiles = reaction["smiles"]
            node = Node(
                iid=str(n), properties={"node_smiles": reaction_smiles}, labels=[]
            )
            iron.add_node(str(n), node)
        iron.source = "sparrow_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        return iron

    @staticmethod
    def handle_reagents(reaction: dict) -> str:
        """To add reagents to the reaction smiles"""
        reactants_products = reaction["smiles"].split(">>")
        reagents = reaction["conditions"]
        return (
            reactants_products[0]
            + ">"
            + ".".join(reagents)
            + ">"
            + reactants_products[-1]
        )

    def from_iron(self, route_iron: Iron) -> dict:
        reactions = []
        compounds = []
        if self.get_node_types(route_iron) == [ChemicalEquation, Molecule]:
            logger.warning(
                "For full compatibility with sparrow software, the graph should be bipartite"
            )
        for id_n, node in route_iron.nodes.items():
            if "node_type" in node.properties:
                if isinstance(node.properties["node_type"], ChemicalEquation):
                    reaction = self.handle_reaction(node)
                    reactions.append(reaction)
                elif isinstance(node.properties["node_type"], Molecule):
                    compound = self.handle_molecules(node, route_iron)
                    compounds.append(compound)
            else:
                node_smiles = node.properties["node_smiles"]
                if node_smiles.count(">") == 2:
                    reactions.append(node_smiles)
                else:
                    compounds.append(node_smiles)
        return {"Compound Nodes": compounds, "Reaction Nodes": reactions}

    @staticmethod
    def get_node_types(iron: Iron) -> list:
        """To check which node types are present in the graph"""
        datatypes = {
            node.properties.get("node_type")
            for node in iron.nodes.values()
            if node.properties.get("node_type") is not None
        }
        return sorted(datatypes)

    def handle_reaction(self, node: Node) -> dict:
        """To create a Reaction Node entry"""
        chemical_equation = node.properties["node_type"]
        reaction = {
            "smiles": node.properties["node_unmapped_smiles"],
            "parents": self.get_molecules(chemical_equation, "reactants"),
            "children": self.get_molecules(chemical_equation, "products"),
            "conditions": self.get_molecules(chemical_equation, "reagents"),
        }
        return reaction

    @staticmethod
    def get_molecules(chemical_equation: ChemicalEquation, role: str) -> List[str]:
        """To get a list of Molecules' smiles with a given role"""
        mols = [reactant.smiles for reactant in chemical_equation.get_reactants()]
        return mols

    @staticmethod
    def handle_molecules(node: Node, iron: Iron) -> dict:
        """To create a Compound Node entry"""
        molecule = node.properties["node_type"]
        compound = {
            "smiles": molecule.smiles,
        }
        parents = iron.get_parent_nodes(node.iid)
        compound["parents"] = [
            n.properties["node_unmapped_smiles"]
            for i, n in iron.nodes.items()
            if i in parents
        ]
        children = iron.get_child_nodes(node.iid)
        compound["children"] = [
            n.properties["node_unmapped_smiles"]
            for i, n in iron.nodes.items()
            if i in children
        ]
        return compound


# helper functions
def get_output_translators():
    """
    To retrieve the formats available as output and thier brief descriptions

    Returns:
    --------
    available_output: dict
        The information about the available output formats

    Example:
    >>> out_formats = get_output_translators()
    """
    all_formats = GraphFormatCatalog.list_formats()
    available_output = {}
    for c in list(GraphFormatTranslator.__subclasses__()):
        if c.as_output is not None:
            if name := next(
                (f for f, info in all_formats.items() if info["class"] == c), None
            ):
                available_output[name] = all_formats[name]["info"]
    available_output.update({"iron": "Iron object", "syngraph": "Syngraph object"})
    return available_output


def get_input_translators():
    """
    To retrieve the formats available as input and thier brief descriptions

    Returns:
    --------
    available_input: dict
        The information about the available input formats

    Example:
    >>> in_formats = get_input_translators()
    """

    all_formats = GraphFormatCatalog.list_formats()
    available_input = {}
    for c in list(GraphFormatTranslator.__subclasses__()):
        if c.as_input is not None:
            if name := next(
                (f for f, info in all_formats.items() if info["class"] == c), None
            ):
                available_input[name] = all_formats[name]["info"]
    available_input.update({"iron": "Iron object", "syngraph": "Syngraph object"})
    return available_input


def get_formats() -> dict:
    """To get all the available formats for input and output"""
    return {"as_input": get_input_translators(), "as_output": get_output_translators()}
