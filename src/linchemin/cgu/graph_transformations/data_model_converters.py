from abc import ABC, abstractmethod
from typing import Tuple, Type, Union

import linchemin.cgu.graph_transformations.exceptions as exceptions
import linchemin.cheminfo.functions as cif
from linchemin.cgu.convert import converter
from linchemin.cgu.graph_transformations.supporting_functions import build_iron_edge
from linchemin.cgu.iron import Iron, Node
from linchemin.cgu.syngraph import (
    BipartiteSynGraph,
    MonopartiteMolSynGraph,
    MonopartiteReacSynGraph,
)
from linchemin.cheminfo.models import ChemicalEquation, Molecule
from linchemin.utilities import console_logger

logger = console_logger(__name__)


class DataModelConverter(ABC):
    """
    Abstract representation of converters handling graph data models.
    """

    @abstractmethod
    def iron_to_syngraph(
        self, iron_graph: Iron
    ) -> Union[
        MonopartiteReacSynGraph, MonopartiteMolSynGraph, BipartiteSynGraph, None
    ]:
        """To translate an Iron graph into a SynGraph object of the correct type"""
        pass

    @abstractmethod
    def syngraph_to_iron(
        self,
        syngraph: Union[
            MonopartiteReacSynGraph, MonopartiteMolSynGraph, BipartiteSynGraph, None
        ],
    ) -> Union[Iron, None]:
        """To translate a SynGraph object into an Iron graph"""
        pass

    @abstractmethod
    def convert_syngraph(
        self,
        syngraph: Union[
            MonopartiteReacSynGraph, MonopartiteMolSynGraph, BipartiteSynGraph
        ],
    ) -> Union[MonopartiteReacSynGraph, MonopartiteMolSynGraph, BipartiteSynGraph]:
        """To convert between SynGraph types"""
        pass

    def get_node_info(
        self, node: Union[ChemicalEquation, Molecule], iron: Iron
    ) -> Tuple[int, Node]:
        """To create or identify a specif node in an Iron instance"""
        # if the node is not yet in the Iron instance, it is created and added to Iron
        if node not in [n.properties["node_type"] for n in iron.nodes.values()]:
            id_n = iron.i_node_number()
            id_n1, node1 = self.build_iron_node(node, id_n)
            iron.add_node(str(id_n), node1)
        else:
            # if the node is already included in Iron, the relative information is retrieved
            id_n1, node1 = next(
                (i, n)
                for i, n in iron.nodes.items()
                if n.properties["node_type"] == node
            )
        return id_n1, node1

    @staticmethod
    def build_iron_node(
        node: Union[ChemicalEquation, Molecule], id_n: int
    ) -> Tuple[int, Node]:
        """To build an Iron node instance from a ChemicalEquation"""
        # the unmapped smiles is built so that the route is suitable to be correctly displayed in a png file
        if type(node) == ChemicalEquation:
            unmapped_smiles = cif.rdrxn_to_string(
                node.rdrxn, out_fmt="smiles", use_atom_mapping=False
            )
        else:
            unmapped_smiles = node.smiles

        prop = {
            "node_unmapped_smiles": unmapped_smiles,
            "node_smiles": node.smiles,
            "node_type": node,
        }
        return id_n, Node(iid=str(id_n), properties=prop, labels=[])


class DataModelCatalog:
    """Class to store the available GraphFormatTranslators"""

    _registered_data_models = {}

    @classmethod
    def register_datamodel(cls, name: str, info: str):
        """
        Decorator for registering a new data model translator.

        Parameters:
        ------------
        name: str
            The name of the data model to be used as a key in the registry
        info: str
            A brief description of the translator

        Returns:
        ---------
        function: The decorator function.
        """

        def decorator(converter_class: Type[DataModelConverter]):
            cls._registered_data_models[name.lower()] = {
                "class": converter_class,
                "info": info,
            }
            return converter_class

        return decorator

    @classmethod
    def get_data_model(cls, name: str) -> DataModelConverter:
        """
        To get an instance of the specified DataModelConverter.

        Parameters:
        ------------
        name: str
            The name of the DataModelConverter

        Returns:
        ---------
        DataModelConverter: An instance of the specified DataModelConverter

        Raises:
        -------
        UnavailableDataModel: If the specified data model is not registered.
        """
        converter = cls._registered_data_models.get(name.lower())
        if converter is None:
            logger.error(f"Data model '{name}' not found")
            raise exceptions.UnavailableDataModel
        return converter["class"]()

    @classmethod
    def list_data_models(cls):
        """
        To list the names of all available data models.

        Returns:
        ---------
        formats: dict
            The names and information of the available data models.
        """
        return cls._registered_data_models


# Data model factory concrete implementations
@DataModelCatalog.register_datamodel(
    "monopartite_reactions", "A graph with only reaction nodes"
)
class MonopartiteReactionsGenerator(DataModelConverter):
    """DataModelConverter subclass to handle translations into and from MonopartiteReacSynGraph instances"""

    def iron_to_syngraph(
        self, iron_graph: Iron
    ) -> Union[MonopartiteReacSynGraph, None]:
        """Translates an Iron instance into a MonopartiteReacSynGraph instance"""
        try:
            if iron_graph is None:
                raise exceptions.EmptyRoute
            return MonopartiteReacSynGraph(iron_graph)
        except exceptions.EmptyRoute:
            logger.warning(
                "While converting from Iron to monopartite-reactions SynGraph object an empty route was found: "
                '"None" returned'
            )
            return None

    def syngraph_to_iron(self, syngraph: MonopartiteReacSynGraph) -> Union[Iron, None]:
        """Translates a MonopartiteReacSynGraph instance into an Iron instance"""
        try:
            if syngraph is None:
                raise exceptions.EmptyRoute
            iron = Iron()
            id_e = 0
            for parent, children in syngraph.graph.items():
                id_n1, node1 = self.get_node_info(parent, iron)

                for c in children:
                    id_n2, node2 = self.get_node_info(c, iron)

                    e = build_iron_edge(str(id_n1), str(id_n2), str(id_e))
                    iron.add_edge(str(id_e), e)
                    id_e += 1
            if syngraph.name is None:
                iron.name = syngraph.uid
            else:
                iron.name = syngraph.name
            return iron

        except exceptions.EmptyRoute:
            logger.warning(
                'While converting from a monopartite-reactions SynGraph to Iron an empty route was found: "None" '
                "returned"
            )
            return None

    def convert_syngraph(
        self,
        syngraph: Union[
            MonopartiteReacSynGraph, MonopartiteMolSynGraph, BipartiteSynGraph
        ],
    ) -> MonopartiteReacSynGraph:
        return converter(syngraph, "monopartite_reactions")


@DataModelCatalog.register_datamodel(
    "bipartite", "A graph with reaction and molecule nodes"
)
class BipartiteGenerator(DataModelConverter):
    """DataModelConverter subclass to handle translations into and from BipartiteSynGraph instances"""

    def iron_to_syngraph(self, iron_graph: Iron) -> Union[BipartiteSynGraph, None]:
        """Translates an Iron instance into a BipartiteSynGraph instance"""
        try:
            if iron_graph is None:
                raise exceptions.EmptyRoute
            return BipartiteSynGraph(iron_graph)
        except exceptions.EmptyRoute:
            logger.warning(
                'While converting from Iron to bipartite SynGraph object an empty route was found: "None" returned'
            )
            return None

    def syngraph_to_iron(self, syngraph: BipartiteSynGraph) -> Union[Iron, None]:
        """Translates a BipartiteSynGraph instance into an Iron instance"""
        try:
            if syngraph is None:
                raise exceptions.EmptyRoute
            iron = Iron()
            id_e = 0
            for parent, children in syngraph.graph.items():
                id_n1, node1 = self.get_node_info(parent, iron)

                for c in children:
                    id_n2, node2 = self.get_node_info(c, iron)

                    e = build_iron_edge(str(id_n1), str(id_n2), str(id_e))
                    iron.add_edge(str(id_e), e)
                    id_e += 1

            if syngraph.name is None:
                iron.name = syngraph.uid
            else:
                iron.name = syngraph.name
            return iron
        except exceptions.EmptyRoute:
            logger.warning(
                'While converting from a bipartite SynGraph to Iron an empty route was found: "None" returned'
            )
            return None

    def convert_syngraph(
        self,
        syngraph: Union[
            MonopartiteReacSynGraph, MonopartiteMolSynGraph, BipartiteSynGraph
        ],
    ) -> BipartiteSynGraph:
        return converter(syngraph, "bipartite")


@DataModelCatalog.register_datamodel(
    "monopartite_molecules", "A graph with only molecule nodes"
)
class MonopartiteMoleculesGenerator(DataModelConverter):
    """DataModelConverter subclass to handle translations into and from MonopartiteMolSynGraph instances"""

    def iron_to_syngraph(self, iron_graph: Iron) -> Union[MonopartiteMolSynGraph, None]:
        """Translates an Iron instance into a MonopartiteMolSynGraph instance"""
        try:
            if iron_graph is None:
                raise exceptions.EmptyRoute
            return MonopartiteMolSynGraph(iron_graph)
        except exceptions.EmptyRoute:
            logger.warning(
                "While converting from Iron to a monopartite-molecules SynGraph object an empty route was found: "
                '"None" returned'
            )
            return None

    def syngraph_to_iron(self, syngraph: MonopartiteMolSynGraph) -> Union[Iron, None]:
        """Translates a MonopartiteReacSynGraph instance into an Iron instance"""
        try:
            if syngraph is None:
                raise exceptions.EmptyRoute
            iron = Iron()
            id_e = 0
            for parent, children in syngraph.graph.items():
                id_n1, node1 = self.get_node_info(parent, iron)

                for c in children:
                    id_n2, node2 = self.get_node_info(c, iron)

                    e = build_iron_edge(str(id_n1), str(id_n2), str(id_e))
                    iron.add_edge(str(id_e), e)
                    id_e += 1

            if syngraph.name is None:
                iron.name = syngraph.uid
            else:
                iron.name = syngraph.name
            return iron
        except exceptions.EmptyRoute:
            logger.warning(
                'While converting from a monopartite-molecules SynGraph to Iron an empty route was found: "None" '
                "returned"
            )
            return None

    def convert_syngraph(
        self,
        syngraph: Union[
            MonopartiteReacSynGraph, MonopartiteMolSynGraph, BipartiteSynGraph
        ],
    ) -> MonopartiteMolSynGraph:
        return converter(syngraph, "monopartite_molecules")
