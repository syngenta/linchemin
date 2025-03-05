import copy
from typing import List, Optional, Union

import linchemin.utilities as utilities
from linchemin.cgu.convert import converter
from linchemin.cgu.syngraph import (
    BipartiteSynGraph,
    MonopartiteMolSynGraph,
    MonopartiteReacSynGraph,
    SynGraph,
)
from linchemin.cheminfo.constructors import MoleculeConstructor
from linchemin.cheminfo.models import Molecule

logger = utilities.console_logger(__name__)


class RouteEnumerator:
    """Class to enumerate all the possible routes in a SynGraph instance.
    It returns a list of SynGraph instances, each of which represents a possible route.
    The routes are built by substituting a leaf node with a list of alternative leaves.
    The alternative leaves are Molecule instances built from the smiles strings provided by the user.
    """

    def __init__(
        self,
        original_route: MonopartiteReacSynGraph,
        leaf_to_substitute: int,
        reaction_products: List[int],
    ):
        self.original_route = self._fix_original_route_format(original_route)
        self.leaf_molecule = self._get_leaf_molecule(leaf_to_substitute)
        if self.leaf_molecule is None:
            logger.error("The selected leaf is not present in the original route")
            raise KeyError("Leaf not found in the original route")
        self.reaction_products = reaction_products

    @staticmethod
    def _fix_original_route_format(
        original_route: MonopartiteReacSynGraph,
    ) -> MonopartiteMolSynGraph:
        return converter(original_route, "monopartite_molecules")

    def _get_leaf_molecule(self, leaf_id: int) -> Optional[Molecule]:
        """To get the Molecule instance of a leaf node"""
        return next(
            (
                node
                for node in self.original_route.get_unique_nodes()
                if node.uid == leaf_id
            ),
            None,
        )

    @staticmethod
    def _build_alternative_leaves(
        leaves_structure: List[str], inp_fmt: str
    ) -> List[Molecule]:
        """To build a Molecule instance from a smiles string"""
        mol_constructor = MoleculeConstructor()
        alternative_leaves = []
        for leaf in leaves_structure:
            try:
                molecule = mol_constructor.build_from_molecule_string(
                    leaf, inp_fmt=inp_fmt
                )
                alternative_leaves.append(molecule)
            except Exception as e:
                logger.warning(f"Failed to build molecule from SMILES {leaf}: {str(e)}")
        return alternative_leaves

    def build_alternative_routes(
        self, alternative_leaves: List[str], inp_fmt: str
    ) -> List[MonopartiteMolSynGraph]:
        """
        Build alternative routes by substituting the leaf node with the alternative leaves.

        Args:
            alternative_leaves: List of alternative leaf nodes
            inp_fmt: Input format specification

        Returns:
            List of alternative routes as MonopartiteMolSynGraph objects
        """
        alternative_leaves = self._build_alternative_leaves(alternative_leaves, inp_fmt)
        new_base_route = copy.deepcopy(self.original_route)
        leaf_children = self._get_filtered_leaf_children()

        if len(leaf_children) == 1:
            new_base_route.remove_node(self.leaf_molecule.uid)

        original_root_count = len(self.original_route.get_roots())
        return self._create_alternative_routes(
            alternative_leaves, new_base_route, leaf_children, original_root_count
        )

    def _get_filtered_leaf_children(self) -> List[Molecule]:
        """To get the list of children of the leaf node,
        if they are also products of the modified reaction"""
        leaf_children = list(self.original_route[self.leaf_molecule])
        if len(leaf_children) > 1:
            return [child for child in leaf_children if child in self.reaction_products]
        return leaf_children

    def _create_alternative_routes(
        self,
        alternative_leaves: List[Molecule],
        new_base_route: MonopartiteMolSynGraph,
        leaf_children: List[Molecule],
        original_root_count: int,
    ) -> List[MonopartiteMolSynGraph]:
        alternative_routes = []
        for leaf in alternative_leaves:
            new_route = self._add_nodes_to_new_route(
                new_base_route, leaf, leaf_children
            )
            if self._is_valid_route(new_route, original_root_count):
                alternative_routes.append(new_route)
            else:
                logger.warning(
                    "The new route does not have the same number of roots as the original route."
                )
        return alternative_routes

    @staticmethod
    def _add_nodes_to_new_route(
        new_base_route, leaf, leaf_children
    ) -> MonopartiteMolSynGraph:
        """To add the new leaf to the base route"""
        new_route = copy.deepcopy(new_base_route)
        new_route.add_node((leaf, leaf_children))
        return new_route

    @staticmethod
    def _is_valid_route(new_route, original_root_count) -> bool:
        if len(new_route.get_roots()) == original_root_count:
            new_route.set_name(new_route.uid)
            return True
        return False


def enumerate_routes(
    original_route: Union[
        BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph
    ],
    starting_material_to_substitute: int,
    reaction_to_modify: int,
    alternative_starting_materials_structures: List[str],
    starting_materials_input_format: str,
    out_data_model: Optional[str],
) -> List[SynGraph]:
    """
    To build all the possible graphs starting from a SynGraph instance
    and substituting a leaf node with a list of alternative leaves.
    The alternative leaves are Molecule instances built from the
    molecular strings provided by the user.

    Parameters:
    -----------
    original_route: SynGraph
        The original route to be substituted
    starting_material_to_substitute: int
        The uid of the starting material node to be substituted
    alternative_starting_materials_structures: List[str]
        The list of molecular strings of the alternative leaves
    starting_materials_input_format: str
        The string format of the starting materials structure (eg, 'smiles')
    out_data_model: Optional[str]
        The desired data model for the output routes. Default to monopartite_molecules

    Returns:
    --------
    List[SynGraph]
        The list of all the possible routes, each of which is a SynGraph instance

    Raises:
    -------
    KeyError:
        If the selected leaf is not present in the original route

    """
    original_route: MonopartiteReacSynGraph = converter(
        original_route, "monopartite_reactions"
    )
    chemical_equation = next(
        (
            node
            for node in original_route.get_unique_nodes()
            if node.uid == reaction_to_modify
        ),
        None,
    )
    if chemical_equation is None:
        logger.error("The selected reaction is not present in the original route")
        raise KeyError("Reaction not found in the original route")
    products = chemical_equation.role_map["products"]
    route_enumerator = RouteEnumerator(
        original_route, starting_material_to_substitute, products
    )
    new_routes = route_enumerator.build_alternative_routes(
        alternative_starting_materials_structures, starting_materials_input_format
    )
    if out_data_model is None:
        return new_routes
    return [converter(route, out_data_model) for route in new_routes]
