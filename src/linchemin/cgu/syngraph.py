from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple, Union

import linchemin.utilities as utilities
from linchemin.cgu.iron import Iron
from linchemin.cheminfo.constructors import (
    ChemicalEquationConstructor,
    MoleculeConstructor,
)
from linchemin.cheminfo.models import ChemicalEquation, Molecule

"""
Module containing the implementation of the SynGraph data model and its sub-types: bipartite, monopartite reactions
and monopartite molecules.

"""
logger = utilities.console_logger(__name__)


class SynGraph(ABC):
    """Abstract class implementing the homonym data model as a dictionary of sets.

    Attributes:
            graph: a dictionary of sets

            source: a string containing the sources of the graph

            uid: a string uniquely identifying the SynGraph instance based on the underlying graph
    """

    def __init__(self, initiator: Union[List[dict], Iron, None] = None):
        """
        To instantiate a new SynGraph instance

        Parameters:
        -------------
        initiator: initiator: Union[List[dict], Iron, None]
            The initiator of the new SynGraph instance (default None -> an empty graph is initialized))
            It can be:
                (i) an Iron instance
                (ii) a list of dictionaries of reaction strings in the form
                    [{'query_id': n, 'output_string': reaction}]

        """
        self.source = str
        self.graph = defaultdict(set)

        if initiator is not None:
            if isinstance(initiator, Iron):
                self.initialize_with_iron(initiator)
            elif isinstance(initiator, list):
                all_smiles = [d["output_string"] for d in initiator]
                chemical_equations = self.build_chemical_equations(all_smiles)
                self.builder_from_reaction_list(chemical_equations)

    def initialize_with_iron(self, iron_graph: Iron) -> None:
        """To initialize the SynGraph builder from an Iron instance"""
        all_smiles = [
            node.properties["node_smiles"] for node in iron_graph.nodes.values()
        ]
        if all(s.count(">") == 2 for s in all_smiles):
            # all smiles are reaction smiles
            chemical_equations = self.build_chemical_equations(all_smiles)
            self.builder_from_reaction_list(chemical_equations)
        elif all(s.count(">") != 2 for s in all_smiles):
            # all smiles are molecule smiles
            self.builder_from_iron(iron_graph)
        else:
            # bipartite: both molecules and reactions smiles are present
            reaction_smiles = [s for s in all_smiles if s.count(">") == 2]
            chemical_equations = self.build_chemical_equations(reaction_smiles)
            self.builder_from_reaction_list(chemical_equations)

    @staticmethod
    def build_chemical_equations(smiles_list: List[str]) -> List[ChemicalEquation]:
        """To build a list of ChemicalEquation objects from a list of smiles"""
        chemical_eq_constructor = ChemicalEquationConstructor(
            molecular_identity_property_name="smiles"
        )
        return [
            chemical_eq_constructor.build_from_reaction_string(
                reaction_string=smiles, inp_fmt="smiles"
            )
            for smiles in smiles_list
        ]

    @property
    def uid(self) -> str:
        """To define the SynGraph unique identifier based on the underlying graph"""
        tups = []
        for parent, children in self:
            if not children:
                # To take into account isolated nodes
                tups.append((parent.uid, "x", 0))
            else:
                tups.extend((parent.uid, ">", child.uid) for child in children)
        sorted_tups = sorted(tups, key=lambda x: (x[0], x[-1]))
        h = utilities.create_hash(str(frozenset(sorted_tups)))
        if type(self) == BipartiteSynGraph:
            h = "".join(["BP", str(h)])
        elif type(self) == MonopartiteReacSynGraph:
            h = "".join(["MPR", str(h)])
        elif type(self) == MonopartiteMolSynGraph:
            h = "".join(["MPM", str(h)])
        return h

    @abstractmethod
    def builder_from_iron(self, iron_graph: Iron) -> None:
        """To build a SynGraph instance from an Iron object"""
        pass

    @abstractmethod
    def builder_from_reaction_list(
        self, chemical_equations: List[ChemicalEquation]
    ) -> None:
        """To build a SynGraph instance from a list of ChemicalEquation objects"""
        pass

    def get_roots(self) -> list:
        """To retrieve the list of 'root' nodes of a SynGraph instance"""
        return [parent for parent, children in self if children == set()]

    @abstractmethod
    def get_leaves(self) -> list:
        """To retrieve the list of 'leaf' nodes of a SynGraph instance"""
        pass

    def get_unique_nodes(self) -> set:
        """To get the set of unique nodes included in a SynGraph instance"""
        return {parent for parent in self.graph}

    def set_source(self, source):
        """To set the source attribute of a SynGraph instance"""
        self.source = source

    def __iter__(self):
        self._iter_obj = iter(self.graph.items())
        return self._iter_obj

    def __next__(self):
        """To iterate over the nodes"""
        return next(self._iter_obj)

    def __eq__(self, other):
        """To check if two SynGraph instances are the same"""
        return type(self) == type(other) and self.graph == other.graph

    def __getitem__(self, item):
        """To make SynGraph subscriptable"""
        return self.graph[item]

    def __str__(self):
        text = ""
        for r, connections in self:
            if connections:
                text = text + "{} -> {} \n".format(r, *connections)
            else:
                text = text + f"{r}\n"
        return text

    def add_node(self, nodes_tup: tuple) -> None:
        """To add a 'parent' node and its 'children' nodes to a SynGraph instance.

        Format of the input tuple: (parent_node, [child1, child2, ...])"""
        # The nodes are added to the SynGraph instance
        if nodes_tup[0] not in self.graph:
            # If the source node is not in the SynGraph instance yet, it is added with its connections
            self.graph[nodes_tup[0]] = set(nodes_tup[1])
        else:
            # otherwise, the connections are added to the pre-existing node
            for c in nodes_tup[1]:
                self.graph[nodes_tup[0]].add(c)

    def remove_node(self, node_to_remove_id: int) -> None:
        """To remove a node by its uid from a SynGraph instance"""
        if node_to_remove := next(
            (node for node in self.get_unique_nodes() if node.uid == node_to_remove_id),
            None,
        ):
            self.graph.pop(node_to_remove)
            for parent, children in self:
                if node_to_remove in children:
                    children.remove(node_to_remove)
        else:
            logger.warning("The selected node is not present in the SynGraph instance.")

    @staticmethod
    def find_reactants_products(
        iron_graph: Iron, node1: int, node2: int, connections: list
    ) -> Tuple[List, List]:
        all_reactants = [
            iron_graph.nodes[tup[0]].properties["node_smiles"]
            for tup in connections
            if tup[1] == node2
        ]
        all_products = [
            iron_graph.nodes[tup[1]].properties["node_smiles"]
            for tup in connections
            if tup[0] == node1
        ]
        return all_reactants, all_products

    def add_molecular_roots(self) -> None:
        """To correctly handle the molecule roots in the graph"""
        roots: list = []
        for products in self.graph.values():
            roots.extend(
                prod
                for prod in products
                if prod not in list(self.graph.keys()) and isinstance(prod, Molecule)
            )

        for root in roots:
            self.add_node((root, []))

    def find_parent_node(
        self, child_node: Union[Molecule, ChemicalEquation]
    ) -> Union[set, None]:
        """To identify the parent node of a given child node"""
        if parent_node := {
            parent for parent, children in self if child_node in children
        }:
            return parent_node
        else:
            return None


class BipartiteSynGraph(SynGraph):
    """SynGraph subclass representing a Bipartite (Molecule and ChemicalEquation nodes) SynGraph"""

    def builder_from_reaction_list(
        self, chemical_equations: List[ChemicalEquation]
    ) -> None:
        """To build a BipartiteSynGraph instance from a list of ChemicalEquation objects"""
        for ch_equation in chemical_equations:
            products = ch_equation.get_products()

            self.add_node((ch_equation, products))
            reactants = ch_equation.get_reactants()

            for reactant in reactants:
                self.add_node((reactant, [ch_equation]))
        self.add_molecular_roots()
        # this is to avoid disconnected nodes due to reactions producing reagents of other reactions
        self.remove_isolated_ces()

        self.set_source(str(self.uid))

    def builder_from_iron(self, iron_graph: Iron) -> None:
        """To build a BipartiteSynGraph instance from an Iron instance"""
        connections = [edge.direction.tup for id_e, edge in iron_graph.edges.items()]
        for node1, node2 in connections:
            all_reactants, all_products = self.find_reactants_products(
                iron_graph, node1, node2, connections
            )

            check = [item for item in all_reactants if item in all_products]
            if not check:
                chemical_equation = get_reaction_instance(all_reactants, all_products)

                reactant = iron_graph.nodes[node1].properties["node_smiles"]
                product = iron_graph.nodes[node2].properties["node_smiles"]

                self.add_nodes_sequence(reactant, product, chemical_equation)
        self.add_molecular_roots()

        self.set_source(iron_graph.source)

    def add_nodes_sequence(
        self, reactant: str, product: str, chemical_equation: ChemicalEquation
    ) -> None:
        """To add Molecule instances of reactants and products of a ChemicalEquation as nodes."""
        # The Molecule instances for the reactant and product of reference are initiated

        molecule_constructor = MoleculeConstructor(
            molecular_identity_property_name="smiles"
        )

        reactant_canonical = molecule_constructor.build_from_molecule_string(
            molecule_string=reactant, inp_fmt="smiles"
        )
        product_canonical = molecule_constructor.build_from_molecule_string(
            molecule_string=product, inp_fmt="smiles"
        )

        self.add_node((reactant_canonical, [chemical_equation]))
        self.add_node((chemical_equation, [product_canonical]))

    def get_leaves(self) -> list:
        """To get the list of leaves of a BipartiteSynGraph instance."""
        connections = []
        for connection_set in self.graph.values():
            connections.extend(iter(connection_set))
        return [
            reac
            for reac in self.graph.keys()
            if reac not in connections and isinstance(reac, Molecule)
        ]

    def get_reaction_roots(self) -> list:
        """To get the list of ChemicalEquation nodes producing the molecule roots"""
        mol_roots = self.get_roots()
        reaction_roots = set()
        for m in mol_roots:
            reaction_roots.update(
                [parent for parent, children in self if m in children]
            )
        return list(reaction_roots)

    def remove_isolated_ces(self) -> None:
        """To remove ChemicalEquation nodes and their reactants/products if they form isolated sequences.
        This happens when a ChemicalEquation produces the reagents of another"""
        mol_roots = self.get_roots()
        all_ces = [
            node
            for node in self.get_unique_nodes()
            if isinstance(node, ChemicalEquation)
        ]
        reaction_roots = self.get_reaction_roots()
        leaves = self.get_leaves()
        if len(mol_roots) > 1:
            for m in mol_roots:
                # identifying the molecule root that is reagent of another reaction
                if [ce for ce in all_ces if m.uid in ce.role_map["reagents"]]:
                    # identifying the chemical equation that has the reagent as product and removing it
                    ce_to_remove = next(
                        r for r in reaction_roots if m.uid in r.role_map["products"]
                    )
                    nodes_to_remove = [m, ce_to_remove]
                    reactants = ce_to_remove.get_reactants()
                    nodes_to_remove.extend(reactants)
                    nodes_to_remove = self.find_dangling_sequence(
                        reactants, leaves, nodes_to_remove
                    )
                    [self.remove_node(n.uid) for n in nodes_to_remove]

    def find_dangling_sequence(self, reactants, leaves, nodes_to_remove):
        """To identify the nodes that are part of a disconnected sequence of nodes."""
        while intermediates := {r for r in reactants if r not in leaves}:
            for intermediate in intermediates:
                parent_ces = self.find_parent_node(intermediate)
                nodes_to_remove.extend(parent_ces)
                for parent in parent_ces:
                    reactants = parent.get_reactants()
                    nodes_to_remove.extend(reactants)

        else:
            return nodes_to_remove


class MonopartiteReacSynGraph(SynGraph):
    """SynGraph subclass representing a Monopartite (ChemicalEquation nodes only) SynGraph"""

    def builder_from_reaction_list(
        self, chemical_equations: List[ChemicalEquation]
    ) -> None:
        """To build a MonopartiteReacSynGraph from a list of ChemicalEquation objects"""
        for ch_equation in chemical_equations:
            next_ch_equations: List[ChemicalEquation] = []
            for c in chemical_equations:
                next_ch_equations.extend(
                    c
                    for m in c.role_map["reactants"]
                    if m in ch_equation.role_map["products"]
                )

            self.add_node((ch_equation, next_ch_equations))
        # this is to avoid disconnected nodes due to reactions producing reagents of other reactions
        self.remove_isolated_ces()

        self.set_source(str(self.uid))

    def builder_from_iron(self, iron_graph: Iron) -> None:
        """To build a MonopartiteReacSynGraph from an Iron instance."""
        connections = [edge.direction.tup for id_e, edge in iron_graph.edges.items()]
        for node1, node2 in connections:
            all_reactants, all_products = self.find_reactants_products(
                iron_graph, node1, node2, connections
            )

            check = [item for item in all_reactants if item in all_products]
            if not check:
                chemical_equation1 = get_reaction_instance(all_reactants, all_products)

                # Searching for the connections in which the product of the "parent" reaction is a reactant
                next_connections = [
                    edge.direction.tup
                    for edge in iron_graph.edges.values()
                    if edge.direction.tup[0] == node2
                ]
                if next_connections:
                    for node_a, node_b in next_connections:
                        all_reactants2, all_products2 = self.find_reactants_products(
                            iron_graph, node_a, node_b, connections
                        )

                        check = [
                            item for item in all_reactants2 if item in all_products2
                        ]
                        if not check:
                            chemical_equation2 = get_reaction_instance(
                                all_reactants2, all_products2
                            )
                            self.add_node((chemical_equation1, [chemical_equation2]))
                else:
                    self.add_node((chemical_equation1, []))

        self.set_source(iron_graph.source)

    def get_leaves(self) -> list:
        """To get the list of ChemicalEquation leaves in a MonopartiteSynGraph."""
        connections = []
        for connection_set in self.graph.values():
            connections.extend(iter(connection_set))
        return list({tup[0] for tup in self if tup[0] not in connections})

    def get_molecule_roots(self) -> list:
        """To get the list of Molecules roots in a MonopartiteReacSynGraph."""
        reaction_roots = self.get_roots()
        mol_roots = {m for reaction in reaction_roots for m in reaction.get_products()}
        return list(mol_roots)

    def get_molecule_leaves(self) -> list:
        """To get the list of Molecule leaves in a MonopartiteReacSynGraph."""
        all_reactants = set()
        all_products = set()
        unique_ces = self.get_unique_nodes()
        for ce in unique_ces:
            all_reactants.update(set(ce.get_reactants()))
            all_products.update(set(ce.get_products()))

        return [m for m in all_reactants if m not in all_products]

    def remove_isolated_ces(self) -> None:
        """To remove sequences of nodes if they form isolated branches.
        This happens when a ChemicalEquation produces the reagents of another"""
        mol_roots = self.get_molecule_roots()
        all_ces = self.get_unique_nodes()
        reaction_roots = self.get_roots()
        reaction_leaves = self.get_leaves()
        if len(mol_roots) > 1:
            for m in mol_roots:
                # identifying the molecule root that is reagent of another reaction
                if [ce for ce in all_ces if m.uid in ce.role_map["reagents"]]:
                    # identifying the chemical equation that has the reagent as product and removing it
                    ce_to_remove = next(
                        r for r in reaction_roots if m.uid in r.role_map["products"]
                    )
                    nodes_to_remove = [ce_to_remove]
                    nodes_to_remove = self.find_dangling_sequence(
                        ce_to_remove, reaction_leaves, nodes_to_remove
                    )
                    [self.remove_node(n.uid) for n in nodes_to_remove]

    def find_dangling_sequence(self, ce_to_remove, leaves, nodes_to_remove):
        """To identify the nodes that are part of a disconnected sequence of nodes."""
        while ce_to_remove not in leaves:
            if parent_nodes := self.find_parent_node(ce_to_remove):
                nodes_to_remove.extend(parent_nodes)
                for parent_node in parent_nodes:
                    ce_to_remove = parent_node
        else:
            return nodes_to_remove


class MonopartiteMolSynGraph(SynGraph):
    """SynGraph subclass representing a Monopartite (Molecule nodes only) SynGraph"""

    def builder_from_reaction_list(self, chemical_equations: list):
        for ch_equation in chemical_equations:
            products = ch_equation.get_products()
            reactants = ch_equation.get_reactants()

            for reactant in reactants:
                self.add_node((reactant, products))
        self.add_molecular_roots()
        # self.remove_isolate_nodes()

        self.set_source(str(self.uid))

    def builder_from_iron(self, iron_graph: Iron) -> None:
        connections = [edge.direction.tup for id_e, edge in iron_graph.edges.items()]
        for node1, node2 in connections:
            reactant = iron_graph.nodes[node1].properties["node_smiles"]
            product = iron_graph.nodes[node2].properties["node_smiles"]
            all_reactants, all_products = self.find_reactants_products(
                iron_graph, node1, node2, connections
            )

            check = [item for item in all_reactants if item in all_products]
            if not check:
                molecule_constructor = MoleculeConstructor(
                    molecular_identity_property_name="smiles"
                )

                reactant_canonical = molecule_constructor.build_from_molecule_string(
                    molecule_string=reactant, inp_fmt="smiles"
                )
                product_canonical = molecule_constructor.build_from_molecule_string(
                    molecule_string=product, inp_fmt="smiles"
                )

                self.add_node((reactant_canonical, [product_canonical]))
        self.add_molecular_roots()

        self.set_source(iron_graph.source)

    def get_leaves(self) -> list:
        """To get the list of leaves of a MonopartiteMolSynGraph instance."""
        connections = []
        for connection_set in self.graph.values():
            connections.extend(iter(connection_set))
        return [reac for reac in self.graph.keys() if reac not in connections]


def get_reaction_instance(
    reactants: List[str], products: List[str]
) -> ChemicalEquation:
    """
    To create a ChemicalEquation instance from a list of reactants and products smiles.

    Parameters:
    -----------
    reactants: List[str]
        The list of smiles of the reactants of the reaction
    products: List[str]
        The list of smiles of the products of the reaction

    Returns:
    ---------
    chemical_equation: ChemicalEquation
        The corresponding ChemicalEquation object
    """

    # The ChemicalEquation instance is created
    reaction_string = ">".join([".".join(reactants), ".".join([]), ".".join(products)])
    chemical_equation_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles"
    )
    return chemical_equation_constructor.build_from_reaction_string(
        reaction_string=reaction_string, inp_fmt="smiles"
    )


if __name__ == "__main__":
    print("main")
