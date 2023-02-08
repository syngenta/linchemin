# Standard library import
import datetime
from abc import ABC, abstractmethod
from collections import defaultdict

import linchemin.utilities as utilities
from linchemin.cgu.iron import Iron
from linchemin.cheminfo.molecule import Molecule, MoleculeConstructor

# Local import
from linchemin.cheminfo.reaction import ChemicalEquation, ChemicalEquationConstructor

"""
Module containing the implementation of the SynGraph data model and its sub-types: bipartite, monopartite reactions
and monopartite molecules.

    Abstract classes:
        SynGraph
        
    Classes:
        BipartiteSynGraph(SynGraph)
        MonopartiteReacSynGraph(SynGraph)
        MonopartiteMolSynGraph(SynGraph)
        
"""


class SynGraph(ABC):
    """Abstract class implementing the homonym data model as a dictionary of sets.

    Attributes:
            graph: a dictionary of sets

            source: a string containing the sources of the graph

            uid: a string uniquely identifying the SynGraph instance based on the underlying graph
    """

    def __init__(self, initiator=None):
        """
        Parameters:
            initiator: object to initialize a SynGraph instance (optional, default: None).
                It can be:
                    (i) an Iron instance
                    (ii) a list of dictionaries of reaction strings in the form
                        [{'id': reaction_id, 'reaction_string': reaction_smiles, 'inp_fmt': 'smiles}]

                If no arguments are passed, an empty graph is initialized.

        """
        self.source = str
        self.graph = defaultdict(set)

        if initiator is not None and isinstance(initiator, Iron):
            self.builder_from_iron(initiator)

        if initiator is not None and isinstance(initiator, list):
            chemical_equations = []
            for d in initiator:
                chemical_eq_constructor = ChemicalEquationConstructor(
                    identity_property_name=d["inp_fmt"]
                )
                chemical_equations.append(
                    chemical_eq_constructor.build_from_reaction_string(
                        reaction_string=d["reaction_string"], inp_fmt=d["inp_fmt"]
                    )
                )
            self.builder_from_reaction_list(chemical_equations)

    @property
    def uid(self):
        tups = []
        for parent, children in self.graph.items():
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

    def builder_from_iron(self, iron_graph):
        pass

    def builder_from_reaction_list(self, chemical_equations: list):
        pass

    def get_roots(self) -> list:
        """To retrieve the list of 'root' nodes of a SynGraph instance"""
        return [tup[0] for tup in self if tup[1] == set()]

    def get_leaves(self) -> list:
        """To retrieve the list of 'leaf' nodes of a SynGraph instance"""
        pass

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
        for r, connections in self.graph.items():
            if connections:
                text = text + "{} -> {} \n".format(r, *connections)
            else:
                text = text + f"{r}\n"
        return text

    def add_node(self, nodes_tup: tuple):
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


class BipartiteSynGraph(SynGraph):
    """SynGraph subclass representing a Bipartite (Molecule and ChemicalEquation nodes) SynGraph"""

    def builder_from_reaction_list(self, chemical_equations: list[ChemicalEquation]):
        """To build a BipartiteSynGraph instance from a list of ChemicalEquation objects"""
        for ch_equation in chemical_equations:
            products = [
                prod
                for h, prod in ch_equation.molecules.items()
                if h in ch_equation.roles["products"]
            ]
            self.add_node((ch_equation, products))
            reactants = [
                reac
                for h, reac in ch_equation.molecules.items()
                if h in ch_equation.roles["reactants"]
            ]

            for reactant in reactants:
                self.add_node((reactant, [ch_equation]))

        roots = []
        for products in self.graph.values():
            roots.extend(
                prod
                for prod in products
                if prod not in list(self.graph.keys()) and isinstance(prod, Molecule)
            )

        for root in roots:
            self.add_node((root, []))

    def builder_from_iron(self, iron_graph):
        """To build a BipartiteSynGraph instance from an Iron instance"""
        connections = [edge.direction.tup for id_e, edge in iron_graph.edges.items()]
        for node1, node2 in connections:
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
            check = [item for item in all_reactants if item in all_products]
            if not check:
                chemical_equation = get_reaction_instance(all_reactants, all_products)

                reactant = iron_graph.nodes[node1].properties["node_smiles"]
                product = iron_graph.nodes[node2].properties["node_smiles"]

                self.add_nodes_sequence(reactant, product, chemical_equation)

        roots = []
        for products in self.graph.values():
            roots.extend(
                prod
                for prod in products
                if prod not in list(self.graph.keys()) and isinstance(prod, Molecule)
            )

        for root in roots:
            self.add_node((root, []))

        self.set_source(iron_graph.source)

    def add_nodes_sequence(
        self, reactant: str, product: str, chemical_equation: ChemicalEquation
    ):
        """To initiate the Molecule instances for the reactant and product of reference
        and initiate the addition of the nodes."""
        # The Molecule instances for the reactant and product of reference are initiated

        molecule_constructor = MoleculeConstructor(identity_property_name="smiles")

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


class MonopartiteReacSynGraph(SynGraph):
    """SynGraph subclass representing a Monopartite (ChemicalEquation nodes only) SynGraph"""

    def builder_from_reaction_list(self, chemical_equations: list):
        """To build a MonopartiteReacSynGraph from a list of ChemicalEquation objects"""
        for ch_equation in chemical_equations:
            next_ch_equations = []
            for c in chemical_equations:
                next_ch_equations.extend(
                    c
                    for m in c.roles["reactants"]
                    if m in ch_equation.roles["products"]
                )

            self.add_node((ch_equation, next_ch_equations))

    def builder_from_iron(self, iron_graph):
        """To build a MonopartiteReacSynGraph from an Iron instance."""
        connections = [edge.direction.tup for id_e, edge in iron_graph.edges.items()]
        for node1, node2 in connections:
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
                        all_reactants2 = [
                            iron_graph.nodes[tup[0]].properties["node_smiles"]
                            for tup in connections
                            if tup[1] == node_b
                        ]
                        all_products2 = [
                            iron_graph.nodes[tup[1]].properties["node_smiles"]
                            for tup in connections
                            if tup[0] == node_a
                        ]

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
        """To get the list of Reaction leaves in a MonopartiteSynGraph."""
        connections = []
        for connection_set in self.graph.values():
            connections.extend(iter(connection_set))
        return [tup[0] for tup in self if tup[0] not in connections]

    def get_molecule_roots(self) -> list:
        """To get the list of Molecules roots in a MonopartiteReacSynGraph."""
        roots = []
        root_reactions = [tup[0] for tup in self if tup[1] == set()]
        for reaction in root_reactions:
            roots = roots + [
                mol
                for h, mol in reaction.molecules.items()
                if h in reaction.roles["products"]
            ]

        return roots

    def get_molecule_leaves(self) -> list:
        """To get the list of Molecule leaves in a MonopartiteReacSynGraph."""
        leaves = []
        for reac in self.graph:
            if reac not in self.graph.values():
                leaves = leaves + [
                    mol
                    for h, mol in reac.molecules.items()
                    if h in reac.roles["reactants"]
                ]

        return leaves


class MonopartiteMolSynGraph(SynGraph):
    """SynGraph subclass representing a Monopartite (Molecule nodes only) SynGraph"""

    def builder_from_reaction_list(self, chemical_equations: list):
        for ch_equation in chemical_equations:
            products = [
                prod
                for h, prod in ch_equation.molecules.items()
                if h in ch_equation.roles["products"]
            ]
            reactants = [
                reac
                for h, reac in ch_equation.molecules.items()
                if h in ch_equation.roles["reactants"]
            ]
            for reactant in reactants:
                self.add_node((reactant, products))

        roots = []
        for products in self.graph.values():
            roots.extend(
                prod
                for prod in products
                if prod not in list(self.graph.keys()) and isinstance(prod, Molecule)
            )

        for root in roots:
            self.add_node((root, []))

    def builder_from_iron(self, iron_graph):
        connections = [edge.direction.tup for id_e, edge in iron_graph.edges.items()]
        for node1, node2 in connections:
            reactant = iron_graph.nodes[node1].properties["node_smiles"]
            product = iron_graph.nodes[node2].properties["node_smiles"]

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
            check = [item for item in all_reactants if item in all_products]
            if not check:
                molecule_constructor = MoleculeConstructor(
                    identity_property_name="smiles"
                )

                reactant_canonical = molecule_constructor.build_from_molecule_string(
                    molecule_string=reactant, inp_fmt="smiles"
                )
                product_canonical = molecule_constructor.build_from_molecule_string(
                    molecule_string=product, inp_fmt="smiles"
                )

                self.add_node((reactant_canonical, [product_canonical]))

        roots = []
        for products in self.graph.values():
            roots.extend(
                prod
                for prod in products
                if prod not in list(self.graph.keys()) and isinstance(prod, Molecule)
            )

        for root in roots:
            self.add_node((root, []))

        self.set_source(iron_graph.source)

    def get_leaves(self) -> list:
        """To get the list of leaves of a MonopartiteMolSynGraph instance."""
        connections = []
        for connection_set in self.graph.values():
            connections.extend(iter(connection_set))
        return [reac for reac in self.graph.keys() if reac not in connections]


def get_reaction_instance(reactants: list, products: list) -> ChemicalEquation:
    """Takes the lists of reactants and products of a reaction and create the ChemicalEquation instance.

    Parameters:
        reactants: a list of smiles corresponding to the reactants of the reaction
        products: a list of smiles corresponding to the products of the reaction

    Return:
        chemical_equation: an instance of the ChemicalEquation class
    """

    # The ChemicalEquation instance is created
    reaction_string = ">".join([".".join(reactants), ".".join([]), ".".join(products)])
    chemical_equation_constructor = ChemicalEquationConstructor(
        identity_property_name="smiles"
    )
    chemical_equation = chemical_equation_constructor.build_from_reaction_string(
        reaction_string=reaction_string, inp_fmt="smiles"
    )

    return chemical_equation


def merge_syngraph(list_syngraph: list) -> SynGraph:
    """Takes a list of SynGraph objects and returns a new 'merged' SynGraph.

    Parameters:
        list_syngraph: a list
            The input SynGraph objects to be merged

    Returns:
        merged: a SynGraph object
            The new SynGraph object resulting from the merging of the input graphs;
            keys and connections are unique (no duplicates)
    """
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
        for step in syngraph:
            merged.add_node(step)
    return merged


# Factory to extract reaction strings from a syngraph object
class ReactionsExtractor(ABC):
    """Abstract class for extracting a list of dictionary of reaction strings from a SynGraph object"""

    @abstractmethod
    def extract(self, syngraph) -> list:
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
            {"id": n, "reaction_string": reaction, "inp_fmt": "smiles"}
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
            {"id": n, "reaction_string": reaction, "inp_fmt": "smiles"}
            for n, reaction in enumerate(sorted_reactions)
        ]


class ReactionsExtractorFromMonopartiteMolecules(ReactionsExtractor):
    """ReactionsExtractor subclass to handle MonopartiteMolSynGraph objects"""

    ## TODO: check behavior when the same reactant appears twice (should be solved once the stoichiometry attribute is developed)
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
                    identity_property_name="smiles"
                )
                chemical_equation = (
                    chemical_equation_constructor.build_from_reaction_string(
                        reaction_string=reaction_string, inp_fmt="smiles"
                    )
                )
                unique_reactions.add(chemical_equation)
        sorted_reactions = sorted([reaction.smiles for reaction in unique_reactions])
        return [
            {"id": n, "reaction_string": reaction, "inp_fmt": "smiles"}
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


def extract_reactions_from_syngraph(syngraph: SynGraph) -> list:
    """Takes a SynGraph object and returns a list of dictionaries of the involved chemical reactions.

    Parameters:
        syngraph: a SynGraph object (MonopartiteReacSynGraph or BipartiteSynGraph)

    Returns:
        reactions: a list of dictionary in the form
                   [{'id': reaction_id, 'reaction_string': reaction_smiles, 'inp_fmt': 'smiles}]
    """
    factory = ExtractorFactory()

    return factory.extract_reactions(syngraph)


if __name__ == "__main__":
    print("main")
