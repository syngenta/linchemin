from abc import ABC, abstractmethod

from linchemin.cgu.syngraph import (
    BipartiteSynGraph,
    MonopartiteMolSynGraph,
    MonopartiteReacSynGraph,
    SynGraph,
)
from linchemin.cheminfo.molecule import Molecule
from linchemin.cheminfo.reaction import ChemicalEquation

"""
Module containing all the functions to convert from and into different SynGraph data models:
-bipartite molecules and reactions
-monopartite molecules
-monopartite reactions
"""


# Abstract strategy class
class Converter(ABC):
    """Abstract class for data model converters"""

    input_data_model: SynGraph
    output_data_model: SynGraph

    out_datamodels = {
        "bipartite": BipartiteSynGraph,
        "monopartite_reactions": MonopartiteReacSynGraph,
        "monopartite_molecules": MonopartiteMolSynGraph,
    }

    @abstractmethod
    def convert(self, graph: SynGraph):
        pass


# Concrete strategy classes
class BipartiteToMonopartiteReactions(Converter):
    """Converter subclass to convert a bipartite SynGraph in a monopartite reactions SynGraph"""

    input_data_model = BipartiteSynGraph
    output_data_model = MonopartiteReacSynGraph

    def convert(self, graph: BipartiteSynGraph):
        """To build a MonopartiteSynGraph with only ReactionStep nodes from a bipartite SynGraph."""
        mp_graph = MonopartiteReacSynGraph()
        for r, connections in graph.graph.items():
            if isinstance(r, ChemicalEquation):
                if connections:
                    for c in connections:
                        if products := [
                            set_products
                            for mol, set_products in graph.graph.items()
                            if mol == c
                        ]:
                            r2 = [
                                p
                                for p in products[0]
                                if p != r and isinstance(p, ChemicalEquation)
                            ]
                            mp_graph.add_node((r, r2))

                else:
                    mp_graph.add_node((r, []))

        mp_graph.source = graph.source
        return mp_graph


class BipartiteToMonopartiteMolecules(Converter):
    """Converter subclass to convert a bipartite SynGraph in a monopartite molecules SynGraph"""

    input_data_model = BipartiteSynGraph
    output_data_model = MonopartiteMolSynGraph

    def convert(self, graph: BipartiteSynGraph):
        """To build a MonopartiteSynGraph with only Molecules nodes from a bipartite SynGraph."""
        mp_graph = MonopartiteMolSynGraph()
        for parent, children in graph.graph.items():
            if isinstance(parent, Molecule):
                if children:
                    for child in children:
                        if products := [
                            set_products
                            for mol, set_products in graph.graph.items()
                            if mol == child
                        ]:
                            r2 = [
                                p
                                for p in products[0]
                                if p != parent and isinstance(p, Molecule)
                            ]
                            mp_graph.add_node((parent, r2))

                else:
                    mp_graph.add_node((parent, []))

        mp_graph.source = graph.source
        return mp_graph


class MonopartiteMoleculesToMonopartiteReactions(Converter):
    """Converter subclass to convert a monopartite reactions SynGraph in a monopartite molecules SynGraph"""

    input_data_model = MonopartiteMolSynGraph
    output_data_model = MonopartiteReacSynGraph

    def convert(self, graph: MonopartiteMolSynGraph):
        raise NotImplementedError


class MonopartiteMoleculesToBiparite(Converter):
    """Converter subclass to convert a monopartite reactions SynGraph in a bipartite SynGraph"""

    input_data_model = MonopartiteMolSynGraph
    output_data_model = BipartiteSynGraph

    def convert(self, graph: MonopartiteMolSynGraph):
        raise NotImplementedError


class MonopartiteReactionsToMonopartiteMolecules(Converter):
    """Converter subclass to convert a monopartite reactions SynGraph in a monopartite molecules SynGraph"""

    input_data_model = MonopartiteReacSynGraph
    output_data_model = MonopartiteMolSynGraph

    def convert(self, graph: MonopartiteReacSynGraph):
        mp_mol_syngraph = MonopartiteMolSynGraph()

        for parent, children in graph.graph.items():
            mol_reactants = [
                mol
                for h, mol in parent.molecules.items()
                if h in parent.roles["reactants"]
            ]
            mol_products = [
                mol
                for h, mol in parent.molecules.items()
                if h in parent.roles["products"]
            ]

            for reactant in mol_reactants:
                for product in mol_products:
                    mp_mol_syngraph.add_node((reactant, [product]))

            if children:
                for child in children:
                    mol_reactants = [
                        mol
                        for h, mol in child.molecules.items()
                        if h in parent.roles["reactants"]
                    ]
                    mol_products = [
                        mol
                        for h, mol in child.molecules.items()
                        if h in parent.roles["products"]
                    ]

                    for reactant in mol_reactants:
                        for product in mol_products:
                            mp_mol_syngraph.add_node((reactant, [product]))

            else:
                for product in mol_products:
                    mp_mol_syngraph.add_node((product, []))

        mp_mol_syngraph.source = graph.source
        return mp_mol_syngraph


class MonopartiteReactionsToBipartite(Converter):
    """Converter subclass to convert a monopartite reactions SynGraph in a bipartite SynGraph"""

    input_data_model = MonopartiteReacSynGraph
    output_data_model = BipartiteSynGraph

    def convert(self, graph: MonopartiteReacSynGraph):
        bp_graph = BipartiteSynGraph()

        for parent, children in graph.graph.items():
            mol_reactants = [
                mol
                for h, mol in parent.molecules.items()
                if h in parent.roles["reactants"]
            ]
            mol_products = [
                mol
                for h, mol in parent.molecules.items()
                if h in parent.roles["products"]
            ]

            for reactant in mol_reactants:
                bp_graph.add_node((reactant, [parent]))
            for product in mol_products:
                bp_graph.add_node((parent, [product]))

            if children:
                for child in children:
                    mol_reactants = [
                        mol
                        for h, mol in child.molecules.items()
                        if h in child.roles["reactants"]
                    ]
                    mol_products = [
                        mol
                        for h, mol in child.molecules.items()
                        if h in child.roles["products"]
                    ]

                    for reactant in mol_reactants:
                        bp_graph.add_node((reactant, [child]))
                    for product in mol_products:
                        bp_graph.add_node((child, [product]))

            else:
                for product in mol_products:
                    bp_graph.add_node((product, []))

        bp_graph.source = graph.source
        return bp_graph


# Context class
class Conversion:
    def __init__(self, input_data_model, output_data_model):
        c = [
            subclass
            for subclass in Converter.__subclasses__()
            if subclass.input_data_model == input_data_model
            and subclass.output_data_model == output_data_model
        ][0]
        self.converter = c()

    def apply_conversion(self, graph):
        return self.converter.convert(graph)


def converter(graph: SynGraph, out_data_model: str):
    """Takes a SynGraph and convert it into the desired data model.

    Parameters:
        graph: a SynGraph object
            It is the input graph as instance of one of the available SynGraph subclasses

        out_data_model: a string
            It indicates the desired output data model. Available data models are:
            'bipartite, 'monopartite_reactions', and 'monopartite_molecules'

    Returns:
        a SynGraph object in the desired data model

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
