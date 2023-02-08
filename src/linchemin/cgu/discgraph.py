from collections import defaultdict

from linchemin.cgu.convert import converter
from linchemin.cgu.syngraph import (
    BipartiteSynGraph,
    MonopartiteMolSynGraph,
    MonopartiteReacSynGraph,
    SynGraph,
)
from linchemin.cheminfo.disconnection import Disconnection


class MissingAtomMapping(Exception):
    """Exception to be raised if a not-mapped reaction is found while building a DisconnectionGraph"""

    pass


class DisconnectionGraph:
    """Class representing a DisconnectionGraph, a graph object whose nodes are instances of the Disconnection class.

    Attributes:

        graph: a dictionary of sets
    """

    def __init__(self, syngraph=None):
        """
        Parameters:
            syngraph: an instance of one of the subclasses of SynGraph (optional, default: None), whose
                      ChemicalEquation nodes are mapped
        """
        self.graph = defaultdict(set)

        if syngraph is not None:
            if type(syngraph) not in [MonopartiteReacSynGraph, BipartiteSynGraph]:
                raise TypeError(
                    "Invalid input type. Only MonopartiteReacSynGraph and BipartiteSynGraph objects"
                    "can be used to build a DisconnectionGraph."
                )

            if type(syngraph) == BipartiteSynGraph:
                syngraph = converter(syngrpah, "monopartite_reactions")

            for parent, children in syngraph.graph.items():
                if parent.disconnection is None:
                    raise MissingAtomMapping(
                        "A not-mapped reaction was found. Building of the Disconnection is not "
                        "possible."
                    )
                parent_disc = parent.disconnection
                if any(child.disconnection is None for child in children):
                    raise MissingAtomMapping(
                        "A not-mapped reaction was found. Building of the Disconnection is not "
                        "possible."
                    )
                children_disc = [child.disconnection for child in children]
                self.add_disc_node((parent_disc, children_disc))

    def add_disc_node(self, nodes_tup: tuple):
        """To add a 'parent' node and its 'children' nodes to a DisconnectionGraph instance.

        Format of the input tuple: (parent_node, [child1, child2, ...])
        Only disconnections instances are accepted as nodes.
        """
        if type(nodes_tup[0]) is not Disconnection or any(
            type(n) != Disconnection for n in nodes_tup[1]
        ):
            raise TypeError(
                "Invalid node type. Only Disconnection instances can be used as nodes of a "
                "DisconnectionGraph"
            )
        # The nodes are added to the DisconnectionGraph instance
        if nodes_tup[0] not in self.graph:
            # If the source node is not in the DisconnectionGraph instance yet, it is added with its connections
            self.graph[nodes_tup[0]] = set(nodes_tup[1])
        else:
            # otherwise, the connections are added to the pre-existing node
            for c in nodes_tup[1]:
                self.graph[nodes_tup[0]].add(c)

    def __eq__(self, other):
        """To check if two DisconnectionGraph instances are the same"""
        return type(self) == type(other) and self.graph == other.graph

    def __str__(self):
        text = ""
        for r, connections in self.graph.items():
            if connections:
                text = text + "{} -> {} \n".format(r, *connections)
            else:
                text = text + f"{r}\n"
        return text
