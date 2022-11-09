from linchemin.cgu.syngraph import SynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph, BipartiteSynGraph
from linchemin.cgu.convert import converter

from collections import defaultdict


class DisconnectionGraph:
    """ Class representing a DisconnectionGraph, a graph object whose nodes are instances of the Disconnection class.

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
            if type(syngraph) not in list(SynGraph.__subclasses__()):
                raise TypeError("Invalid input type. Only SynGraph objects can be used to build a DisconnectionGraph.")

            if type(syngraph) in [BipartiteSynGraph, MonopartiteMolSynGraph]:
                syngraph = converter(syngrpah, 'monopartite_reactions')

            for parent, children in syngraph.graph.items():
                parent_disc = parent.disconnection
                children_disc = [child.disconnection for child in children]
                self.add_disc_node((parent_disc, children_disc))

    def add_disc_node(self, nodes_tup: tuple):
        """ To add a 'parent' node and its 'children' nodes to a DisconnectionGraph instance.

            Format of the input tuple: (parent_node, [child1, child2, ...])"""
        # The nodes are added to the DisconnectionGraph instance
        if nodes_tup[0] not in self.graph:
            # If the source node is not in the DisconnectionGraph instance yet, it is added with its connections
            self.graph[nodes_tup[0]] = set(nodes_tup[1])
        else:
            # otherwise, the connections are added to the pre-existing node
            for c in nodes_tup[1]:
                self.graph[nodes_tup[0]].add(c)

    def __eq__(self, other):
        """ To check if two DisconnectionGraph instances are the same"""
        return type(self) == type(other) and self.graph == other.graph

    def __str__(self):
        text = ''
        for r, connections in self.graph.items():
            if connections:
                text = text + '{} -> {} \n'.format(r, *connections)
            else:
                text = text + '{}\n'.format(r)
        return text

