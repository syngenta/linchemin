from dataclasses import dataclass, field
from typing import Optional

"""
Module containing the definitions of the Iron class.

"""


@dataclass(frozen=True, order=True)
class Properties:
    pass


class Direction:
    """Class representing the direction of an Iron edge.

    Attributes:
    ------------
    string: a string representing the direction ('parent_node>child_node')

    tup: a tuple representing the direction (parent_node, child_node)
    """

    def __init__(self, dir_string: str):
        """
        Parameters:
        ------------
        dir_string: str
            The string corresponding to the direction in the form 'parent_node>child_node'
        """
        self.string = dir_string
        self.tup = (dir_string.split(">")[0], dir_string.split(">")[1])

    def __eq__(self, other):
        return isinstance(other, Direction) and self.string == other.string


@dataclass(frozen=True, order=True)
class Node:
    """Class representing a node in an Iron instance.

    Attributes:
    -------------
    iid: a string representing the id of the node

    properties: a dictionary to store node properties

    labels: a list to store the labels of the node
    """

    iid: str = field()
    properties: dict = field(hash=False, compare=False, repr=True)
    labels: list = field(default_factory=list, repr=True)

    def get_properties_names(self):
        self.properties.keys()


@dataclass(frozen=True, order=True)
class Edge:
    """Class representing an edge in an Iron instance.

    Attributes:
    -------------
    iid: a string representing the id of the edge

    a_iid: a string representing the id of the parent node

    b_iid: a string representing the id of the child node

    direction: a Direction object indicating the direction of the edge

    properties: a dictionary to store the properties of the edge

    labels: a list to store the labels of the edge
    """

    iid: str
    a_iid: str
    b_iid: str
    direction: Direction
    properties: dict = field(hash=False, compare=False, repr=True)
    labels: list = field(default_factory=list, repr=True)

    def get_properties_names(self):
        self.properties.keys()


class Iron:
    """Class representing a graph in the Internal Representation Of Networks (IRON) format.

    Attributes:
    -------------
    nodes: a dictionary storing the nodes of the graph

    edges: a dictionary storing the edges of the graph

    name: a string representing the name of the graph
    """

    def __init__(self):
        """
        Parameters:
        -------------
        None. When a new Iron instance is created, an empty graph is initialized.
        It can be populated by adding nodes and edges.
        """
        self.nodes = {}
        self.edges = {}
        self.name: Optional[str] = None

    def __str__(self):
        return f"Nodes: {self.nodes} \nEdges: {self.edges}"

    def add_node(self, k: str, node: Node):
        """To add a node to an Iron instance."""
        self.nodes[k] = node

    def add_edge(self, k: str, edge: Edge):
        """To add an edge to an Iron instance."""
        self.edges[k] = edge

    def i_node_number(self):
        """To get the number of nodes in an Iron instance"""
        return len(self.nodes)

    def i_edge_number(self) -> int:
        """To get the number of edges in an Iron instance"""
        return len(self.edges)

    def get_neighbors(self, a: Node) -> list:
        """To get the list of nodes sharing an edge with a given node"""
        neigh_lst = []
        for id_e, edge in self.edges.items():
            if edge.a_iid == str(a):
                neigh_lst.append(edge.b_iid)
            elif edge.b_iid == str(a):
                neigh_lst.append(edge.a_iid)
        return neigh_lst

    def get_child_nodes(self, b_iid: str) -> list:
        """To get the list of 'child' nodes of a given node"""
        return [edge.a_iid for id_e, edge in self.edges.items() if edge.b_iid == b_iid]

    def get_parent_nodes(self, a_iid: str) -> list:
        """To get the list of 'parent' nodes of a given node"""
        return [edge.b_iid for id_e, edge in self.edges.items() if edge.a_iid == a_iid]

    def get_edge_id(self, a: Node, b: Node) -> list:
        """To get the list of edge ids connecting the nodes a and b (direction ignored)"""
        ids_lst = []
        for id_e, edge in self.edges.items():
            if (
                edge.a_iid == str(a)
                and edge.b_iid == str(b)
                or edge.b_iid == str(a)
                and edge.a_iid == str(b)
            ):
                ids_lst.append(id_e)
        return ids_lst

    def get_degree_sequence(self) -> list:
        """To get the degree sequence of an Iron instance."""
        degrees = []
        for id_n in self.nodes:
            node_degree = len(self.get_neighbors(id_n))
            degrees.append(node_degree)
        return sorted(degrees, reverse=True)
