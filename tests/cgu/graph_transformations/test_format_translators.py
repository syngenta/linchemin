import networkx as nx
import pydot
import pytest

from linchemin.cgu.graph_transformations.format_translators import (
    AzRetro,
    IbmRetro,
    MitRetro,
    Networkx,
    PyDot,
    ReaxysRT,
    Sparrow,
    get_input_translators,
    get_output_translators,
)
from linchemin.cgu.iron import Direction, Edge, Iron, Node


@pytest.fixture
def iron_test_instance():
    """Returns a graph toy model as Iron instance."""

    nodes = {
        "1": Node(
            iid="1", properties={"id": "1", "node_smiles": "smile_1"}, labels=["C"]
        ),
        "2": Node(
            iid="2", properties={"id": "2", "node_smiles": "smile_2"}, labels=["C"]
        ),
        "3": Node(
            iid="3", properties={"id": "3", "node_smiles": "smile_3"}, labels=["C"]
        ),
        "5": Node(
            iid="5", properties={"id": "5", "node_smiles": "smile_5"}, labels=["C"]
        ),
        "102": Node(
            iid="102",
            properties={"id": "102", "node_smiles": "smile_102"},
            labels=["R"],
        ),
        "104": Node(
            iid="104",
            properties={"id": "104", "node_smiles": "smile_104"},
            labels=["R"],
        ),
    }
    edges = {
        "1": Edge(
            iid="1",
            a_iid="102",
            b_iid="1",
            direction=Direction("102>1"),
            properties={},
            labels=[],
        ),
        "2": Edge(
            iid="2",
            a_iid="2",
            b_iid="102",
            direction=Direction("2>102"),
            properties={},
            labels=[],
        ),
        "3": Edge(
            iid="3",
            a_iid="3",
            b_iid="102",
            direction=Direction("3>102"),
            properties={},
            labels=[],
        ),
        "4": Edge(
            iid="4",
            a_iid="104",
            b_iid="2",
            direction=Direction("104>2"),
            properties={},
            labels=[],
        ),
        "5": Edge(
            iid="5",
            a_iid="5",
            b_iid="104",
            direction=Direction("5>104"),
            properties={},
            labels=[],
        ),
    }

    graph_iron = Iron()

    for id_n, node in nodes.items():
        graph_iron.add_node(id_n, node)

    for id_e, edge in edges.items():
        graph_iron.add_edge(id_e, edge)
    return graph_iron


def count_dict_nodes(d, counter=0):
    """Returns the number of nodes in a graph in json format.

    Parameters:
        d: a nested dictionary representing a graph
        counter: an integer (default: 0)

    Returns:
        counter: an integer
    """
    if "children" in d:
        for child in d["children"]:
            if isinstance(child, dict):
                # Recursive call
                counter = count_dict_nodes(child, counter + 1)
            else:
                counter += 1
    return counter


def test_pydot_translation(iron_test_instance):
    """To test pydot translation"""
    translator = PyDot()
    pydot_graph = translator.from_iron(iron_test_instance)
    assert isinstance(pydot_graph, pydot.Dot)
    assert len(pydot_graph.get_nodes()) == 6
    assert len(pydot_graph.get_edges()) == 5

    new_iron = translator.to_iron(pydot_graph)
    assert new_iron.i_node_number() == 6
    assert new_iron.i_edge_number() == 5
    new_node_smiles = [
        node.properties["node_smiles"] for node in new_iron.nodes.values()
    ]
    original_node_smiles = [
        node.properties["node_smiles"] for node in iron_test_instance.nodes.values()
    ]
    assert sorted(new_node_smiles) == sorted(original_node_smiles)
    assert new_iron.name is not None


def test_nx_translation(iron_test_instance):
    """To test networkx translation"""
    translator = Networkx()
    nx_graph = translator.from_iron(iron_test_instance)
    assert isinstance(nx_graph, nx.DiGraph)
    assert nx_graph.number_of_nodes() == 6
    assert nx_graph.number_of_edges() == 5

    new_iron = translator.to_iron(nx_graph)
    assert new_iron.i_node_number() == 6
    assert new_iron.i_edge_number() == 5
    new_node_smiles = [
        node.properties["node_smiles"] for node in new_iron.nodes.values()
    ]
    original_node_smiles = [
        node.properties["node_smiles"] for node in iron_test_instance.nodes.values()
    ]
    assert sorted(new_node_smiles) == sorted(original_node_smiles)
    assert new_iron.name is not None


def test_nx_translation_single_node():
    """To test that a single node graph is correctly translated to a netwoekx object"""
    iron = Iron()
    node = Node(iid="1", properties={"id": "1", "node_smiles": "smile_1"}, labels=["C"])
    iron.add_node("1", node)
    translator = Networkx()
    nx_graph = translator.from_iron(iron)
    assert nx_graph.number_of_nodes() == 1
    assert nx_graph.number_of_edges() == 0


def test_ibm_output(ibm1_as_dict):
    translator = IbmRetro()
    route_ibm = ibm1_as_dict[0]
    iron = translator.to_iron(route_ibm)
    assert iron.i_node_number() == count_dict_nodes(route_ibm, counter=1)


def test_az_output(az_as_dict):
    translator = AzRetro()
    route_az = az_as_dict[0]
    iron = translator.to_iron(route_az)
    assert iron.i_node_number() == 5


def test_mit_output(mit_as_dict):
    translator = MitRetro()
    route_mit = mit_as_dict[0]
    iron = translator.to_iron(route_mit)
    assert iron.i_edge_number() == 7
    assert iron.i_node_number() == 8


def test_sparrow_translation():
    """To test the translation from sparrow output"""
    graph = {
        "CC(=O)NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1": {
            "Compound Nodes": ["CC(=O)Cl", "NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1"],
            "Reaction Nodes": [
                {
                    "smiles": "CC(=O)Cl.NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1>>CC(=O)NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1",
                    "conditions": ["CCN(CC)CC", "[Na+].[OH-]"],
                    "score": 0.24807077478828643,
                },
                {
                    "smiles": "O=C1c2ccccc2C(=O)N1CC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1>>NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1",
                    "conditions": [],
                    "score": 0.9898093781319756,
                },
                {"smiles": ">>CC(=O)Cl", "starting material cost ($/g)": 0.8},
            ],
            "Reward": 20,
        }
    }
    translator = Sparrow()
    iron = translator.to_iron(graph)
    assert iron.i_node_number() == 2
    assert iron.i_edge_number() == 0
    assert iron.name is not None

    new_sparrow = translator.from_iron(iron)

    assert "Reaction Nodes" in new_sparrow
    assert "Compound Nodes" in new_sparrow
    assert len(new_sparrow["Reaction Nodes"]) == 2
    assert len(new_sparrow["Compound Nodes"]) == 0
    assert new_sparrow["Reaction Nodes"] == [
        {
            "conditions": "CCN(CC)CC.[Na+].[OH-]",
            "smiles": "CC(=O)Cl.NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1>>CC(=O)NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1",
        },
        {
            "conditions": "",
            "smiles": "O=C1c2ccccc2C(=O)N1CC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1>>NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1",
        },
    ]


def test_reaxys_translation(reaxys_as_dict):
    translator = ReaxysRT()
    route = reaxys_as_dict[0]
    iron = translator.to_iron(route)
    assert iron.i_node_number() == 3
    assert iron.i_edge_number() == 2


def test_get_input():
    d = get_input_translators()
    assert isinstance(d, dict)
    assert "mit_retro" in d
    assert "pydot_visualization" not in d


def test_get_output():
    d = get_output_translators()
    assert isinstance(d, dict)
    assert "pydot_visualization" in d
    assert "az_retro" not in d
