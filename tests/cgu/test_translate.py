import json
import os

import networkx as nx
import pydot
import pytest

from linchemin.cgu.iron import Direction, Edge, Iron, Node
from linchemin.cgu.syngraph import (
    BipartiteSynGraph,
    MonopartiteMolSynGraph,
    MonopartiteReacSynGraph,
    merge_syngraph,
)
from linchemin.cgu.translate import (
    az_dict_to_iron,
    get_available_data_models,
    get_available_formats,
    get_input_formats,
    get_output_formats,
    ibm_dict_to_iron,
    translator,
)
from linchemin.cheminfo.reaction import ChemicalEquation


def generate_iron_test_graph():
    """Returns a general graph (SynForest) useful for testing as a list with a single Iron instance."""

    nodes = {
        "1": Node(
            iid="1", properties={"id": "1", "node_smiles": "smile_1"}, labels=["C"]
        ),
        "2": Node(
            iid="2", properties={"id": "101", "node_smiles": "smile_101"}, labels=["R"]
        ),
        "3": Node(
            iid="3", properties={"id": "102", "node_smiles": "smile_102"}, labels=["R"]
        ),
        "4": Node(
            iid="4", properties={"id": "103", "node_smiles": "smile_103"}, labels=["R"]
        ),
        "5": Node(
            iid="5", properties={"id": "2", "node_smiles": "smile_2"}, labels=["C"]
        ),
        "6": Node(
            iid="6", properties={"id": "3", "node_smiles": "smile_3"}, labels=["C"]
        ),
        "7": Node(
            iid="7", properties={"id": "4", "node_smiles": "smile_4"}, labels=["C"]
        ),
        "8": Node(
            iid="8", properties={"id": "104", "node_smiles": "smile_104"}, labels=["R"]
        ),
        "9": Node(
            iid="9", properties={"id": "5", "node_smiles": "smile_5"}, labels=["C"]
        ),
        "10": Node(
            iid="10", properties={"id": "105", "node_smiles": "smile_105"}, labels=["R"]
        ),
        "11": Node(
            iid="11", properties={"id": "106", "node_smiles": "smile_106"}, labels=["R"]
        ),
        "12": Node(
            iid="12", properties={"id": "8", "node_smiles": "smile_8"}, labels=["C"]
        ),
        "13": Node(
            iid="13", properties={"id": "6", "node_smiles": "smile_6"}, labels=["C"]
        ),
        "14": Node(
            iid="14", properties={"id": "7", "node_smiles": "smile_7"}, labels=["C"]
        ),
        "15": Node(
            iid="15", properties={"id": "9", "node_smiles": "smile_9"}, labels=["C"]
        ),
    }
    edges = {
        "1": Edge(
            iid="1",
            a_iid="2",
            b_iid="1",
            direction=Direction("2>1"),
            properties={},
            labels=[],
        ),
        "2": Edge(
            iid="2",
            a_iid="3",
            b_iid="1",
            direction=Direction("3>1"),
            properties={},
            labels=[],
        ),
        "3": Edge(
            iid="3",
            a_iid="4",
            b_iid="1",
            direction=Direction("4>1"),
            properties={},
            labels=[],
        ),
        "4": Edge(
            iid="4",
            a_iid="5",
            b_iid="2",
            direction=Direction("5>2"),
            properties={},
            labels=[],
        ),
        "5": Edge(
            iid="5",
            a_iid="6",
            b_iid="3",
            direction=Direction("6>3"),
            properties={},
            labels=[],
        ),
        "6": Edge(
            iid="6",
            a_iid="7",
            b_iid="4",
            direction=Direction("7>4"),
            properties={},
            labels=[],
        ),
        "7": Edge(
            iid="7",
            a_iid="5",
            b_iid="3",
            direction=Direction("5>3"),
            properties={},
            labels=[],
        ),
        "8": Edge(
            iid="8",
            a_iid="5",
            b_iid="4",
            direction=Direction("5>4"),
            properties={},
            labels=[],
        ),
        "9": Edge(
            iid="9",
            a_iid="8",
            b_iid="5",
            direction=Direction("8>5"),
            properties={},
            labels=[],
        ),
        "10": Edge(
            iid="10",
            a_iid="11",
            b_iid="7",
            direction=Direction("11>6"),
            properties={},
            labels=[],
        ),
        "11": Edge(
            iid="11",
            a_iid="10",
            b_iid="7",
            direction=Direction("10>7"),
            properties={},
            labels=[],
        ),
        "12": Edge(
            iid="12",
            a_iid="9",
            b_iid="8",
            direction=Direction("9>8"),
            properties={},
            labels=[],
        ),
        "13": Edge(
            iid="13",
            a_iid="12",
            b_iid="11",
            direction=Direction("12>11"),
            properties={},
            labels=[],
        ),
        "14": Edge(
            iid="14",
            a_iid="13",
            b_iid="10",
            direction=Direction("13>10"),
            properties={},
            labels=[],
        ),
        "15": Edge(
            iid="15",
            a_iid="14",
            b_iid="10",
            direction=Direction("14>10"),
            properties={},
            labels=[],
        ),
        "16": Edge(
            iid="16",
            a_iid="10",
            b_iid="15",
            direction=Direction("10>15"),
            properties={},
            labels=[],
        ),
        "17": Edge(
            iid="17",
            a_iid="6",
            b_iid="4",
            direction=Direction("6>4"),
            properties={},
            labels=[],
        ),
    }

    graph_iron = Iron()

    for id_n, node in nodes.items():
        graph_iron.add_node(id_n, node)

    for id_e, edge in edges.items():
        graph_iron.add_edge(id_e, edge)

    return [graph_iron]


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


def test_iron_source_attribute(ibm1_path):
    """To test that the expected source is correctly assigned to an Iron instance."""
    graph = json.loads(open(ibm1_path).read())
    translated_graph = translator(
        "ibm_retro", graph[0], "iron", out_data_model="monopartite_reactions"
    )

    assert "ibm" in translated_graph.source
    assert type(translated_graph) == Iron


def test_iron_dot_translation(ibm1_path):
    print("")
    graph = json.loads(open(ibm1_path).read())
    dot_graph = translator(
        "ibm_retro", graph[2], "pydot", out_data_model="monopartite_molecules"
    )

    assert type(dot_graph) == pydot.Dot

    syngraph = translator("pydot", dot_graph, "syngraph", "monopartite_molecules")
    assert len(dot_graph.get_nodes()) == len(syngraph.graph)

    syngraph = translator("pydot", dot_graph, "syngraph", "bipartite")
    assert syngraph


def test_translating_into_nx_returns_nx_object(ibm1_path):
    """To test that the expected type of object (NetworkX DiGraph) is correctly generated."""
    graph = json.loads(open(ibm1_path).read())
    nx_graph = translator(
        "ibm_retro", graph[1], "networkx", out_data_model="monopartite_molecules"
    )
    assert type(nx_graph) == nx.classes.digraph.DiGraph

    syngraph = translator("networkx", nx_graph, "syngraph", "monopartite_reactions")
    assert syngraph and type(syngraph) == MonopartiteReacSynGraph


def test_wrong_format_name(ibm1_path):
    """To test that the correct exception is raised when an unavailable input format is requested."""
    with pytest.raises(KeyError) as ke:
        graph = json.loads(open(ibm1_path).read())
        translator("wrong_input_format", graph[1], "iron", out_data_model="bipartite")
    assert "KeyError" in str(ke.type)

    with pytest.raises(KeyError) as ke:
        graph = json.loads(open(ibm1_path).read())
        translator(
            "ibm_retro", graph[1], "wrong_output_format", out_data_model="bipartite"
        )
    assert "KeyError" in str(ke.type)


def testing_dict_to_iron(az_path, ibm1_path):
    """To test that a graph in json format is correctly converted in the expected Iron object (json -> Iron)."""
    all_routes = json.loads(open(az_path).read())
    route = all_routes[0]
    prop = {"node_type": "type", "node_smiles": "smiles"}
    iron_route = az_dict_to_iron(route, prop, iron=None, parent=None)
    assert iron_route.i_node_number() == 5

    all_routes_ibm = json.loads(open(ibm1_path).read())
    route_ibm = all_routes_ibm[0]
    prop = {"node_smiles": "smiles"}
    iron_route_ibm = ibm_dict_to_iron(route_ibm, prop, iron=None, parent=None)
    assert iron_route_ibm.i_node_number() == count_dict_nodes(route_ibm, counter=1)


def test_translating_into_iron_and_into_nx_returns_isomorphic_graphs(ibm1_path):
    """To test that a graph in json format is correctly converted in the expected Networkx object
    (json -> Iron -> Networkx)."""
    graph = json.loads(open(ibm1_path).read())
    iron_graph = translator(
        "ibm_retro", graph[1], "iron", out_data_model="monopartite_molecules"
    )
    nx_graph = translator(
        "ibm_retro", graph[1], "networkx", out_data_model="monopartite_molecules"
    )

    iron_nodes = iron_graph.i_node_number()
    nx_nodes = nx_graph.number_of_nodes()
    assert iron_nodes == nx_nodes

    iron_edges = iron_graph.i_edge_number()
    nx_edges = nx_graph.number_of_edges()
    assert iron_edges == nx_edges

    iron_deg_sequence = iron_graph.get_degree_sequence()
    nx_deg_sequence = sorted([d for n, d in nx_graph.degree()], reverse=True)
    assert iron_deg_sequence == nx_deg_sequence


def test_translating_into_iron_and_into_dot_returns_isomorphic_graphs(az_path):
    """To test that a graph in json format is correctly converted in the expected Pydot object
    (json -> Iron -> Dot)."""
    graph = json.loads(open(az_path).read())
    # Translate each route in the input graph into a list of routes in the desired format
    iron_graph = translator(
        "az_retro", graph[2], "iron", out_data_model="monopartite_molecules"
    )
    dot_graph = translator(
        "az_retro", graph[2], "pydot", out_data_model="monopartite_molecules"
    )

    iron_nodes = iron_graph.i_node_number()
    dot_nodes = len(dot_graph.get_nodes())
    assert iron_nodes == dot_nodes

    iron_edges = iron_graph.i_edge_number()
    dot_edges = len(dot_graph.get_edges())
    assert iron_edges == dot_edges


def test_none_returned_if_empty_route():
    """To test that 'None' is returned if an empty route is passed."""
    graph = {}
    t_graph_iron = translator(
        "az_retro", graph, "iron", out_data_model="monopartite_molecules"
    )
    assert t_graph_iron is None

    t_graph_nx = translator(
        "az_retro", graph, "networkx", out_data_model="monopartite_molecules"
    )
    assert t_graph_nx is None


def test_syngraph_to_syngraph(az_path):
    graph_az = json.loads(open(az_path).read())
    bp_syngraph = translator(
        "az_retro", graph_az[2], "syngraph", out_data_model="bipartite"
    )
    with pytest.raises(ValueError) as ke:
        translator(
            "syngraph", bp_syngraph, "syngraph", out_data_model="monopartite_reactions"
        )
    assert "ValueError" in str(ke.type)


def test_iron_to_syngraph(ibm1_path):
    """To test the Iron -> SynGraph transformation (mainly tested in the test_syngraph)."""
    graph_ibm = json.loads(open(ibm1_path).read())
    syngraph = translator(
        "ibm_retro", graph_ibm[2], "syngraph", out_data_model="bipartite"
    )

    assert type(syngraph) == BipartiteSynGraph
    roots = syngraph.get_roots()
    assert [root.smiles for root in roots] == ["CCNC(=O)CC"]


def test_iron_to_mp_syngraph(az_path):
    """To test the Iron -> MonopartiteReaSynGraph transformation (mainly tested in test_syngraph)."""
    graph_az = json.loads(open(az_path).read())
    mp_syngraph = translator(
        "az_retro", graph_az[2], "syngraph", out_data_model="monopartite_reactions"
    )

    assert type(mp_syngraph) == MonopartiteReacSynGraph
    roots = mp_syngraph.get_roots()
    for item in mp_syngraph.graph.keys():
        assert type(item) == ChemicalEquation
    assert [root.smiles for root in roots] == [
        "Cc1cccc(C)c1N(CC(=O)Cl)C(=O)C1CCS(=O)(=O)CC1.Nc1ccc("
        "-c2ncon2)cc1>>Cc1cccc(C)c1N(CC(=O)Nc1ccc(-c2ncon2)cc1)C(=O)C1CCS(=O)("
        "=O)CC1"
    ]


def test_one_node_iron_to_nx(ibm1_path):
    graph_ibm = json.loads(open(ibm1_path).read())
    mp_syngraph = translator(
        "ibm_retro", graph_ibm[0], "syngraph", out_data_model="monopartite_reactions"
    )

    one_node_nx = translator(
        "syngraph", mp_syngraph, "networkx", out_data_model="monopartite_reactions"
    )
    assert one_node_nx.number_of_nodes() == 1
    assert one_node_nx.number_of_edges() == 0


def test_route_depiction(az_path):
    graph_az = json.loads(open(az_path).read())
    syngraph = translator(
        "az_retro", graph_az[0], "syngraph", out_data_model="bipartite"
    )
    translator(
        "syngraph",
        syngraph,
        "pydot_visualization",
        out_data_model="monopartite_reactions",
    )

    fname_png = f"route_{syngraph.source}.png"
    assert os.path.exists(fname_png)
    fname_dot = f"route_{syngraph.source}.dot"
    assert os.path.exists(fname_dot)
    os.remove(fname_png)
    os.remove(fname_dot)


def test_get_available_formats():
    options = get_available_formats()
    assert type(options) == dict and "syngraph" in options


def test_available_data_models():
    options = get_available_data_models()
    assert type(options) == dict and "bipartite" in options


def test_translate_into_NOC_document(ibm1_path):
    graph_ibm = json.loads(open(ibm1_path).read())
    route_noc_doc = translator(
        "ibm_retro", graph_ibm[0], "noc", out_data_model="bipartite"
    )
    assert route_noc_doc
    assert type(route_noc_doc) == dict
    assert "nodes" in route_noc_doc.keys() and "edges" in route_noc_doc.keys()


def test_out_format():
    out = get_output_formats()
    assert type(out) == dict and "pydot_visualization" in out and "iron" in out


def test_in_format():
    in_f = get_input_formats()
    assert type(in_f) == dict and "ibm_retro" in in_f and "syngraph" in in_f


def test_mit_to_iron(mit_path):
    graph = json.loads(open(mit_path).read())
    iron_route = translator(
        "mit_retro", graph[0], "iron", out_data_model="monopartite_molecules"
    )
    assert iron_route.i_edge_number() == 7
    assert iron_route.i_node_number() == 8
    assert "mit" in iron_route.source

    syngraph = translator(
        "mit_retro", graph[1], "syngraph", out_data_model="monopartite_reactions"
    )
    assert syngraph
    assert len(syngraph.graph) == 4
