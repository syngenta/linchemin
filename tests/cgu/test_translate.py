import json
import os
import unittest.mock

import networkx as nx
import pydot
import pytest

from linchemin import settings
from linchemin.cgu.iron import Direction, Edge, Iron, Node
from linchemin.cgu.syngraph import BipartiteSynGraph, MonopartiteReacSynGraph
from linchemin.cgu.translate import (
    PyDot,
    TranslationError,
    az_dict_to_iron,
    get_available_data_models,
    get_available_formats,
    get_input_formats,
    get_output_formats,
    ibm_dict_to_iron,
    translator,
)
from linchemin.cheminfo.models import ChemicalEquation


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


def test_factory(ibm1_path):
    graph = {}
    with unittest.TestCase().assertLogs(
        "linchemin.cgu.translate", level="WARNING"
    ) as cm:
        t_graph_iron = translator("az_retro", graph, "iron", out_data_model="bipartite")
    assert t_graph_iron is None
    unittest.TestCase().assertEqual(len(cm.records), 1)
    unittest.TestCase().assertIn(
        "While translating from AZ", cm.records[0].getMessage()
    )

    t_graph_nx = translator(
        "az_retro", graph, "networkx", out_data_model="monopartite_molecules"
    )
    assert t_graph_nx is None

    graph = json.loads(open(ibm1_path).read())
    with pytest.raises(TranslationError) as ke:
        translator("wrong_input_format", graph[1], "iron", out_data_model="bipartite")
    assert "UnavailableFormat" in str(ke.type)

    with pytest.raises(TranslationError) as ke:
        translator("ibm_retro", graph[1], "pydot", out_data_model="bipartite_reactions")
    assert "UnavailableDataModel" in str(ke.type)


def test_iron_source_attribute(ibm1_path):
    """To test that the expected source is correctly assigned to an Iron instance."""
    graph = json.loads(open(ibm1_path).read())
    translated_graph = translator(
        "ibm_retro", graph[0], "iron", out_data_model="monopartite_reactions"
    )

    assert "ibm" in translated_graph.source
    assert type(translated_graph) == Iron


def test_dot_translation(az_path):
    nodes = {
        "0": Node(
            iid="0",
            properties={
                "node_smiles": "Cc1cccc(C)c1N(CC(=O)Nc1ccc(-c2ncon2)cc1)C(=O)C1CCS(=O)(=O)CC1"
            },
            labels=[],
        ),
        "1": Node(
            iid="1", properties={"node_smiles": "Nc1ccc(-c2ncon2)cc1"}, labels=[]
        ),
        "2": Node(
            iid="2",
            properties={"node_smiles": "Cc1cccc(C)c1N(CC(=O)Cl)C(=O)C1CCS(=O)(=O)CC1"},
            labels=[],
        ),
        "3": Node(
            iid="3", properties={"node_smiles": "O=C(O)C1CCS(=O)(=O)CC1"}, labels=[]
        ),
        "4": Node(
            iid="4", properties={"node_smiles": "Cc1cccc(C)c1NCC(=O)Cl"}, labels=[]
        ),
        "5": Node(
            iid="5", properties={"node_smiles": "Cc1cccc(C)c1NCC(=O)O"}, labels=[]
        ),
    }
    edges = {
        "0": Edge(
            iid="0",
            a_iid="1",
            b_iid="0",
            direction=Direction("1>0"),
            properties={},
            labels=[],
        ),
        "1": Edge(
            iid="1",
            a_iid="2",
            b_iid="0",
            direction=Direction("2>0"),
            properties={},
            labels=[],
        ),
        "2": Edge(
            iid="2",
            a_iid="3",
            b_iid="2",
            direction=Direction("3>2"),
            properties={},
            labels=[],
        ),
        "3": Edge(
            iid="3",
            a_iid="4",
            b_iid="2",
            direction=Direction("4>2"),
            properties={},
            labels=[],
        ),
        "4": Edge(
            iid="4",
            a_iid="5",
            b_iid="4",
            direction=Direction("5>4"),
            properties={},
            labels=[],
        ),
    }
    iron_graph = Iron()

    for id_n, node in nodes.items():
        iron_graph.add_node(id_n, node)

    for id_e, edge in edges.items():
        iron_graph.add_edge(id_e, edge)
    pydot_translator = PyDot()
    dot_graph = pydot_translator.from_iron(iron_graph)
    new_iron = pydot_translator.to_iron(dot_graph)
    assert iron_graph.nodes == new_iron.nodes
    assert iron_graph.edges == new_iron.edges

    assert type(dot_graph) == pydot.Dot
    # the two graphs have the same number of nodes
    assert iron_graph.i_node_number() == len(dot_graph.get_nodes())
    # the two graphs have the same number of edges
    assert iron_graph.i_edge_number() == iron_graph.i_edge_number()

    syngraph = translator("pydot", dot_graph, "syngraph", "monopartite_molecules")
    assert len(dot_graph.get_nodes()) == len(syngraph.graph)

    syngraph = translator("pydot", dot_graph, "syngraph", "bipartite")
    assert syngraph


def test_nx_translation(ibm1_path, az_path):
    """To test that the expected type of object (NetworkX DiGraph) is correctly generated."""
    graph = json.loads(open(ibm1_path).read())
    nx_graph = translator(
        "ibm_retro", graph[1], "networkx", out_data_model="monopartite_molecules"
    )
    assert type(nx_graph) == nx.classes.digraph.DiGraph
    for node, data in nx_graph.nodes.items():
        assert data["label"] == settings.ROUTE_MINING.molecule_node_label

    iron_graph = translator(
        "ibm_retro", graph[1], "iron", out_data_model="monopartite_molecules"
    )
    # the two graphs have the same number of nodes
    assert iron_graph.i_node_number() == nx_graph.number_of_nodes()
    # the two graphs have the same number of edges
    assert iron_graph.i_edge_number() == nx_graph.number_of_edges()

    iron_deg_sequence = iron_graph.get_degree_sequence()
    nx_deg_sequence = sorted([d for n, d in nx_graph.degree()], reverse=True)
    assert iron_deg_sequence == nx_deg_sequence
    for edge, data in nx_graph.in_edges.items():
        assert data.get("label", None) is None
    syngraph = translator("networkx", nx_graph, "syngraph", "monopartite_reactions")
    assert syngraph and type(syngraph) == MonopartiteReacSynGraph

    graph = json.loads(open(az_path).read())
    nx_graph_bp = translator(
        "az_retro", graph[1], "networkx", out_data_model="bipartite"
    )
    edge_labels = [data["label"] for data in nx_graph_bp.in_edges.values()]
    assert edge_labels.count(settings.ROUTE_MINING.reactant_edge_label) == 4
    assert edge_labels.count(settings.ROUTE_MINING.reagent_edge_label) == 0
    assert edge_labels.count(settings.ROUTE_MINING.product_edge_label) == 2


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
    assert one_node_nx.graph["source"]
    for node, data in one_node_nx.nodes.items():
        assert data["label"] == "CE"


def testing_dict_to_iron(az_path, ibm1_path):
    """To test that a graph in json format is correctly converted in the expected Iron object (json -> Iron)."""
    all_routes = json.loads(open(az_path).read())
    route = all_routes[0]
    iron_route = az_dict_to_iron(route, iron=None, parent=None)
    assert iron_route.i_node_number() == 5

    all_routes_ibm = json.loads(open(ibm1_path).read())
    route_ibm = all_routes_ibm[0]
    iron_route_ibm = ibm_dict_to_iron(route_ibm, iron=None, parent=None)
    assert iron_route_ibm.i_node_number() == count_dict_nodes(route_ibm, counter=1)


def test_syngraph_to_syngraph(az_path):
    bp_syngraph = BipartiteSynGraph()
    with pytest.raises(TranslationError) as ke:
        translator(
            "syngraph", bp_syngraph, "syngraph", out_data_model="monopartite_reactions"
        )
    assert "UnavailableTranslation" in str(ke.type)


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


def test_route_depiction(az_path):
    graph_az = json.loads(open(az_path).read())
    syngraph = translator(
        "az_retro", graph_az[0], "syngraph", out_data_model="bipartite"
    )
    translator("syngraph", syngraph, "pydot_visualization", out_data_model="bipartite")
    fname_png = f"route_{syngraph.uid}.png"
    assert os.path.exists(fname_png)
    fname_dot = f"route_{syngraph.uid}.dot"
    assert os.path.exists(fname_dot)
    os.remove(fname_png)
    os.remove(fname_dot)


def test_get_available_options():
    options = get_available_formats()
    assert type(options) == dict and "syngraph" in options
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
    assert isinstance(out, dict) and "pydot_visualization" in out and "iron" in out
    assert "mit_retro" not in out


def test_in_format():
    in_f = get_input_formats()
    assert isinstance(in_f, dict) and "ibm_retro" in in_f and "syngraph" in in_f
    assert "noc" not in in_f


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


def test_sparrow_to_iron():
    graph = {
        "CC(=O)NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1": {
            "Compounds": ["CC(=O)Cl", "NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1"],
            "Reactions": [
                {
                    "smiles": "CC(=O)Cl.NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1>>CC(=O)NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1",
                    "condition": ["CCN(CC)CC", "[Na+].[OH-]"],
                    "score": 0.24807077478828643,
                },
                {
                    "smiles": "O=C1c2ccccc2C(=O)N1CC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1>>NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1",
                    "condition": [],
                    "score": 0.9898093781319756,
                },
                {"smiles": ">>CC(=O)Cl", "starting material cost ($/g)": 0.8},
            ],
            "Reward": 20,
        }
    }
    iron = translator("sparrow", graph, "iron", "monopartite_reactions")
    assert iron.i_node_number() == 2
    assert iron.i_edge_number() == 0

    syngraph = translator("sparrow", graph, "syngraph", "monopartite_reactions")
    assert len(syngraph.graph) == 2
    assert len(syngraph.get_roots()) == 1
    assert len(syngraph.get_leaves()) == 1
    nodes = syngraph.get_unique_nodes()
    assert any(node.role_map["reagents"] != [] for node in nodes)

    syngraph = translator("sparrow", graph, "syngraph", "bipartite")
    assert len(syngraph.graph) == 6
    assert len(syngraph.get_roots()) == 1
    assert len(syngraph.get_leaves()) == 2

    assert "sparrow" in get_input_formats()


def test_iron_to_sparrow():
    reaction_list = [
        {
            "output_string": "C1COCCO1.CC(=O)O.CC1(C)OB([B:6]2[O:7][C:8]([CH3:9])([CH3:10])[C:11]([CH3:12])([CH3:13])[O:14]2)OC1(C)C.Br[C:5]1=[CH:4][CH2:3][N:2]([CH3:1])[CH2:17][C:15]1=[O:16].Cl[Pd]Cl.[CH]1[CH][CH]C(P(c2ccccc2)c2ccccc2)[CH]1.[CH]1[CH][CH]C(P(c2ccccc2)c2ccccc2)[CH]1.[Fe].[K+]>>[CH3:1][N:2]1[CH2:3][CH:4]=[C:5]([B:6]2[O:7][C:8]([CH3:9])([CH3:10])[C:11]([CH3:12])([CH3:13])[O:14]2)[C:15](=[O:16])[CH2:17]1",
            "query_id": "0",
        },
        {
            "output_string": "CC1(C)OB([C:23]2=[CH:24][CH2:25][N:26]([CH3:27])[CH2:28][C:29]2=[O:30])OC1(C)C.COCCOC.Br[c:22]1[c:3]([O:2][CH3:1])[cH:4][c:5]([O:6][CH3:7])[c:8]2[c:9](=[O:10])[cH:11][c:12](-[c:13]3[cH:14][cH:15][cH:16][cH:17][c:18]3[Cl:19])[o:20][c:21]21.[Cs+].[F-]>>[CH3:1][O:2][c:3]1[cH:4][c:5]([O:6][CH3:7])[c:8]2[c:9](=[O:10])[cH:11][c:12](-[c:13]3[cH:14][cH:15][cH:16][cH:17][c:18]3[Cl:19])[o:20][c:21]2[c:22]1[C:23]1=[CH:24][CH2:25][N:26]([CH3:27])[CH2:28][C:29]1=[O:30]",
            "query_id": "1",
        },
        {
            "output_string": "C[O:8][c:7]1[c:6]([C:5]2=[CH:4][CH2:3][N:2]([CH3:1])[CH2:28][CH:26]2[OH:27])[c:25]2[c:12]([c:10]([O:11]C)[cH:9]1)[c:13](=[O:14])[cH:15][c:16](-[c:17]1[cH:18][cH:19][cH:20][cH:21][c:22]1[Cl:23])[o:24]2.Cc1cc(C)nc(C)c1.[Li]I>>[CH3:1][N:2]1[CH2:3][CH:4]=[C:5]([c:6]2[c:7]([OH:8])[cH:9][c:10]([OH:11])[c:12]3[c:13](=[O:14])[cH:15][c:16](-[c:17]4[cH:18][cH:19][cH:20][cH:21][c:22]4[Cl:23])[o:24][c:25]23)[CH:26]([OH:27])[CH2:28]1",
            "query_id": "2",
        },
        {
            "output_string": "C[O:8][c:7]1[c:6]([C:5]2=[CH:4][CH2:3][N:2]([CH3:1])[CH2:28][CH:26]2[OH:27])[c:25]2[c:12]([c:10]([O:11]C)[cH:9]1)[c:13](=[O:14])[cH:15][c:16](-[c:17]1[cH:18][cH:19][cH:20][cH:21][c:22]1[Cl:23])[o:24]2.Cl.O.c1ccncc1>>[CH3:1][N:2]1[CH2:3][CH:4]=[C:5]([c:6]2[c:7]([OH:8])[cH:9][c:10]([OH:11])[c:12]3[c:13](=[O:14])[cH:15][c:16](-[c:17]4[cH:18][cH:19][cH:20][cH:21][c:22]4[Cl:23])[o:24][c:25]23)[CH:26]([OH:27])[CH2:28]1",
            "query_id": "3",
        },
        {
            "output_string": "C1CCOC1.[CH3:1][N:2]1[CH2:3][CH:4]=[C:5]([c:6]2[c:7]([OH:8])[cH:9][c:10]([OH:11])[c:12]3[c:13](=[O:14])[cH:15][c:16](-[c:17]4[cH:18][cH:19][cH:20][cH:21][c:22]4[Cl:23])[o:24][c:25]23)[CH:26]([OH:27])[CH2:28]1.CO>>[CH3:1][N:2]1[CH2:3][CH2:4][CH:5]([c:6]2[c:7]([OH:8])[cH:9][c:10]([OH:11])[c:12]3[c:13](=[O:14])[cH:15][c:16](-[c:17]4[cH:18][cH:19][cH:20][cH:21][c:22]4[Cl:23])[o:24][c:25]23)[CH:26]([OH:27])[CH2:28]1",
            "query_id": "4",
        },
        {
            "output_string": "CO.[CH3:1][O:2][c:3]1[cH:4][c:5]([O:6][CH3:7])[c:8]2[c:9](=[O:10])[cH:11][c:12](-[c:13]3[cH:14][cH:15][cH:16][cH:17][c:18]3[Cl:19])[o:20][c:21]2[c:22]1[C:23]1=[CH:24][CH2:25][N:26]([CH3:27])[CH2:28][C:29]1=[O:30].[BH4-].[Na+]>>[CH3:1][O:2][c:3]1[cH:4][c:5]([O:6][CH3:7])[c:8]2[c:9](=[O:10])[cH:11][c:12](-[c:13]3[cH:14][cH:15][cH:16][cH:17][c:18]3[Cl:19])[o:20][c:21]2[c:22]1[C:23]1=[CH:24][CH2:25][N:26]([CH3:27])[CH2:28][CH:29]1[OH:30]",
            "query_id": "5",
        },
    ]
    syngraph = BipartiteSynGraph(reaction_list)
    sparrow_dict = translator("syngraph", syngraph, "sparrow", "bipartite")
    assert "Compound Nodes" in sparrow_dict
    assert "Reaction Nodes" in sparrow_dict

    # check if a warning is logged if the required data model is not bipartite
    test_context = unittest.TestCase()
    with test_context.assertLogs("linchemin.cgu.translate", level="WARNING") as cm:
        sparrow_dict = translator(
            "syngraph", syngraph, "sparrow", "monopartite_reactions"
        )
    test_context.assertEqual(
        cm.output,
        [
            "WARNING:linchemin.cgu.translate:For full compatibility with sparrow software, the graph should be "
            "bipartite"
        ],
    )
    assert sparrow_dict["Compound Nodes"] == []

    assert "sparrow" in get_output_formats()
