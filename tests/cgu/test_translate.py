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
    ReaxysRT,
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


def test_reaxys_to_iron():
    route = {
        "@id": "rxsp:id-acd758fc-9499-4f91-a1a8-a1a3e9852cfe#id-a4fdb8c8-8591-4305-aafe-2552cdb041cf",
        "@type": "rxspm:Route",
        "rxspm:hasStep": [
            {
                "@id": "rxsp:id-acd758fc-9499-4f91-a1a8-a1a3e9852cfe#id-fed1acaf-5a37-41e9-956a-27501ab66b31",
                "@type": "rxspm:Step",
                "rxspm:hasReaction": {
                    "@id": "rxsp:reaction/id-43bf5439-4224-42b9-ae47-7d4a9ada0e16",
                    "@type": "rxspm:Reaction",
                    "rxspm:hasProduct": [
                        {
                            "@id": "rxsp:product/id-237ba3da-54ce-42db-bfb3-dea653580cd2",
                            "@type": "rxspm:Product",
                            "rxspm:hasSubstance": {
                                "@id": "rxsp:substance/id-4cf2b0e5-beb4-420a-8529-1d75ec5df351",
                                "@type": "edm:Substance",
                                "edm:smiles": "C[C@H](N=Cc1ccccc1)[C@H](O)c1ccccc1",
                            },
                        }
                    ],
                    "rxspm:hasStartingMaterial": [
                        {
                            "@id": "rxsp:reactant/id-9ea10464-2b1a-4c5d-a9f2-655cd7d3541c",
                            "@type": "rxspm:StartingMaterial",
                            "rxspm:hasSubstance": {
                                "@id": "rxsp:substance/id-6a394a6c-e5c1-495c-9d41-96bb8089d07a",
                                "@type": "edm:Substance",
                                "edm:smiles": "C[C@H](N)[C@H](O)c1ccccc1",
                            },
                        },
                        {
                            "@id": "rxsp:reactant/id-227ba5e1-5298-4732-a203-e2e7ca62a836",
                            "@type": "rxspm:StartingMaterial",
                            "rxspm:hasSubstance": {
                                "@id": "rxsp:substance/id-65de8903-1700-4bf2-b106-c2ed490410e5",
                                "@type": "edm:Substance",
                                "edm:smiles": "O=Cc1ccccc1",
                            },
                        },
                    ],
                    "rxspm:hasExample": [
                        {
                            "@id": "rxsp:example/id-fa5603dc-6153-4262-ba29-d99226e7b998",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "2680352",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "1.0",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-febb1787-d7e8-4cd2-8dc2-01ae869e0458",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "2681831",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "1.0",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-cad52a57-bb6c-458a-aec5-63f323c188d1",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "29079193",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "1.0",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-263dff1e-9a8c-4204-89f9-c03f91df28fa",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "2682182",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "1.0",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-83175178-ea60-44a7-821b-023d48483765",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "2680354",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "1.0",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-a5e85803-7ce1-41b8-b574-04d337ec146a",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "8760921",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "1.0",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-2f9e38a8-6c9d-498e-935f-53cdcd4aa203",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "8769043",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "1.0",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-9bc3d086-eabc-42d6-bd6c-3b759ba0b289",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "8769038",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "1.0",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-8df21fb8-7379-4687-bbc4-f8f9d755cd97",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "8770951",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "1.0",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-256f4e43-2753-4744-9ed1-e5bdf512a288",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "25925669",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "1.0",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-eb5430c5-9546-485f-a272-2cfe8cbdebb2",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "1203418",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "1.0",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-aec0d332-572e-48ff-8352-f9bca34ca11d",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "11275549",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "1.0",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-1a326275-02d8-43cc-a09d-22a47e7c10c7",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "8759312",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "1.0",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-11c82e07-0d71-4e35-bf49-dadb6aa94974",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "9465224",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.8095238",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-c509bb93-ec70-4794-84c2-87cdf0a18da5",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "25925663",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.8095238",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-13bcdf98-ffdd-47ba-845d-6b0953bd4239",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "25925667",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.8095238",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-798da0e0-0cae-4b9d-a9d5-879ef920687c",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "25925659",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.8095238",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-2fa2cff8-aac2-46d1-a037-4db235b8b7c0",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "25925665",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.8095238",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-411451ee-8d79-4a30-825c-289f38439be5",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "2950102",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.77272725",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-c62eb01f-5ab7-4ef1-b27a-923f34ebb0f5",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "30853783",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.65217394",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-08607e94-a88d-456e-adab-005fa26c5234",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "1527194",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.65217394",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-eec21e42-fd85-4040-8c41-133225338019",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "29575434",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.31132075",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-38403c81-24dd-4e76-a0a1-ced516ab7776",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "40643236",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.21929824",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-2d225721-f5a3-4b8f-bd03-71b8ccdccce6",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "29575437",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.13864307",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-4d99e02b-a04d-4c90-a58c-f9192e95162b",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "29575444",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.046499476",
                            },
                        },
                    ],
                },
                "rxspm:number": {
                    "@type": "http://www.w3.org/2001/XMLSchema#int",
                    "@value": "1",
                },
                "rxspm:score": {
                    "@type": "http://www.w3.org/2001/XMLSchema#double",
                    "@value": "1.0",
                },
            },
            {
                "@id": "rxsp:id-acd758fc-9499-4f91-a1a8-a1a3e9852cfe#id-d2e53939-533e-4902-92cc-21ecd972e602",
                "@type": "rxspm:Step",
                "rxspm:hasReaction": {
                    "@id": "rxsp:reaction/id-aedc86c0-5a0a-4a01-9c7d-7c5574894a98",
                    "@type": "rxspm:Reaction",
                    "rxspm:hasProduct": [
                        {
                            "@id": "rxsp:product/id-5e159165-2f17-4f84-a1f8-9a3e6f80d5da",
                            "@type": "rxspm:Product",
                            "rxspm:hasSubstance": {
                                "@id": "rxsp:substance/id-5c89a37b-bc1a-4e89-a584-fec2240cbbd7",
                                "@type": "edm:Substance",
                                "edm:smiles": "C[C@H](NCc1ccccc1)[C@H](O)c1ccccc1",
                            },
                        }
                    ],
                    "rxspm:hasStartingMaterial": [
                        {
                            "@id": "rxsp:reactant/id-1ae0a176-10a9-4ae7-a3e5-9f4a622bafcf",
                            "@type": "rxspm:StartingMaterial",
                            "rxspm:hasSubstance": {
                                "@id": "rxsp:substance/id-297b5689-5366-4c44-b768-a795418e97a2",
                                "@type": "edm:Substance",
                                "edm:smiles": "C[C@H](N=Cc1ccccc1)[C@H](O)c1ccccc1",
                            },
                        }
                    ],
                    "rxspm:hasExample": [
                        {
                            "@id": "rxsp:example/id-d3475bca-6949-444d-b942-572a93649bdc",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "29079183",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "1.0",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-9ceedff9-2780-4874-b97c-6907b5518f30",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "9208996",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "1.0",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-20da4dcd-53ab-458e-a1f2-a65ffcebbd5d",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "8778516",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "1.0",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-1570a77d-6c84-403c-a60e-cc437f0be061",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "3044702",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.7647059",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-b00029f0-9dcd-4f9a-8905-a75fc945fad8",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "29874820",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.7647059",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-b33e1c0a-91f5-4839-afae-5fb2dd1a6847",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "28492313",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.7647059",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-62eef74c-8872-4370-966d-dc4a039a795c",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "3044701",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.7647059",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-22b0b451-6f1e-4930-8485-4e7fd0e97df7",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "25925632",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.75",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-c3d1e026-3c88-4986-b165-b48584349634",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "25925636",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.57894737",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-b019cc43-b5c3-473a-8c02-0398f9347284",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "11275559",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.57894737",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-4f862510-3179-4dd6-a9e2-c733abe79461",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "1163480",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.57894737",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-25201302-af05-462b-acf6-58cf35e68dd2",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "8780283",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.57894737",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-fcfd6588-c082-4b79-a763-69aa89f9f163",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "8780092",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.57894737",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-cf6b4040-c6b7-480c-a02a-83aaaefbc649",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "8781474",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.57894737",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-5bbdcafc-d2c8-4aa0-8965-0399cfda4a6f",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "3039223",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.57894737",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-169f93a2-53bf-4807-860a-f7a3dc74af75",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "30689165",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.57894737",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-84afe4be-5261-4c05-aaba-c75d2bbcd12a",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "3578120",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.5555556",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-7a7b648d-6d07-44b0-acfe-a222cf92dfed",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "25925629",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.5555556",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-b3229ef2-424b-495e-bb48-e0bcb7bbc0dd",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "25925633",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.5555556",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-bce98d88-8519-4ec3-b128-38e48a587423",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "33358169",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.55",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-b2401d84-147d-4ac1-9145-d8f65391882c",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "3076031",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.55",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-3c9cd04c-09f9-45a5-b92c-77344ee15437",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "5071516",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.55",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-2e201a15-0e84-4f95-9031-64edcceafd08",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "25925641",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.5",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-66b4a540-2788-4463-815a-f452d35be286",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "9489183",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.5",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-9ef48984-f111-4a27-851a-6484facb09fe",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "10171255",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.3859649",
                            },
                        },
                    ],
                },
                "rxspm:number": {
                    "@type": "http://www.w3.org/2001/XMLSchema#int",
                    "@value": "2",
                },
                "rxspm:score": {
                    "@type": "http://www.w3.org/2001/XMLSchema#double",
                    "@value": "0.9999",
                },
            },
            {
                "@id": "rxsp:id-acd758fc-9499-4f91-a1a8-a1a3e9852cfe#id-227086ad-96c5-4b4f-9f58-583be8a2c0e3",
                "@type": "rxspm:Step",
                "rxspm:hasReaction": {
                    "@id": "rxsp:reaction/id-8f3a20b8-ccf4-48cb-a342-f3a23409ac1a",
                    "@type": "rxspm:Reaction",
                    "rxspm:hasProduct": [
                        {
                            "@id": "rxsp:product/id-f3191531-b007-469f-b7d5-64121859c6ce",
                            "@type": "rxspm:Product",
                            "rxspm:hasSubstance": {
                                "@id": "rxsp:substance/id-9ab93ca4-4975-4a40-af46-cc87c9cc4111",
                                "@type": "edm:Substance",
                                "edm:smiles": "C[C@@H]([C@H](O)c1ccccc1)N(C)Cc1ccccc1",
                            },
                        }
                    ],
                    "rxspm:hasStartingMaterial": [
                        {
                            "@id": "rxsp:reactant/id-960fb89a-5213-4db6-a2fe-50474f1110ad",
                            "@type": "rxspm:StartingMaterial",
                            "rxspm:hasSubstance": {
                                "@id": "rxsp:substance/id-58a34bc2-8aff-49e4-b10b-8c58736cd84d",
                                "@type": "edm:Substance",
                                "edm:smiles": "C=O",
                            },
                        },
                        {
                            "@id": "rxsp:reactant/id-54260561-f467-4a8d-8834-e692024df8aa",
                            "@type": "rxspm:StartingMaterial",
                            "rxspm:hasSubstance": {
                                "@id": "rxsp:substance/id-12376f5e-2652-47c2-aa5f-244403ed445d",
                                "@type": "edm:Substance",
                                "edm:smiles": "C[C@H](NCc1ccccc1)[C@H](O)c1ccccc1",
                            },
                        },
                    ],
                    "rxspm:hasExample": [
                        {
                            "@id": "rxsp:example/id-440eff2c-f06f-40f0-ba66-8879b512e476",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "36097645",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.84",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-8de1db2a-ab62-41fa-b2ba-4521c9812d82",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "4662100",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.76",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-6fe76939-254c-4415-962c-76b121d1c3ab",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "95554",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.6666667",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-94ddd7b5-191b-46ab-a0a0-2b1cae53c622",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "4463564",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.48387095",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-d4fc5670-4c4a-4ba6-aaff-52a725c900d3",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "933395",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.48387095",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-19ef730b-f235-4351-a78d-821dc62270fe",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "934001",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.48387095",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-701c1b9c-978b-4919-8d39-1c5f29487a63",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "934281",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.48387095",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-c3b3a9fa-488b-4970-b698-8156f8767477",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "10189172",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.45454547",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-90d31c24-785c-4847-b44c-6e19fdde7e5b",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "42338762",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.45454547",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-2f36286e-95e6-48ef-ae44-7602a1d54adb",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "41277764",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.45454547",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-a7890b00-9434-4732-9d06-1efb7f69a0a3",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "29033546",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.45454547",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-dc5ab744-5b4a-4a8c-924e-cc4ca5d2cfed",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "933534",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.45454547",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-3f439768-53fe-49cb-90e4-1decfe62f824",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "934069",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.45454547",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-dd5dafb3-1fc6-408d-acd5-85e4a3816f8a",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "934332",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.45454547",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-4a8d5ae3-53b2-4c4e-89d4-7ca8ecef7548",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "23664717",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.45454547",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-c3d6ff56-d149-401e-93a9-707dd69a00fc",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "934068",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.45454547",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-1f4e9fc5-c8c5-4286-8aca-199cd02f0fc4",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "934283",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.45454547",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-cf7a5bd0-f1bd-4b56-9570-1c0058e339fa",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "933293",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.45454547",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-896ac4cb-8bef-43ce-808c-073e9983d874",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "933905",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.45454547",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-fb227bb5-e826-4e6d-b1b1-042ca1d68956",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "933915",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.45454547",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-03495a6e-d38e-4f05-9bc1-02f20c7b5d7b",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "934331",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.45454547",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-c3591378-5cc0-4b39-a0a6-743a889cfe57",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "934765",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.45454547",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-ee3b71a0-0930-471c-a822-0bb019a4ea12",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "934836",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.45454547",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-28643df5-a631-4619-a2d5-6376989fc264",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "933763",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.45454547",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-07681315-9808-4339-8e6e-74084370850b",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "755557",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.4",
                            },
                        },
                    ],
                },
                "rxspm:number": {
                    "@type": "http://www.w3.org/2001/XMLSchema#int",
                    "@value": "3",
                },
                "rxspm:score": {
                    "@type": "http://www.w3.org/2001/XMLSchema#double",
                    "@value": "0.9968",
                },
            },
            {
                "@id": "rxsp:id-acd758fc-9499-4f91-a1a8-a1a3e9852cfe#id-c23caae2-734d-4679-845d-a7eecaf9b20a",
                "@type": "rxspm:Step",
                "rxspm:hasReaction": {
                    "@id": "rxsp:reaction/id-fcc1e8ca-ba09-45a3-bfac-09357733d66c",
                    "@type": "rxspm:Reaction",
                    "rxspm:hasProduct": [
                        {
                            "@id": "rxsp:product/id-7f4dfb8c-bdcb-4494-8935-92b5737ef660",
                            "@type": "rxspm:Product",
                            "rxspm:hasSubstance": {
                                "@id": "rxsp:substance/id-f13c5458-93ec-4633-a12f-3619c782f21b",
                                "@type": "edm:Substance",
                                "edm:smiles": "CN[C@@H](C)[C@H](O)C1=CC=CC=C1",
                            },
                        }
                    ],
                    "rxspm:hasStartingMaterial": [
                        {
                            "@id": "rxsp:reactant/id-383475e5-abfd-4e51-b87d-fcb9df8937bb",
                            "@type": "rxspm:StartingMaterial",
                            "rxspm:hasSubstance": {
                                "@id": "rxsp:substance/id-35214d2a-c86a-4679-bea4-4434c8c1dac9",
                                "@type": "edm:Substance",
                                "edm:smiles": "C[C@@H]([C@H](O)c1ccccc1)N(C)Cc1ccccc1",
                            },
                        }
                    ],
                    "rxspm:hasExample": [
                        {
                            "@id": "rxsp:example/id-90172e74-6fcf-466c-82e7-be985b900fb9",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "4619567",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "1.0",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-b1b07b5f-1fb7-403e-8ba8-de6475082430",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "2937932",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.9852941",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-5bd7cdfe-122a-46b9-b43a-1aafca8d3179",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "23674887",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.875",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-5791ce17-15fb-4b50-a02b-819152aa0b07",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "1141755",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.7241379",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-74acda7c-e23d-47f3-96f7-af9f91cbd3aa",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "508723",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.7241379",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-82546ae9-0773-4c5c-a8df-fbcebca29dc4",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "1135400",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.7241379",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-38932392-73d8-4f44-aa87-9137cc4c1329",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "1135822",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.7241379",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-b9f809d9-1953-4374-b28f-18ce0e5e90a7",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "541794",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.71590906",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-f1e11c69-c717-40f7-bf3f-05247eea3eaf",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "4409838",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.71590906",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-499bd0d7-5546-4318-ba52-4b3daa68549a",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "3097624",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.71590906",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-127fcdc8-081e-48bb-a767-f1874467c9de",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "495933",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.70454544",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-02476321-7928-48ac-b731-29ecbe757e9b",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "353589",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.6923077",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-1c070132-fb83-4143-8810-5bec74a01b1f",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "4396486",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.6923077",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-abf815f1-a31d-43ed-b55b-651271add114",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "4396487",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.6923077",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-1c099a63-2353-4f1a-ae5a-df23998760b3",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "7912717",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.6847826",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-88534bbf-28ae-4994-a74e-06b8f9877600",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "7912716",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.6847826",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-a6de5284-a3de-4c3a-a476-ff0ded8540cc",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "7912721",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.6847826",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-1308d4f3-fad1-4bec-b668-9082ae6cdb57",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "2208852",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.6847826",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-786a5134-0702-46f6-ba56-a065ab61ce79",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "7912718",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.6666667",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-0b4de68f-0c54-4f26-907d-05d30594237e",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "7912722",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.6666667",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-689356dc-d179-4be9-84f5-630be6d19b1c",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "8097973",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.6666667",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-74dd4e75-ff96-4991-993e-6504da6a4656",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "7994011",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.6666667",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-ca746b3d-a7c4-4926-832a-539b97455fcb",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "3724798",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.6333333",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-528dffc0-f423-48d8-9f69-8f9d6e890be3",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "4399579",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.5470086",
                            },
                        },
                        {
                            "@id": "rxsp:example/id-30647200-471a-4a86-889a-2065b4c0522c",
                            "@type": "rxspm:Example",
                            "edm:reaxysRN": "513099",
                            "edm:similarity": {
                                "@type": "http://www.w3.org/2001/XMLSchema#double",
                                "@value": "0.53051645",
                            },
                        },
                    ],
                },
                "rxspm:number": {
                    "@type": "http://www.w3.org/2001/XMLSchema#int",
                    "@value": "4",
                },
                "rxspm:score": {
                    "@type": "http://www.w3.org/2001/XMLSchema#double",
                    "@value": "0.9998",
                },
            },
        ],
        "rxspm:score": {
            "@type": "http://www.w3.org/2001/XMLSchema#double",
            "@value": "0.94",
        },
        "rxspm:numberOfSteps": {
            "@type": "http://www.w3.org/2001/XMLSchema#int",
            "@value": "4",
        },
        "rxspm:longestPathLength": {
            "@type": "http://www.w3.org/2001/XMLSchema#int",
            "@value": "4",
        },
        "rxspm:numberOfStartingMaterials": {
            "@type": "http://www.w3.org/2001/XMLSchema#int",
            "@value": "3",
        },
    }
    iron = ReaxysRT().to_iron(route)
    assert len(iron.edges) == 6
    assert len(iron.nodes) == 7
    # testing with factory machinery
    syngraph = translator("reaxys", route, "syngraph", "monopartite_reactions")
    assert len(syngraph.graph) == 4
