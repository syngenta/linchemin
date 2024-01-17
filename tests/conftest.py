from pathlib import Path

import pytest
import json
from linchemin.cgu.iron import Iron, Node, Edge, Direction
from linchemin.cgu.syngraph import BipartiteSynGraph


@pytest.fixture
def iron_w_smiles():
    nodes = {
        "0": Node(
            iid="0",
            properties={"node_smiles": "CN[C@@H](C)[C@H](O)C1=CC=CC=C1"},
            labels=[],
        ),
        "1": Node(iid="1", properties={"node_smiles": "CN"}, labels=[]),
        "2": Node(
            iid="2", properties={"node_smiles": "C[C@@H]1O[C@@H]1c1ccccc1"}, labels=[]
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
    }
    graph_iron = Iron()

    for id_n, node in nodes.items():
        graph_iron.add_node(id_n, node)

    for id_e, edge in edges.items():
        graph_iron.add_edge(id_e, edge)
    return graph_iron


@pytest.fixture
def bp_syngraph_instance():
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
            "output_string": "CO.[CH3:1][O:2][c:3]1[cH:4][c:5]([O:6][CH3:7])[c:8]2[c:9](=[O:10])[cH:11][c:12](-[c:13]3[cH:14][cH:15][cH:16][cH:17][c:18]3[Cl:19])[o:20][c:21]2[c:22]1[C:23]1=[CH:24][CH2:25][N:26]([CH3:27])[CH2:28][C:29]1=[O:30].[BH4-].[Na+]>>[CH3:1][O:2][c:3]1[cH:4][c:5]([O:6][CH3:7])[c:8]2[c:9](=[O:10])[cH:11][c:12](-[c:13]3[cH:14][cH:15][cH:16][cH:17][c:18]3[Cl:19])[o:20][c:21]2[c:22]1[C:23]1=[CH:24][CH2:25][N:26]([CH3:27])[CH2:28][CH:29]1[OH:30]",
            "query_id": "5",
        },
    ]
    return BipartiteSynGraph(reaction_list)


@pytest.fixture
def ibm1_path():
    conftest_path = Path(__file__)
    data_path = conftest_path.parent / "test_file"
    return data_path.joinpath("ibmrxn_retro_output_raw.json")


@pytest.fixture
def ibm1_as_dict():
    conftest_path = Path(__file__)
    data_path = conftest_path.parent / "test_file"
    data = data_path.joinpath("ibmrxn_retro_output_raw.json")
    return json.loads(open(data).read())


@pytest.fixture
def ibm2_path():
    conftest_path = Path(__file__)
    data_path = conftest_path.parent / "test_file"
    return data_path.joinpath("ibm_output2.json")


@pytest.fixture
def ibm2_as_dict():
    conftest_path = Path(__file__)
    data_path = conftest_path.parent / "test_file"
    data = data_path.joinpath("ibm_output2.json")
    return json.loads(open(data).read())


@pytest.fixture
def az_path():
    conftest_path = Path(__file__)
    data_path = conftest_path.parent / "test_file"
    return data_path.joinpath("az_retro_output_raw.json")


@pytest.fixture
def az_as_dict():
    conftest_path = Path(__file__)
    data_path = conftest_path.parent / "test_file"
    data = data_path.joinpath("az_retro_output_raw.json")
    return json.loads(open(data).read())


@pytest.fixture
def mit_path():
    conftest_path = Path(__file__)
    data_path = conftest_path.parent / "test_file"
    return data_path.joinpath("askos_output.json")


@pytest.fixture
def mit_as_dict():
    conftest_path = Path(__file__)
    data_path = conftest_path.parent / "test_file"
    data = data_path.joinpath("askos_output.json")
    return json.loads(open(data).read())


@pytest.fixture
def trees_path():
    conftest_path = Path(__file__)
    data_path = conftest_path.parent / "test_file"
    return data_path.joinpath("trees.json")


@pytest.fixture
def reaxys_path():
    conftest_path = Path(__file__)
    data_path = conftest_path.parent / "test_file"
    return data_path.joinpath("reaxys_output.json")


@pytest.fixture
def reaxys_as_dict():
    conftest_path = Path(__file__)
    data_path = conftest_path.parent / "test_file"
    data = data_path.joinpath("reaxys_output.json")
    return json.loads(open(data).read())


@pytest.fixture
def cli():
    conftest_path = Path(__file__)
    data_path = conftest_path.parent.parent
    return data_path.joinpath("src/linchemin/interfaces/cli.py")
