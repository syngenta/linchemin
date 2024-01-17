from linchemin.cgu.graph_transformations.supporting_functions import (
    populate_iron,
    build_iron_edge,
)
from linchemin.cgu.iron import Iron, Node, Edge


def test_populate_iron():
    """To test that an Iron instance is correctly populated with nodes and edges"""
    iron = Iron()
    parent = Node(iid="0", properties={}, labels=[])
    iron.add_node(str(parent.iid), parent)
    mol = "new_node"
    iron, new_node_id = populate_iron(parent=parent.iid, mol=mol, iron=iron)
    assert len(iron.nodes) == 2
    assert iron.edges
    assert iron.edges["0"].direction.string == "1>0"


def test_build_iron_edge():
    """To test that the correct Edge instance is created"""
    edge = build_iron_edge(source_node_id="0", target_node_id="1", id_e="0")
    assert isinstance(edge, Edge)
    assert edge.direction.tup == ("0", "1")
