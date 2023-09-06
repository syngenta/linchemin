from linchemin.cgu.iron import Direction, Edge, Iron, Node


def test_direction():
    """To test that the instance of Direction class is correctly initiated."""
    direc = Direction("a>b")
    assert direc.string == "a>b"
    assert direc.tup == ("a", "b")


def test_count_nodes():
    """To test that the Iron method 'i_node_number' correctly returned the number of nodes in an Iron instance."""
    iron = Iron()
    assert iron.i_node_number() == 0

    node = Node(iid="2", properties={"id": "101"}, labels=[])
    iron.add_node("1", node)
    assert iron.i_node_number() == 1


def test_count_edges():
    """To test that the Iron method 'i_edge_number' correctly returned the number of edges in an Iron instance."""
    iron = Iron()
    node1 = Node(iid="1", properties={"id": "1"}, labels=[])
    node2 = Node(iid="2", properties={"id": "101"}, labels=[])
    edge = Edge(
        iid="1",
        a_iid="2",
        b_iid="1",
        direction=Direction("2>1"),
        properties={},
        labels=[],
    )
    iron.add_node("1", node1)
    iron.add_node("2", node2)
    iron.add_edge("1", edge)
    assert iron.i_edge_number() == 1


def test_iron_neighbors():
    """To test that the Iron method 'get_neighbors' correctly returns the neighbors of a specified node
    in an Iron instance."""
    iron = Iron()
    node1 = Node(iid="1", properties={"id": "1"}, labels=[])
    node2 = Node(iid="2", properties={"id": "101"}, labels=[])
    edge = Edge(
        iid="1",
        a_iid="2",
        b_iid="1",
        direction=Direction("2>1"),
        properties={},
        labels=[],
    )
    iron.add_node("1", node1)
    assert len(iron.get_neighbors("1")) == 0

    iron.add_node("2", node2)
    iron.add_edge("1", edge)
    assert len(iron.get_neighbors("1")) == 1


def test_iron_child_node():
    """To test that the Iron method 'get_child_nodes' correctly returns the 'children' of a specified node
    in an Iron instance."""
    iron = Iron()
    node1 = Node(iid="1", properties={"id": "1"}, labels=[])
    node2 = Node(iid="2", properties={"id": "101"}, labels=[])
    edge = Edge(
        iid="1",
        a_iid="2",
        b_iid="1",
        direction=Direction("2>1"),
        properties={},
        labels=[],
    )
    iron.add_node("1", node1)
    iron.add_node("2", node2)
    iron.add_edge("1", edge)
    assert len(iron.get_child_nodes("1")) == 1
    assert len(iron.get_child_nodes("2")) == 0


def test_iron_parent_node():
    """To test that the Iron method 'get_parent_nodes' correctly returns the 'parents' of a specified node
    in an Iron instance."""
    iron = Iron()
    node1 = Node(iid="1", properties={"id": "1"}, labels=[])
    node2 = Node(iid="2", properties={"id": "101"}, labels=[])
    edge = Edge(
        iid="1",
        a_iid="2",
        b_iid="1",
        direction=Direction("2>1"),
        properties={},
        labels=[],
    )
    iron.add_node("1", node1)
    iron.add_node("2", node2)
    iron.add_edge("1", edge)
    assert len(iron.get_parent_nodes("1")) == 0
    assert len(iron.get_parent_nodes("2")) == 1


def test_get_edge_id():
    """To test that the Iron method 'get_edge_id' correctly returns the edges linking a pair of specified nodes
    in an Iron instance."""
    iron = Iron()
    node1 = Node(iid="1", properties={"id": "1"}, labels=[])
    node2 = Node(iid="2", properties={"id": "101"}, labels=[])
    edge = Edge(
        iid="1",
        a_iid="2",
        b_iid="1",
        direction=Direction("2>1"),
        properties={},
        labels=[],
    )
    iron.add_node("1", node1)
    iron.add_node("2", node2)
    iron.add_edge("1", edge)
    assert iron.get_edge_id("1", "2") == ["1"]


def test_degree_sequence():
    """To test that the Iron method 'get_degree_sequence' correctly returns the degree sequence of an Iron instance."""
    nodes = {
        1: Node(iid="1", properties={"id": "1"}, labels=[]),
        2: Node(iid="2", properties={"id": "101"}, labels=[]),
        3: Node(iid="3", properties={"id": "102"}, labels=[]),
        4: Node(iid="4", properties={"id": "103"}, labels=[]),
        5: Node(iid="5", properties={"id": "2"}, labels=[]),
        6: Node(iid="6", properties={"id": "3"}, labels=[]),
        7: Node(iid="7", properties={"id": "4"}, labels=[]),
        8: Node(iid="8", properties={"id": "104"}, labels=[]),
        9: Node(iid="9", properties={"id": "5"}, labels=[]),
        10: Node(iid="10", properties={"id": "105"}, labels=[]),
        11: Node(iid="11", properties={"id": "106"}, labels=[]),
        12: Node(iid="12", properties={"id": "8"}, labels=[]),
        13: Node(iid="13", properties={"id": "6"}, labels=[]),
        14: Node(iid="14", properties={"id": "7"}, labels=[]),
        15: Node(iid="15", properties={"id": "9"}, labels=[]),
    }
    edges = {
        1: Edge(
            iid="1",
            a_iid="2",
            b_iid="1",
            direction=Direction("2>1"),
            properties={},
            labels=[],
        ),
        2: Edge(
            iid="2",
            a_iid="3",
            b_iid="1",
            direction=Direction("3>1"),
            properties={},
            labels=[],
        ),
        3: Edge(
            iid="3",
            a_iid="4",
            b_iid="1",
            direction=Direction("4>1"),
            properties={},
            labels=[],
        ),
        4: Edge(
            iid="4",
            a_iid="5",
            b_iid="2",
            direction=Direction("5>2"),
            properties={},
            labels=[],
        ),
        5: Edge(
            iid="5",
            a_iid="6",
            b_iid="3",
            direction=Direction("6>3"),
            properties={},
            labels=[],
        ),
        6: Edge(
            iid="6",
            a_iid="7",
            b_iid="4",
            direction=Direction("7>4"),
            properties={},
            labels=[],
        ),
        7: Edge(
            iid="7",
            a_iid="5",
            b_iid="3",
            direction=Direction("5>3"),
            properties={},
            labels=[],
        ),
        8: Edge(
            iid="8",
            a_iid="5",
            b_iid="4",
            direction=Direction("5>4"),
            properties={},
            labels=[],
        ),
        9: Edge(
            iid="9",
            a_iid="8",
            b_iid="5",
            direction=Direction("8>5"),
            properties={},
            labels=[],
        ),
        10: Edge(
            iid="10",
            a_iid="11",
            b_iid="7",
            direction=Direction("11>6"),
            properties={},
            labels=[],
        ),
        11: Edge(
            iid="11",
            a_iid="10",
            b_iid="7",
            direction=Direction("10>7"),
            properties={},
            labels=[],
        ),
        12: Edge(
            iid="12",
            a_iid="9",
            b_iid="8",
            direction=Direction("9>8"),
            properties={},
            labels=[],
        ),
        13: Edge(
            iid="13",
            a_iid="12",
            b_iid="11",
            direction=Direction("12>11"),
            properties={},
            labels=[],
        ),
        14: Edge(
            iid="14",
            a_iid="13",
            b_iid="10",
            direction=Direction("13>10"),
            properties={},
            labels=[],
        ),
        15: Edge(
            iid="15",
            a_iid="14",
            b_iid="10",
            direction=Direction("14>10"),
            properties={},
            labels=[],
        ),
        16: Edge(
            iid="16",
            a_iid="10",
            b_iid="15",
            direction=Direction("10>15"),
            properties={},
            labels=[],
        ),
        17: Edge(
            iid="17",
            a_iid="6",
            b_iid="4",
            direction=Direction("6>4"),
            properties={},
            labels=[],
        ),
    }

    iron = Iron()

    for id_n, node in nodes.items():
        iron.add_node(id_n, node)

    for id_e, edge in edges.items():
        iron.add_edge(id_e, edge)

    assert iron.get_degree_sequence() == [4, 4, 4, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1]
