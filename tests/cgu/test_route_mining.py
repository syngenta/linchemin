import json
import unittest

import pytest

from linchemin.cgu.route_mining import route_miner, RouteFinder
from linchemin.cgu.syngraph import MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph
from linchemin.cgu.translate import nx


def test_route_extractor():
    nodes = [{'query_id': 0, 'output_string': 'CCN.CCOC(=O)CC>>CCNC(=O)CC'}]
    original_routes = [MonopartiteReacSynGraph(nodes)]

    new_routes = route_miner(original_routes, ['CCO.CCC(O)=O>>CCOC(=O)CC'])
    assert len(new_routes) == 1
    assert len(new_routes[0].get_molecule_roots())
    assert original_routes[0] not in new_routes
    assert len(new_routes[0].graph) > len(original_routes[0].graph)

    new_routes2 = route_miner(new_routes, ['CCO.CCC(Cl)=O>>CCOC(=O)CC'])
    assert len(new_routes2) == 1
    assert all(r not in new_routes2 for r in new_routes)

    # If a new nodes are not connected to any of the pre-existing routes, a warning is raised and None is returned
    with unittest.TestCase().assertLogs('linchemin.cgu.route_mining', level='WARNING'):
        new_routes3 = route_miner(new_routes2, ['CCO.OC(=O)C1=NC=CC(Cl)=C1>>CCOC(=O)C1=NC=CC(Cl)=C1'])
    assert new_routes3 is None

    with pytest.raises(TypeError) as te:
        original_routes = [MonopartiteMolSynGraph(nodes)]
        route_miner(original_routes, ['CCO.CCC(O)=O>>CCOC(=O)CC'])
    assert "TypeError" in str(te.type)


def test_route_extractor_bipartite():
    nodes = [{'query_id': 0, 'output_string': 'CCN.CCOC(=O)CC>>CCNC(=O)CC'}]
    original_routes = [BipartiteSynGraph(nodes)]

    new_routes = route_miner(original_routes, ['CCO.CCC(O)=O>>CCOC(=O)CC'])
    assert len(new_routes) == 1
    assert len(new_routes[0].get_molecule_roots())
    assert original_routes[0] not in new_routes


def build_graph_from_tree_data(tree_data):
    node_data = tree_data.get("nodes")
    edge_data = tree_data.get("edges")
    # Create a new graph object
    G = nx.DiGraph()
    G.add_nodes_from([(n.get('uid'), n) for n in node_data])
    G.add_edges_from([(e.get('nodes')[0], e.get('nodes')[1], e) for e in edge_data])
    return G


def get_route_uids(route_graph):
    node_uids = [route_graph.nodes[n]["uid"] for n in route_graph.nodes()]
    edge_uids = [(route_graph.edges[e]["nodes"][0], route_graph.edges[e]["nodes"][1]) for e in route_graph.edges()]
    return {"nodes": sorted(node_uids), "edges": sorted(edge_uids)}


def compare_routes(expected_route, route_graph):
    extracted_route = get_route_uids(route_graph)
    # print(extracted_route["nodes"], expected_route["nodes"])
    return sorted(expected_route["nodes"]) == sorted(
        extracted_route["nodes"])  # and expected_route["edges"] == extracted_route["edges"]


def test_RouteFinder(trees_path):
    with open(trees_path) as f:
        json_content = json.load(f)

    root_node_uid = "1"
    for case_idx, data in json_content.items():
        tree_data = data.get('tree')
        routes_data = data.get('routes')
        G = build_graph_from_tree_data(tree_data)
        routes = RouteFinder(G, root_node_uid).find_routes()
        for idx, route in enumerate(routes, start=1):
            matches = [compare_routes(expected_route, route) for expected_route in routes_data]
            assert any(matches)
