import unittest

import pytest

from linchemin.cgu.route_mining import route_miner
from linchemin.cgu.syngraph import MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph


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
