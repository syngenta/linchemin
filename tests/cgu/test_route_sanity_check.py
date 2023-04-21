from linchemin.cgu.route_sanity_check import route_checker
import pytest
from linchemin.cgu.syngraph import MonopartiteReacSynGraph, BipartiteSynGraph
from linchemin.cgu.translate import translator
from linchemin.cheminfo.constructors import ChemicalEquationConstructor
import pprint


def test_checker_factory():
    # the factory is correctly called with a MonopartiteReacSynGraph
    route = MonopartiteReacSynGraph()
    checked_route = route_checker(route, 'cycle_check')
    assert checked_route

    # and with a BipartiteSynGraph
    route = BipartiteSynGraph()
    checked_route = route_checker(route, 'cycle_check')
    assert checked_route

    # a type error is raised if the wrong input type is given
    with pytest.raises(TypeError) as e:
        route = {}
        route_checker(route, 'cycle_check')
    assert "TypeError" in str(e.type)

    # a key error is raised if an unavailable checker is given
    with pytest.raises(KeyError) as e:
        route = MonopartiteReacSynGraph()
        route_checker(route, 'not_a_check')
    assert 'KeyError' in str(e.type)


def test_cycle_checker():
    route_cycle = [
        {'query_id': 0,
         'output_string': '[CH3:3][C:2]#[N:1].[OH2:4]>>[CH3:3][C:2]([OH:4])=[O:4]'},
        {'query_id': 1,
         'output_string': 'O[C:2]([CH3:1])=[O:3].[CH3:4][NH2:5]>>[CH3:1][C:2](=[O:3])[NH:5][CH3:4]'},
        {'query_id': 2,
         'output_string': '[CH3:5][NH:4][C:2]([CH3:1])=[O:3].[OH2:6]>>[CH3:1][C:2]([OH:6])=[O:3]'},
        {'query_id': 3,
         'output_string': 'ClP(Cl)[Cl:4].O[C:2]([CH3:1])=[O:3]>>[Cl:4][C:2]([CH3:1])=[O:3]'},
    ]
    route = MonopartiteReacSynGraph(route_cycle)
    checked_route = route_checker(route, 'cycle_check')
    assert len(checked_route.graph) == 2
    assert len(checked_route.graph) != len(route.graph)
    assert checked_route.uid != route.uid
    ce = ChemicalEquationConstructor().build_from_reaction_string(
        '[CH3:5][NH:4][C:2]([CH3:1])=[O:3].[OH2:6]>>[CH3:1][C:2]([OH:6])=[O:3]',
        'smiles')
    assert ce in route.graph
    assert ce not in checked_route.graph


def test_isolated_nodes():
    reactions = [
        {'output_string': 'Cl[C:2]([CH3:1])=[O:3].[CH3:4][OH:5]>>[CH3:1][C:2](=[O:3])[O:5][CH3:4]',
         'query_id': '0'},
        {'output_string': '[CH3:5][O:4][C:3]([CH3:2])=[O:1]>>[CH3:2][C:3]([OH:4])=[O:1]',
         'query_id': '1'},
        {'output_string': '[CH3:4][C:5](Cl)=[O:6].CC(O)=O.[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][O:3][C:5]([CH3:4])=[O:6]',
         'query_id': '2'},
        {'output_string': 'O=[C:2](OC[CH3:4])[CH3:1].[Li][CH3:3]>>[CH2:1]=[C:2]([CH3:3])[CH3:4]',
         'query_id': '3'}
    ]
    route = MonopartiteReacSynGraph(reactions)
    checked_route = route_checker(route, 'isolated_nodes_check')
    assert len(checked_route.graph) == 2
    assert len(checked_route.graph) != len(route.graph)
    assert checked_route.uid != route.uid
    ce = ChemicalEquationConstructor().build_from_reaction_string(
        '[CH3:5][O:4][C:3]([CH3:2])=[O:1]>>[CH3:2][C:3]([OH:4])=[O:1]',
        'smiles')
    assert ce in route.graph
    assert ce not in checked_route.graph