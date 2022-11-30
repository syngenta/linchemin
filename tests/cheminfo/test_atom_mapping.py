from linchemin.cheminfo.atom_mapping import pipeline_atom_mapping, perform_atom_mapping, get_available_mappers
from linchemin.cgu.syngraph import extract_reactions_from_syngraph, MonopartiteReacSynGraph, merge_syngraph
from linchemin.cgu.translate import translator
import pytest
import json


def test_basic_factory(capfd):
    with pytest.raises(KeyError) as ke:
        perform_atom_mapping('unavailable_mapper', [])
    assert "UnavailableMapper" in str(ke.type)


def test_helper_function():
    h = get_available_mappers()
    assert h
    assert 'namerxn' in h


def test_rxnmapper(az_path):
    graph_az = json.loads(open(az_path).read())
    syngraph = translator('az_retro', graph_az[0], 'syngraph', 'monopartite_reactions')
    reaction_list = extract_reactions_from_syngraph(syngraph)
    out = perform_atom_mapping('rxnmapper', reaction_list)
    assert out is not None
    assert out.pipeline_success_rate == {}
    s = MonopartiteReacSynGraph(out.mapped_reactions)
    assert len(syngraph.graph) == len(s.graph)
    for parent, children in s.graph.items():
        assert parent.disconnection is not None
        assert parent.template is not None


def test_namerxn_service(ibm1_path):
    graph = json.loads(open(ibm1_path).read())
    syngraph = translator('ibm_retro', graph[2], 'syngraph', 'monopartite_reactions')
    reaction_list = extract_reactions_from_syngraph(syngraph)
    out = perform_atom_mapping('namerxn', reaction_list)
    assert out is not None
    s = MonopartiteReacSynGraph(out.mapped_reactions)
    assert len(syngraph.graph) == len(s.graph)
    disc1 = set()
    for parent, children in s.graph.items():
        assert parent.disconnection is not None
        assert parent.template is not None
    # With test file ibm2_path, even if the reactions are all mapped, some disconnections remain None;
    # the code below is used to debug
    #     print('***')
    #     print(parent.smiles)
    #     print(parent.disconnection)
    #     print(parent.template)
    #     disc1.add(parent.template)
    # out2 = perform_atom_mapping('rxnmapper', reaction_list)
    # s2 = MonopartiteReacSynGraph(out2.mapped_reactions)
    # disc2 = set()
    # for parent, children in s2.graph.items():
    #     print('***')
    #     print(parent.smiles)
    #     print(parent.disconnection)
    #     print(parent.template)
    #     disc2.add(parent.template)


def test_pipeline(az_path):
    graph = json.loads(open(az_path).read())
    syngraph = translator('az_retro', graph[2], 'syngraph', 'monopartite_reactions')
    reaction_list = extract_reactions_from_syngraph(syngraph)
    out = pipeline_atom_mapping(reaction_list)
    assert out is not None
    assert out.pipeline_success_rate != {}
    s = MonopartiteReacSynGraph(out.mapped_reactions)
    assert len(syngraph.graph) == len(s.graph)

