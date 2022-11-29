from linchemin.cheminfo.atom_mapping import pipeline_atom_mapping, perform_atom_mapping, get_available_mappers
from linchemin.cgu.syngraph import extract_reactions_from_syngraph, MonopartiteReacSynGraph
from linchemin.cgu.translate import translator
import pytest
import json


def test_basic_factory(capfd):
    perform_atom_mapping('namerxn', [])
    out, err = capfd.readouterr()
    assert 'NameRxn' in out
    with pytest.raises(KeyError) as ke:
        perform_atom_mapping('unavailable_mapper', [])
    assert "KeyError" in str(ke.type)


def test_helper_function():
    h = get_available_mappers()
    assert h
    assert 'namerxn' in h


def test_rxnmapper(az_path, ibm1_path):
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

    graph_ibm = json.loads(open(ibm1_path).read())
    syngraph = translator('ibm_retro', graph_ibm[0], 'syngraph', 'monopartite_reactions')
    reaction_list = extract_reactions_from_syngraph(syngraph)
    out = pipeline_atom_mapping(reaction_list)
    assert out is not None
    assert out.pipeline_success_rate != {}
    s = MonopartiteReacSynGraph(out.mapped_reactions)
    assert len(syngraph.graph) == len(s.graph)
