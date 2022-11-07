from linchemin.cgu.convert import converter
from linchemin.cgu.translate import translator
from linchemin.cgu.syngraph import (SynGraph, BipartiteSynGraph, MonopartiteReacSynGraph)

import pytest
import json


def test_strategy_structure():
    graph = json.loads(open("data/ibmrxn_retro_output_raw.json").read())
    syngraphs = translator('ibm_retro', graph[0], 'syngraph', out_data_model='bipartite')
    mp_graph = converter(syngraphs, 'monopartite_reactions')
    assert type(mp_graph) == MonopartiteReacSynGraph

    with pytest.raises(TypeError) as ke:
        graph = json.loads(open("data/ibmrxn_retro_output_raw.json").read())
        converter(graph[0], 'monopartite_reactions')
    assert "TypeError" in str(ke.type)

    with pytest.raises(KeyError) as ke:
        graph = json.loads(open("data/ibmrxn_retro_output_raw.json").read())
        syngraph = translator('ibm_retro', graph[0], 'syngraph', out_data_model='bipartite')
        converter(syngraph, 'networkx')
    assert "KeyError" in str(ke.type)


def test_bp_to_mpr():
    graphs = json.loads(open("data/az_retro_output_raw.json").read())
    bp_syngraph = translator('az_retro', graphs[0], 'syngraph', out_data_model='bipartite')
    mp_syngraph = converter(bp_syngraph, 'monopartite_reactions')
    mp_syngraph_t = translator('az_retro', graphs[0], 'mp_syngraph', out_data_model='monopartite_reactions')
    assert mp_syngraph == mp_syngraph_t


def test_bp_to_mpm():
    graphs = json.loads(open("data/az_retro_output_raw.json").read())
    bp_syngraphs = translator('az_retro', graphs[0], 'syngraph', out_data_model='bipartite')
    mp_syngraph = converter(bp_syngraphs, 'monopartite_molecules')
    mp_syngraph_t = translator('az_retro', graphs[0], 'mp_mol_syngraph', out_data_model='monopartite_molecules')
    assert mp_syngraph == mp_syngraph_t


def test_mpr_to_mpm():
    graphs = json.loads(open("data/az_retro_output_raw.json").read())
    mpr_syngraphs = translator('az_retro', graphs[0], 'mp_syngraph', out_data_model='monopartite_reactions')
    mpm_syngraph = converter(mpr_syngraphs, 'monopartite_molecules')

    mpm_syngraphs_t = translator('az_retro', graphs[0], 'mp_mol_syngraph', out_data_model='monopartite_molecules')
    assert mpm_syngraph == mpm_syngraphs_t


def test_mpr_to_bp():
    graphs = json.loads(open("data/az_retro_output_raw.json").read())
    mpr_syngraphs = translator('az_retro', graphs[1], 'syngraph', out_data_model='monopartite_reactions')
    bp_syngraph = converter(mpr_syngraphs, 'bipartite')
    bp_syngraphs_t = translator('az_retro', graphs[1], 'syngraph', out_data_model='bipartite')

    assert bp_syngraph == bp_syngraphs_t


# Testing not implemented converters
def test_not_implemented():
    graphs = json.loads(open("data/az_retro_output_raw.json").read())
    mpr_syngraphs = translator('az_retro', graphs[1], 'syngraph', out_data_model='monopartite_molecules')
    with pytest.raises(NotImplementedError) as ke:
        converter(mpr_syngraphs, 'bipartite')
    assert "NotImplementedError" in str(ke.type)
    with pytest.raises(NotImplementedError) as ke:
        converter(mpr_syngraphs, 'monopartite_reactions')
    assert "NotImplementedError" in str(ke.type)
