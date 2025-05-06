import json

import pytest

from linchemin.cgu.convert import converter
from linchemin.cgu.syngraph import MonopartiteReacSynGraph
from linchemin.cgu.translate import translator


def test_strategy_structure(ibm1_path):
    graph = json.loads(open(ibm1_path).read())
    syngraphs = translator(
        "ibm_retro", graph[0], "syngraph", out_data_model="bipartite"
    )
    mp_graph = converter(syngraphs, "monopartite_reactions")
    assert isinstance(mp_graph, MonopartiteReacSynGraph)

    with pytest.raises(TypeError) as ke:
        graph = json.loads(open(ibm1_path).read())
        converter(graph[0], "monopartite_reactions")
    assert "TypeError" in str(ke.type)

    with pytest.raises(KeyError) as ke:
        graph = json.loads(open(ibm1_path).read())
        syngraph = translator(
            "ibm_retro", graph[0], "syngraph", out_data_model="bipartite"
        )
        converter(syngraph, "networkx")
    assert "KeyError" in str(ke.type)


def test_bp_to_mpr(az_path):
    graphs = json.loads(open(az_path).read())
    bp_syngraph = translator(
        "az_retro", graphs[0], "syngraph", out_data_model="bipartite"
    )
    mp_syngraph = converter(bp_syngraph, "monopartite_reactions")
    mp_syngraph_t = translator(
        "az_retro", graphs[0], "syngraph", out_data_model="monopartite_reactions"
    )
    assert mp_syngraph == mp_syngraph_t


def test_bp_to_mpm(az_path):
    graphs = json.loads(open(az_path).read())
    bp_syngraphs = translator(
        "az_retro", graphs[0], "syngraph", out_data_model="bipartite"
    )
    mp_syngraph = converter(bp_syngraphs, "monopartite_molecules")
    mp_syngraph_t = translator(
        "az_retro", graphs[0], "syngraph", out_data_model="monopartite_molecules"
    )
    assert mp_syngraph == mp_syngraph_t


def test_mpr_to_mpm(az_path):
    graphs = json.loads(open(az_path).read())
    mpr_syngraphs = translator(
        "az_retro", graphs[0], "syngraph", out_data_model="monopartite_reactions"
    )
    mpm_syngraph = converter(mpr_syngraphs, "monopartite_molecules")

    mpm_syngraphs_t = translator(
        "az_retro", graphs[0], "syngraph", out_data_model="monopartite_molecules"
    )
    assert mpm_syngraph == mpm_syngraphs_t


def test_mpr_to_bp(az_path):
    graphs = json.loads(open(az_path).read())
    mpr_syngraphs = translator(
        "az_retro", graphs[1], "syngraph", out_data_model="monopartite_reactions"
    )
    bp_syngraph = converter(mpr_syngraphs, "bipartite")
    bp_syngraphs_t = translator(
        "az_retro", graphs[1], "syngraph", out_data_model="bipartite"
    )

    assert bp_syngraph == bp_syngraphs_t


def test_mpm_to_mpt(az_path):
    graphs = json.loads(open(az_path).read())
    mpm_syngraphs = translator(
        "az_retro", graphs[1], "syngraph", out_data_model="monopartite_molecules"
    )
    mpr_syngraph = converter(mpm_syngraphs, "monopartite_reactions")
    mpr_syngraph_t = translator(
        "az_retro", graphs[1], "syngraph", out_data_model="monopartite_reactions"
    )

    assert mpr_syngraph == mpr_syngraph_t


def test_mpm_to_bp(az_path):
    graphs = json.loads(open(az_path).read())
    mpm_syngraphs = translator(
        "az_retro", graphs[1], "syngraph", out_data_model="monopartite_molecules"
    )
    bp_syngraph = converter(mpm_syngraphs, "bipartite")
    bp_syngraph_t = translator(
        "az_retro", graphs[1], "syngraph", out_data_model="bipartite"
    )

    assert bp_syngraph == bp_syngraph_t
