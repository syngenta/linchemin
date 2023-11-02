import json
import unittest.mock

import pytest

from linchemin.cgu.syngraph import (
    MonopartiteReacSynGraph,
    extract_reactions_from_syngraph,
    merge_syngraph,
)
from linchemin.cgu.translate import translator
from linchemin.cheminfo.atom_mapping import (
    get_available_mappers,
    perform_atom_mapping,
    pipeline_atom_mapping,
)


def test_basic_factory(capfd):
    with pytest.raises(KeyError) as ke:
        perform_atom_mapping([], "unavailable_mapper")
    assert "UnavailableMapper" in str(ke.type)


def test_helper_function():
    h = get_available_mappers()
    assert h
    assert "namerxn" in h


@unittest.mock.patch("linchemin.services.rxnmapper.service.EndPoint.submit")
def test_rxnmapper(mock_rxnmapper_endpoint, az_path):
    graph_az = json.loads(open(az_path).read())
    syngraph = translator("az_retro", graph_az[0], "syngraph", "monopartite_reactions")
    reaction_list = extract_reactions_from_syngraph(syngraph)
    out = perform_atom_mapping(reaction_list, "rxnmapper")
    mock_rxnmapper_endpoint.assert_called()
    assert out is not None
    assert out.pipeline_success_rate == {}
    s = MonopartiteReacSynGraph(out.mapped_reactions)
    for parent, children in s.graph.items():
        assert parent.disconnection is not None
        assert parent.template is not None


@unittest.mock.patch("linchemin.services.namerxn.service.EndPoint.submit")
def test_namerxn_service(mock_namerxn_endpoint, ibm1_path):
    graph = json.loads(open(ibm1_path).read())
    syngraph = translator("ibm_retro", graph[2], "syngraph", "monopartite_reactions")
    reaction_list = extract_reactions_from_syngraph(syngraph)
    out = perform_atom_mapping(reaction_list, "namerxn")
    mock_namerxn_endpoint.assert_called()
    assert out is not None
    s = MonopartiteReacSynGraph(out.mapped_reactions)
    # disc1 = set()
    for parent, children in s.graph.items():
        assert parent.disconnection is not None
        assert parent.template is not None
    # With test file ibm2_path, even if the reactions are all mapped, some disconnections remain None;
    # the code below is used to debug -> The issue is due to deprotection reaction, where the new bond is with an H
    # and not detected by the current system
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


@unittest.mock.patch("linchemin.services.rxnmapper.service.EndPoint.submit")
@unittest.mock.patch("linchemin.services.namerxn.service.EndPoint.submit")
def test_pipeline(mock_namerxn, mock_rxnmapper, az_path):
    graph = json.loads(open(az_path).read())
    syngraph = translator("az_retro", graph[2], "syngraph", "monopartite_reactions")
    reaction_list = extract_reactions_from_syngraph(syngraph)
    out = pipeline_atom_mapping(reaction_list)
    mock_namerxn.assert_called()
    mock_rxnmapper.assert_called()
    assert out is not None
    assert out.pipeline_success_rate != {}
