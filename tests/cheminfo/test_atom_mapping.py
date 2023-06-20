from linchemin.cheminfo.atom_mapping import (
    pipeline_atom_mapping,
    perform_atom_mapping,
    get_available_mappers,
)
from linchemin.cgu.syngraph import MonopartiteReacSynGraph
from linchemin.cgu.syngraph_operations import (
    merge_syngraph,
    extract_reactions_from_syngraph,
)
from linchemin.cgu.translate import translator

import pytest
import json
import unittest.mock


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
    for parent, children in s.graph.items():
        assert parent.disconnection is not None
        assert parent.template is not None


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
