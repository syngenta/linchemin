import json
import unittest.mock

import pytest

from linchemin.cgu.syngraph import MonopartiteReacSynGraph
from linchemin.cgu.syngraph_operations import extract_reactions_from_syngraph
from linchemin.cgu.translate import translator
from linchemin.cheminfo.atom_mapping import (
    get_available_mappers,
    perform_atom_mapping,
    pipeline_atom_mapping,
)


def test_basic_factory(capfd):
    with pytest.raises(KeyError) as ke:
        perform_atom_mapping(reactions_list=[], mapper_name="unavailable_mapper")
    assert "UnavailableMapper" in str(ke.type)


def test_helper_function():
    h = get_available_mappers()
    assert h
    assert "namerxn" in h


@unittest.mock.patch("linchemin.services.rxnmapper.service.EndPoint.submit")
def test_rxnmapper(mock_rxnmapper_endpoint, az_path):
    graphs = json.loads(open(az_path).read())
    graph = graphs[0]
    syngraph = translator(
        input_format="az_retro",
        original_graph=graph,
        output_format="syngraph",
        out_data_model="monopartite_reactions",
    )
    reaction_list = extract_reactions_from_syngraph(syngraph)
    out = perform_atom_mapping(reactions_list=reaction_list, mapper_name="rxnmapper")
    mock_rxnmapper_endpoint.assert_called()
    assert out is not None
    assert out.pipeline_success_rate == {}
    s = MonopartiteReacSynGraph(out.mapped_reactions)
    for parent, children in s.graph.items():
        assert parent.disconnection is not None
        assert parent.template is not None


@unittest.mock.patch("linchemin.services.namerxn.service.EndPoint.submit")
def test_namerxn_service(mock_namerxn_endpoint, ibm1_path, az_path):
    graphs = json.loads(open(az_path).read())
    graph = graphs[0]
    syngraph = translator(
        input_format="az_retro",
        original_graph=graph,
        output_format="syngraph",
        out_data_model="monopartite_reactions",
    )
    reaction_list = extract_reactions_from_syngraph(syngraph)
    out = perform_atom_mapping(reactions_list=reaction_list, mapper_name="namerxn")
    mock_namerxn_endpoint.assert_called()
    assert out is not None
    s = MonopartiteReacSynGraph(out.mapped_reactions)
    for parent, children in s.graph.items():
        assert parent.disconnection is not None
        assert parent.template is not None


@unittest.mock.patch("linchemin.services.rxnmapper.service.EndPoint.submit")
@unittest.mock.patch("linchemin.services.namerxn.service.EndPoint.submit")
def test_pipeline(mock_namerxn, mock_rxnmapper, az_path):
    graphs = json.loads(open(az_path).read())
    graph = graphs[0]
    syngraph = translator("az_retro", graph, "syngraph", "monopartite_reactions")
    reaction_list = extract_reactions_from_syngraph(syngraph)
    out = pipeline_atom_mapping(reaction_list)
    mock_namerxn.assert_called()
    mock_rxnmapper.assert_called()
    assert out is not None
    assert out.pipeline_success_rate != {}
