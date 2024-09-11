import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from linchemin.cgu.syngraph import BipartiteSynGraph, MonopartiteReacSynGraph
from linchemin.interfaces.facade import (
    AtomMappingFacade,
    ClusteringFacade,
    DescriptorError,
    Facade,
    FacadeFactory,
    GedFacade,
    GraphTypeError,
    ReactionExtractionFacade,
    RouteSanityCheckFacade,
    RoutesDescriptorsFacade,
    TranslateFacade,
    UnavailableFunctionality,
    facade,
    facade_helper,
)


@unittest.mock.patch("linchemin.interfaces.facade.translator")
def test_translate(mock_translate, ibm2_as_dict):
    """To test the TranslateFacade correctly call the translator function"""
    facade_translate = TranslateFacade(
        input_format="ibm_retro", out_data_model="monopartite_molecules"
    )
    output, metadata = facade_translate.perform_functionality(routes=ibm2_as_dict)
    mock_translate.assert_called()
    assert isinstance(output, list) and isinstance(metadata, dict)

    # missing mandatory parameter
    with pytest.raises(TypeError):
        TranslateFacade()


def test_translate_get_options():
    params = TranslateFacade.get_available_options()
    assert "routes" in params
    assert "input_format" in params


def test_parallelization_translate(capfd, bp_syngraph_instance):
    with patch("linchemin.interfaces.facade.translator"), patch(
        "linchemin.interfaces.facade.mp.Pool"
    ) as mock_mp:
        routes = [bp_syngraph_instance] * 5
        output, _ = TranslateFacade(
            input_format="syngraph",
            output_format="networkx",
            out_data_model="monopartite_reactions",
            parallelization=True,
            n_cpu=8,
        ).perform_functionality(
            routes,
        )
        mock_mp.assert_called()
    assert isinstance(output, list)
    assert all(isinstance(x, MonopartiteReacSynGraph) for x in output)


@patch("linchemin.interfaces.facade.descriptor_calculator")
@patch("linchemin.interfaces.facade.get_available_descriptors")
def test_descriptors_valid_routes(
    mock_get_available_descriptors, mock_descriptor_calculator
):
    # Mock the get_available_descriptors function to return a list of descriptors
    mock_get_available_descriptors.return_value = ["nr_steps", "nr_branches"]

    # Mock the descriptor_calculator function to
    # return a value based on the descriptor name
    def descriptor_calculator_side_effect(route, descriptor):
        return f"{descriptor}_value"

    mock_descriptor_calculator.side_effect = descriptor_calculator_side_effect

    # Create a list of mock routes with unique identifiers
    mock_routes = [MagicMock(uid=i) for i in range(3)]

    # Instantiate the facade and call the method under test
    f = RoutesDescriptorsFacade()
    output, meta = f.perform_functionality(mock_routes)

    assert isinstance(output, pd.DataFrame)
    assert isinstance(output.attrs["configuration"], list)
    assert "nr_steps" in output.columns
    assert "nr_branches" in output.columns
    assert all(output["nr_steps"] == "nr_steps_value")
    assert all(output["nr_branches"] == "nr_branches_value")
    assert meta["invalid_routes"] == 0
    assert meta["errors"] == []
    assert meta["descriptors"] == ["nr_steps", "nr_branches"]

    # Verify that descriptor_calculator was called correctly
    assert mock_descriptor_calculator.call_count == len(mock_routes) * len(
        mock_get_available_descriptors.return_value
    )


@patch("linchemin.interfaces.facade.descriptor_calculator")
def test_descriptors_mixed_routes(mock_descriptor_calculator):
    # Mock the descriptor_calculator function to return a fixed value
    mock_descriptor_calculator.return_value = "calculated_value"

    # Create a list of mock routes with one None value
    mock_routes = [MagicMock(uid=1), None, MagicMock(uid=2)]

    # Instantiate the facade and call the method under test
    f = RoutesDescriptorsFacade(["nr_steps"])
    output, meta = f.perform_functionality(mock_routes)

    # Assertions to verify the method behavior
    assert isinstance(output, pd.DataFrame)
    assert "nr_steps" in output.columns
    assert all(output["nr_steps"] == "calculated_value")
    assert meta["invalid_routes"] == 1  # One route was None
    assert meta["errors"] == []


@patch("linchemin.interfaces.facade.descriptor_calculator")
def test_descriptors_error(mock_descriptor_calculator):
    # Mock the descriptor_calculator to raise a
    # DescriptorError for a specific descriptor
    def descriptor_calculator_side_effect(route, descriptor):
        if descriptor == "error_descriptor":
            raise DescriptorError("Error calculating descriptor")
        return "calculated_value"

    mock_descriptor_calculator.side_effect = descriptor_calculator_side_effect

    # Create a list of mock routes
    mock_routes = [MagicMock(uid=i) for i in range(3)]

    # Instantiate the facade and call the method under test
    f = RoutesDescriptorsFacade(["nr_steps", "error_descriptor"])
    output, meta = f.perform_functionality(mock_routes)

    # Assertions to verify the method behavior
    assert isinstance(output, pd.DataFrame)
    assert "error_descriptor" not in output.columns
    assert len(meta["errors"]) == 1
    assert isinstance(meta["errors"][0], DescriptorError)


@patch("linchemin.interfaces.facade.compute_distance_matrix")
def test_ged(mock_ged, bp_syngraph_instance):
    f = GedFacade()
    f.perform_functionality(routes=[bp_syngraph_instance])
    mock_ged.assert_called()


def test_ged_options():
    d = GedFacade().get_available_options()
    assert "routes" in d
    assert "ged_method" in d
    assert "parallelization" in d


@unittest.mock.patch("linchemin.interfaces.facade.get_clustered_routes_metrics")
@unittest.mock.patch(
    "linchemin.rem.clustering.AgglomerativeClusterCalculator.get_clustering"
)
def test_clustering(mock_clusterer, mock_metrics, mpr_syngraph_instance):
    f = ClusteringFacade(
        ged_method="nx_ged",
        clustering_method="agglomerative_cluster",
        ged_params={
            "reaction_fp": "difference_fp",
            "reaction_fp_params": {"fpSize": 1024},
            "reaction_similarity_name": "dice",
        },
        save_dist_matrix=True,
        linkage="average",
    )
    routes = [mpr_syngraph_instance] * 5

    f.perform_functionality(routes)
    mock_clusterer.assert_called()

    # Test compute cluster metrics
    f = ClusteringFacade(compute_metrics=True)
    f.perform_functionality(
        routes,
    )
    mock_clusterer.assert_called()
    mock_metrics.assert_called()

    # If one of the route is None, it is recognized as such
    f = ClusteringFacade()
    routes.append(None)
    _, meta5 = f.perform_functionality(routes)
    assert meta5["invalid_routes"] == 1

    # If only 1 route is passed, an error is raised
    single_route = [routes[0]]
    with unittest.TestCase().assertLogs(
        "linchemin.rem.clustering", level="ERROR"
    ) as cm:
        f.perform_functionality(single_route)
    unittest.TestCase().assertIn("Less than 2 routes", cm.records[0].getMessage())


@patch("linchemin.interfaces.facade.clusterer")
def test_clustering_parallelization(mock_cluster, bp_syngraph_instance):
    ClusteringFacade(parallelization=True).perform_functionality([bp_syngraph_instance])
    mock_cluster.assert_called()


def test_reaction_extraction(bp_syngraph_instance, iron_w_smiles):
    routes = [bp_syngraph_instance] * 2
    with patch(
        "linchemin.interfaces.facade.extract_reactions_from_syngraph"
    ) as mock_extract:
        reactions = ReactionExtractionFacade().perform_functionality(routes)
        mock_extract.assert_called()
        assert len(routes) == len(reactions)


def test_reaction_extraction_with_errors(mocker, mpr_syngraph_instance):
    mock_extract = mocker.patch(
        "linchemin.interfaces.facade.extract_reactions_from_syngraph"
    )
    mock_extract.side_effect = [
        ["reaction_1"],
        ["reaction_2"],
        GraphTypeError("Invalid graph format"),
    ]

    routes = [mpr_syngraph_instance] * 3

    # Call the perform_functionality method
    reactions, meta = ReactionExtractionFacade().perform_functionality(routes)

    assert reactions == [
        {mpr_syngraph_instance.uid: ["reaction_1"]},
        {mpr_syngraph_instance.uid: ["reaction_2"]},
    ]
    assert meta["nr_routes_in_input_list"] == 3
    assert meta["invalid_routes"] == 0
    assert meta["errors"]

    mock_extract.assert_has_calls(
        [
            unittest.mock.call(mpr_syngraph_instance),
            unittest.mock.call(mpr_syngraph_instance),
            unittest.mock.call(mpr_syngraph_instance),
        ]
    )


@unittest.mock.patch("linchemin.cheminfo.atom_mapping.RxnMapper.map_chemical_equations")
@unittest.mock.patch("linchemin.interfaces.facade.pipeline_atom_mapping")
def test_mapping(mock_pipeline, mock_rxnmapper, mpr_syngraph_instance):
    routes = [mpr_syngraph_instance] * 2

    f = AtomMappingFacade(mapper=None)
    # with mapping pipeline
    mapped_routes, meta = f.perform_functionality(routes)
    mock_pipeline.assert_called()
    assert meta["mapping_success_rate"]
    for r in mapped_routes:
        assert isinstance(r, MonopartiteReacSynGraph)
        assert r.source

    # using default mapper=rxnmapper
    f = AtomMappingFacade()
    mapped_routes, meta = f.perform_functionality(
        routes,
    )
    mock_rxnmapper.assert_called()
    assert meta["mapping_success_rate"]
    for r in mapped_routes:
        assert isinstance(r, MonopartiteReacSynGraph)


def test_routes_sanity_checks(mocker, mpr_syngraph_instance):
    mock_checker = mocker.patch("linchemin.interfaces.facade.route_checker")
    mock_checker.side_effect = [
        mpr_syngraph_instance,
        mpr_syngraph_instance,
    ]
    routes = [mpr_syngraph_instance, mpr_syngraph_instance, None]
    f = RouteSanityCheckFacade(checks=None)
    checked_routes, meta = f.perform_functionality(routes)
    mock_checker.assert_called()
    assert len(checked_routes) == len(routes) - meta["invalid_routes"]
    assert [isinstance(r, BipartiteSynGraph) for r in checked_routes]
    assert meta["invalid_routes"] == 1


def test_facade_registration():
    class TestFacade(Facade):
        name = "test_facade"
        info = "Some info"

        def __init__(self, param):
            self.param = param

        def perform_functionality(self, routes: list) -> tuple:
            """Mocked abstract method"""

    # Register the TestService
    FacadeFactory.register_facade(TestFacade)
    # Assert: Check if the facade is now registered
    assert "test_facade" in FacadeFactory.list_functionalities()

    facade_class = FacadeFactory.select_functionality("test_facade")
    # Assert: Verify that the retrieved facade class is TestFacade
    assert facade_class is TestFacade


def test_facade_mechanism(ibm2_as_dict):
    with patch("linchemin.interfaces.facade.TranslateFacade"):
        out, meta = facade("translate", routes=ibm2_as_dict, input_format="ibm_retro")
        assert out
        assert isinstance(out, list)
        assert meta
        assert isinstance(meta, dict)

    # a non-existing facade functionality  is requested
    with pytest.raises(UnavailableFunctionality):
        facade("not_existing_facade", routes=ibm2_as_dict)

    # an argument not compatible with the selected facade is passed
    with pytest.raises(TypeError):
        facade(
            "translate",
            routes=ibm2_as_dict,
            input_format="ibm_retro",
            clustering="hdbscan",
        )


def test_facade_helper(capfd):
    functionalities = facade_helper()
    assert isinstance(functionalities, dict) and "translate" in functionalities

    facade_helper("translate", verbose=True)
    out, _ = capfd.readouterr()
    assert "parallelization" in out


def test_facade_helper_verbose(capfd):
    facade_helper(verbose=True)
    out, _ = capfd.readouterr()
    assert "clustering" in out

    facade_helper("translate", verbose=True)
    out, _ = capfd.readouterr()
    assert "out_data_model" in out
