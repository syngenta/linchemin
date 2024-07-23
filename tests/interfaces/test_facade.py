import json
import unittest
from unittest.mock import MagicMock, Mock, patch

import pandas as pd

from linchemin.cgu.syngraph import BipartiteSynGraph, MonopartiteReacSynGraph
from linchemin.interfaces.facade import (
    DescriptorError,
    GedFacade,
    RoutesDescriptorsFacade,
    TranslateFacade,
    facade,
    facade_helper,
)


@unittest.mock.patch("linchemin.interfaces.facade.translator")
def test_translate(mock_translate, ibm2_as_dict):
    """To test the TranslateFacade correctly call the translator function"""
    facade_translate = TranslateFacade()
    output, metadata = facade_translate.perform_functionality(
        input_format="ibm_retro", input_list=ibm2_as_dict
    )
    mock_translate.assert_called()
    assert isinstance(output, list) and isinstance(metadata, dict)


@patch("linchemin.interfaces.facade.descriptor_calculator")
@patch("linchemin.interfaces.facade.get_available_descriptors")
def test_descriptors_valid_routes(
    mock_get_available_descriptors, mock_descriptor_calculator
):
    # Mock the get_available_descriptors function to return a list of descriptors
    mock_get_available_descriptors.return_value = ["nr_steps", "nr_branches"]

    # Mock the descriptor_calculator function to return a value based on the descriptor name
    def descriptor_calculator_side_effect(route, descriptor):
        return f"{descriptor}_value"

    mock_descriptor_calculator.side_effect = descriptor_calculator_side_effect

    # Create a list of mock routes with unique identifiers
    mock_routes = [MagicMock(uid=i) for i in range(3)]

    # Instantiate the facade and call the method under test
    f = RoutesDescriptorsFacade()
    output, meta = f.perform_functionality(mock_routes)

    # Assertions to verify the method behavior
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
    f = RoutesDescriptorsFacade()
    output, meta = f.perform_functionality(mock_routes, descriptors=["nr_steps"])

    # Assertions to verify the method behavior
    assert isinstance(output, pd.DataFrame)
    assert "nr_steps" in output.columns
    assert all(output["nr_steps"] == "calculated_value")
    assert meta["invalid_routes"] == 1  # One route was None
    assert meta["errors"] == []


@patch("linchemin.interfaces.facade.descriptor_calculator")
def test_descriptors_error(mock_descriptor_calculator):
    # Mock the descriptor_calculator to raise a DescriptorError for a specific descriptor
    def descriptor_calculator_side_effect(route, descriptor):
        if descriptor == "error_descriptor":
            raise DescriptorError("Error calculating descriptor")
        return "calculated_value"

    mock_descriptor_calculator.side_effect = descriptor_calculator_side_effect

    # Create a list of mock routes
    mock_routes = [MagicMock(uid=i) for i in range(3)]

    # Instantiate the facade and call the method under test
    f = RoutesDescriptorsFacade()
    output, meta = f.perform_functionality(
        mock_routes, descriptors=["nr_steps", "error_descriptor"]
    )

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


@unittest.mock.patch("linchemin.interfaces.facade.get_clustered_routes_metrics")
@unittest.mock.patch(
    "linchemin.rem.clustering.AgglomerativeClusterCalculator.get_clustering"
)
def test_clustering(mock_clusterer, mock_metrics, az_as_dict):
    routes, _ = facade(
        "translate",
        "az_retro",
        az_as_dict,
        out_format="syngraph",
        out_data_model="monopartite_reactions",
    )
    # Test with all default parameters
    facade("clustering", routes)
    mock_clusterer.assert_called()

    # Test with some changed parameters
    facade(
        "clustering",
        routes,
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
    mock_clusterer.assert_called()

    # Test compute cluster metrics
    facade("clustering", routes, compute_metrics=True)
    mock_clusterer.assert_called()
    mock_metrics.assert_called()

    # If one of the route is None, it is recognized as such
    routes.append(None)
    _, meta5 = facade("clustering", routes)
    assert meta5["invalid_routes"] == 1

    # If only 1 route is passed, an error is raised
    single_route = [routes[0]]
    with unittest.TestCase().assertLogs(
        "linchemin.rem.clustering", level="ERROR"
    ) as cm:
        facade("clustering", single_route)
    unittest.TestCase().assertIn("Less than 2 routes", cm.records[0].getMessage())


@patch("linchemin.interfaces.facade.clusterer")
def test_clustering_parallelization(mock_cluster, bp_syngraph_instance):
    facade("clustering", [bp_syngraph_instance], parallelization=True)
    mock_cluster.assert_called()


def test_subset(az_as_dict):
    routes, _ = facade(
        "translate",
        "az_retro",
        az_as_dict,
        out_format="syngraph",
        out_data_model="monopartite_reactions",
    )
    subsets = facade("subsets", routes)
    assert isinstance(subsets, list)


def test_find_duplicates(ibm2_as_dict):
    routes, _ = facade(
        "translate",
        "ibm_retro",
        ibm2_as_dict,
        out_format="syngraph",
        out_data_model="monopartite_reactions",
    )
    duplicates = facade("duplicates", routes)
    assert duplicates is None
    routes.append(routes[0])
    duplicates = facade("duplicates", routes)
    assert duplicates


def test_facade_helper():
    functionalities = facade_helper()
    assert isinstance(functionalities, dict) and "translate" in functionalities


def test_facade_helper_verbose(capfd):
    facade_helper(verbose=True)
    out, _ = capfd.readouterr()
    assert "clustering" in out

    facade_helper("translate", verbose=True)
    out, _ = capfd.readouterr()
    assert "out_data_model" in out


def test_parallelization_translate(capfd, ibm2_as_dict):
    output, _ = facade(
        "translate",
        "ibm_retro",
        ibm2_as_dict,
        out_data_model="monopartite_reactions",
        parallelization=True,
        n_cpu=8,
    )
    assert isinstance(output, list)
    assert all(isinstance(x, MonopartiteReacSynGraph) for x in output)
    facade_helper("translate", verbose=True)
    out, _ = capfd.readouterr()
    assert "parallelization" in out


def test_merging(ibm2_as_dict):
    routes, _ = facade(
        "translate", "ibm_retro", ibm2_as_dict, out_data_model="bipartite"
    )
    tree = facade("merging", routes)
    assert isinstance(tree, BipartiteSynGraph)
    roots = tree.get_roots()
    assert len(roots) == 1 and roots == routes[0].get_roots()

    tree_mp = facade("merging", routes, out_data_model="monopartite_reactions")
    assert isinstance(tree_mp, MonopartiteReacSynGraph)


def test_reaction_extraction(mit_path):
    graph = json.loads(open(mit_path).read())
    routes, _ = facade(
        "translate", "mit_retro", graph, out_data_model="monopartite_reactions"
    )
    reactions, m = facade("extract_reactions_strings", routes)
    assert isinstance(reactions, list)
    assert len(routes) == len(reactions)

    # If one route in the list is not a SynGraph, an exception is captured in the 'errors' field of metadata
    r_nx = facade(
        "translate",
        "mit_retro",
        [graph[0]],
        out_format="networkx",
        out_data_model="monopartite_reactions",
    )
    routes.append(r_nx[0][0])
    reactions, m = facade("extract_reactions_strings", routes)
    assert len(reactions) == 4
    assert isinstance(m["errors"][0], TypeError)


@unittest.mock.patch("linchemin.cheminfo.atom_mapping.RxnMapper.map_chemical_equations")
@unittest.mock.patch("linchemin.interfaces.facade.pipeline_atom_mapping")
def test_mapping(mock_pipeline, mock_rxnmapper, ibm1_as_dict):
    routes, meta = facade(
        "translate", "ibm_retro", ibm1_as_dict, out_data_model="monopartite_reactions"
    )
    # with mapping pipeline
    mapped_routes, meta = facade("atom_mapping", routes, mapper=None)
    mock_pipeline.assert_called()
    assert meta["mapping_success_rate"]
    for r in mapped_routes:
        assert isinstance(r, BipartiteSynGraph)
        assert r.source

    # with other values
    mapped_routes, meta = facade(
        "atom_mapping",
        routes,
        mapper="rxnmapper",
        out_data_model="monopartite_reactions",
    )
    mock_rxnmapper.assert_called()
    assert meta["mapping_success_rate"]
    for r in mapped_routes:
        assert isinstance(r, MonopartiteReacSynGraph)


def test_routes_sanity_checks():
    route_cycle = [
        {
            "query_id": 0,
            "output_string": "[CH3:3][C:2]#[N:1].[OH2:4]>>[CH3:3][C:2]([OH:4])=[O:4]",
        },
        {
            "query_id": 1,
            "output_string": "O[C:2]([CH3:1])=[O:3].[CH3:4][NH2:5]>>[CH3:1][C:2](=[O:3])[NH:5][CH3:4]",
        },
        {
            "query_id": 2,
            "output_string": "[CH3:5][NH:4][C:2]([CH3:1])=[O:3].[OH2:6]>>[CH3:1][C:2]([OH:6])=[O:3]",
        },
        {
            "query_id": 3,
            "output_string": "ClP(Cl)[Cl:4].O[C:2]([CH3:1])=[O:3]>>[Cl:4][C:2]([CH3:1])=[O:3]",
        },
    ]
    route_isolated_nodes = [
        {
            "output_string": "Cl[C:2]([CH3:1])=[O:3].[CH3:4][OH:5]>>[CH3:1][C:2](=[O:3])[O:5][CH3:4]",
            "query_id": "0",
        },
        {
            "output_string": "[CH3:5][O:4][C:3]([CH3:2])=[O:1]>>[CH3:2][C:3]([OH:4])=[O:1]",
            "query_id": "1",
        },
        {
            "output_string": "[CH3:4][C:5](Cl)=[O:6].CC(O)=O.[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][O:3][C:5]([CH3:4])=[O:6]",
            "query_id": "2",
        },
        {
            "output_string": "O=[C:2](OC[CH3:4])[CH3:1].[Li][CH3:3]>>[CH2:1]=[C:2]([CH3:3])[CH3:4]",
            "query_id": "3",
        },
    ]
    routes = [
        BipartiteSynGraph(route_cycle),
        BipartiteSynGraph(route_isolated_nodes),
        None,
    ]

    checked_routes, meta = facade("routes_sanity_checks", routes, checks=None)
    assert len(checked_routes) == len(routes) - meta["invalid_routes"]
    assert [isinstance(r, BipartiteSynGraph) for r in checked_routes]
    assert meta["invalid_routes"] == 1
    assert [len(r.get_roots()) == 1 for r in checked_routes]

    checked_routes, meta = facade(
        "routes_sanity_checks",
        routes,
        checks=["cycle_check"],
        out_data_model="monopartite_reactions",
    )
    assert len(checked_routes) == len(routes) - meta["invalid_routes"]
    assert [isinstance(r, MonopartiteReacSynGraph) for r in checked_routes]
    assert [len(r.get_roots()) == 1 for r in checked_routes]

    assert facade_helper(functionality="routes_sanity_checks")


def test_node_removal(ibm2_as_dict):
    routes, meta = facade(
        "translate",
        "ibm_retro",
        ibm2_as_dict,
        out_data_model="monopartite_reactions",
    )

    with unittest.TestCase().assertLogs(
        "linchemin.cgu.syngraph_operations", level="WARNING"
    ) as cm:
        node_to_remove = "COc1cc(OC)c2c(=O)cc(-c3ccccc3Cl)oc2c1C1=CCN(C)CC1O.Cc1cc(C)nc(C)c1.[Li]I>>CN1CC=C(c2c(O)cc(O)c3c(=O)cc(-c4ccccc4Cl)oc23)C(O)C1"
        routes_without_node, meta = facade("node_removal", routes, node_to_remove)
    unittest.TestCase().assertEqual(len(cm.records), 17)
    unittest.TestCase().assertIn(
        "The selected node is not present in the input graph",
        cm.records[0].getMessage(),
    )
    assert len(routes_without_node) == len(routes)
    assert meta["unchanged_routes"] == 17
    assert meta["modified_routes"] == 1
