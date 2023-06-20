from linchemin.interfaces.facade import facade, facade_helper
from linchemin.cgu.syngraph import MonopartiteReacSynGraph, BipartiteSynGraph
from rdkit.Chem import rdChemReactions
import pandas as pd

import unittest.mock
import unittest
import json


@unittest.mock.patch("linchemin.cgu.translate.ibm_dict_to_iron")
def test_translate(mock_os, ibm2_path):
    graph = json.loads(open(ibm2_path).read())
    output, metadata = facade("translate", input_format="ibm_retro", input_list=graph)
    mock_os.assert_called()
    assert type(output) == list and type(metadata) == dict
    output, metadata = facade(
        "translate",
        "ibm_retro",
        graph,
        out_format="syngraph",
        out_data_model="monopartite_reactions",
    )
    mock_os.assert_called()
    assert type(output[0]) == MonopartiteReacSynGraph

    graph.extend([{}])
    with unittest.TestCase().assertLogs(
        "linchemin.cgu.translate", level="WARNING"
    ) as cm:
        facade("translate", input_format="ibm_retro", input_list=graph)
    unittest.TestCase().assertIn(
        "While translating from IBM", cm.records[0].getMessage()
    )


@unittest.mock.patch("linchemin.rem.route_descriptors.NrBranches.compute_descriptor")
@unittest.mock.patch(
    "linchemin.rem.route_descriptors.NrReactionSteps.compute_descriptor"
)
def test_metrics(mock_n_steps, mock_branch, ibm2_path):
    graph = json.loads(open(ibm2_path).read())
    routes, meta = facade(
        "translate",
        "ibm_retro",
        graph,
        out_format="syngraph",
        out_data_model="monopartite_reactions",
    )
    descriptors, meta = facade("routes_descriptors", routes)
    mock_n_steps.assert_called()
    mock_branch.assert_called()
    assert type(descriptors) == pd.DataFrame
    for n in [5, 6, 7]:
        assert n in descriptors["nr_steps"]

    n_steps, meta = facade("routes_descriptors", routes, descriptors=["nr_steps"])
    mock_n_steps.assert_called()
    assert "nr_branches" not in n_steps.columns.any()
    routes.append(None)
    b, m = facade("routes_descriptors", routes, descriptors=["nr_branches", "metric"])
    mock_branch.assert_called()
    assert m["invalid_routes"] == 1
    assert m["errors"] is not []


def test_ged(az_path):
    graph = json.loads(open(az_path).read())
    routes_bp, m = facade(
        "translate",
        "az_retro",
        graph,
        out_format="syngraph",
        out_data_model="bipartite",
    )
    d, meta = facade("distance_matrix", routes_bp)
    assert len(d) == 6
    routes_mp, m2 = facade(
        "translate",
        "az_retro",
        graph,
        out_format="syngraph",
        out_data_model="monopartite_reactions",
    )
    d1, meta1 = facade("distance_matrix", routes_mp, ged_method="nx_ged")
    d2, meta2 = facade(
        "distance_matrix",
        routes_mp,
        ged_method="nx_ged",
        ged_params={
            "reaction_fp": "structure_fp",
            "reaction_fp_params": {
                "fpSize": 1024,
                "fpType": rdChemReactions.FingerprintType.MorganFP,
            },
            "reaction_similarity_name": "dice",
        },
    )
    assert meta2["graph_type"] == "monopartite"
    assert not d1.equals(d2)
    d3, meta3 = facade(
        "distance_matrix", routes_mp, ged_method="nx_ged", parallelization=True, n_cpu=8
    )
    assert d3.equals(d1)

    single_route = [routes_mp[0]]
    d3, meta3 = facade("distance_matrix", single_route)
    assert d3.empty
    assert meta3["errors"] != []


@unittest.mock.patch("linchemin.interfaces.facade.get_clustered_routes_metrics")
@unittest.mock.patch(
    "linchemin.rem.clustering.AgglomerativeClusterCalculator.get_clustering"
)
def test_clustering(mock_clusterer, mock_metrics, az_path):
    graph = json.loads(open(az_path).read())
    routes, m = facade(
        "translate",
        "az_retro",
        graph,
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
    cluster5, meta5 = facade("clustering", routes)
    assert meta5["invalid_routes"] == 1

    # If only 1 route is passed, an error is raised
    single_route = [routes[0]]
    with unittest.TestCase().assertLogs(
        "linchemin.rem.clustering", level="ERROR"
    ) as cm:
        facade("clustering", single_route)
    unittest.TestCase().assertIn("Less than 2 routes", cm.records[0].getMessage())


def test_clustering_parallelization(az_path):
    graph = json.loads(open(az_path).read())
    routes, m = facade(
        "translate",
        "az_retro",
        graph,
        out_format="syngraph",
        out_data_model="monopartite_reactions",
    )
    cluster4, meta4 = facade("clustering", routes, parallelization=True)
    assert all(cluster4[0].labels_) is not None


def test_subset(az_path):
    graph = json.loads(open(az_path).read())
    routes, meta = facade(
        "translate",
        "az_retro",
        graph,
        out_format="syngraph",
        out_data_model="monopartite_reactions",
    )
    subsets = facade("subsets", routes)
    assert type(subsets) == list


def test_find_duplicates(ibm2_path):
    graph = json.loads(open(ibm2_path).read())
    routes, meta = facade(
        "translate",
        "ibm_retro",
        graph,
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
    assert type(functionalities) == dict and "translate" in functionalities


def test_facade_helper_verbose(capfd):
    facade_helper(verbose=True)
    out, err = capfd.readouterr()
    assert "clustering" in out

    facade_helper("translate", verbose=True)
    out, err = capfd.readouterr()
    assert "out_data_model" in out


def test_parallelization_translate(capfd, ibm2_path):
    graph = json.loads(open(ibm2_path).read())
    output, metadata = facade(
        "translate",
        "ibm_retro",
        graph,
        out_data_model="monopartite_reactions",
        parallelization=True,
        n_cpu=8,
    )
    assert type(output) == list
    assert all(isinstance(x, MonopartiteReacSynGraph) for x in output)
    facade_helper("translate", verbose=True)
    out, err = capfd.readouterr()
    assert "parallelization" in out


def test_merging(ibm2_path):
    graph = json.loads(open(ibm2_path).read())
    routes, meta = facade("translate", "ibm_retro", graph, out_data_model="bipartite")
    tree = facade("merging", routes)
    assert type(tree) == BipartiteSynGraph
    roots = tree.get_roots()
    assert len(roots) == 1 and roots == routes[0].get_roots()

    tree_mp = facade("merging", routes, out_data_model="monopartite_reactions")
    assert type(tree_mp) == MonopartiteReacSynGraph


def test_reaction_extraction(mit_path):
    graph = json.loads(open(mit_path).read())
    routes, meta = facade(
        "translate", "mit_retro", graph, out_data_model="monopartite_reactions"
    )
    reactions, m = facade("extract_reactions_strings", routes)
    assert type(reactions) == list
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
    assert type(m["errors"][0]) == TypeError


@unittest.mock.patch("linchemin.cheminfo.atom_mapping.RxnMapper.map_chemical_equations")
@unittest.mock.patch("linchemin.interfaces.facade.pipeline_atom_mapping")
def test_mapping(mock_pipeline, mock_rxnmapper, ibm1_path):
    graph = json.loads(open(ibm1_path).read())
    routes, meta = facade(
        "translate", "ibm_retro", graph, out_data_model="monopartite_reactions"
    )
    # with mapping pipeline
    mapped_routes, meta = facade("atom_mapping", routes, mapper=None)
    mock_pipeline.assert_called()
    assert meta["mapping_success_rate"]
    for r in mapped_routes:
        assert type(r) == BipartiteSynGraph
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
        assert type(r) == MonopartiteReacSynGraph


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


def test_node_removal(ibm2_path):
    graph = json.loads(open(ibm2_path).read())
    routes, meta = facade(
        "translate",
        "ibm_retro",
        graph,
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
