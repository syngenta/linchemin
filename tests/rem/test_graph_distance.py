import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from rdkit.Chem import rdChemReactions

from linchemin.cgu.convert import converter
from linchemin.cgu.translate import translator
from linchemin.rem.graph_distance import (
    GraphDistanceError,
    build_similarity_matrix,
    compute_distance_matrix,
    compute_nodes_fingerprints,
    get_available_ged_algorithms,
    get_ged_default_parameters,
    get_ged_parameters,
    graph_distance_factory,
)


def test_similarity_factory(az_path):
    # Test similarity with oneself with monopartite syngraphs and factory workflow
    graph_az = json.loads(open(az_path).read())
    mp_syngraphs = [
        translator("az_retro", g, "syngraph", out_data_model="monopartite_reactions")
        for g in graph_az
    ]

    ged_identical = graph_distance_factory(
        mp_syngraphs[0], mp_syngraphs[0], ged_method="nx_ged"
    )
    assert ged_identical == 0.0

    ged = graph_distance_factory(mp_syngraphs[0], mp_syngraphs[3], ged_method="nx_ged")
    ged_params = graph_distance_factory(
        mp_syngraphs[0],
        mp_syngraphs[3],
        ged_method="nx_ged",
        ged_params={
            "reaction_fp": "structure_fp",
            "reaction_fp_params": {
                "fpSize": 1024,
                "fpType": rdChemReactions.FingerprintType.MorganFP,
            },
        },
    )
    assert ged not in [0.0, ged_params]

    ged_matrix = graph_distance_factory(
        mp_syngraphs[0], mp_syngraphs[3], ged_method="nx_ged_matrix"
    )
    assert ged_matrix == ged

    # Test that passing two different types of syngraphs raises an error "
    with pytest.raises(GraphDistanceError) as ke:
        syngraphs = [
            translator("az_retro", g, "syngraph", out_data_model="bipartite")
            for g in graph_az
        ]
        graph_distance_factory(mp_syngraphs[0], syngraphs[0], ged_method="nx_ged")
    assert "MismatchingGraph" in str(ke.type)

    # Test that asking for a ged method not available raises an error
    with pytest.raises(GraphDistanceError) as ke:
        syngraphs = [
            translator("az_retro", g, "syngraph", out_data_model="bipartite")
            for g in graph_az
        ]
        graph_distance_factory(
            mp_syngraphs[0], syngraphs[0], ged_method="non_available_method"
        )
    assert "UnavailableGED" in str(ke.type)


def test_bipartite_graph(az_path):
    # Test similarity with oneself using bipartite syngraphs
    graph_az = json.loads(open(az_path).read())
    syngraphs = [
        translator("az_retro", g, "syngraph", out_data_model="bipartite")
        for g in graph_az
    ]

    ged_identical = graph_distance_factory(
        syngraphs[0], syngraphs[0], ged_method="nx_ged"
    )

    assert ged_identical == 0.0

    ged = graph_distance_factory(syngraphs[0], syngraphs[3], ged_method="nx_ged")
    assert ged != 0.0


def test_compute_nodes_fingerprints(ibm1_path):
    # Test that compute_nodes_fingerprints returns the correct dictionary
    graph_ibm = json.loads(open(ibm1_path).read())
    syngraphs = [
        translator("ibm_retro", g, "syngraph", out_data_model="bipartite")
        for g in graph_ibm
    ]

    reaction_fp, molecule_fp = compute_nodes_fingerprints(
        syngraphs[1],
        reaction_fingerprints="structure_fp",
        molecular_fingerprint="rdkit",
    )
    assert len(reaction_fp) == 1 and len(molecule_fp) == 4

    mp_syngraphs = [converter(g, "monopartite_reactions") for g in syngraphs]
    mp_reaction_fp, mp_mol_fp = compute_nodes_fingerprints(
        mp_syngraphs[1],
        reaction_fingerprints="structure_fp",
        molecular_fingerprint="rdkit",
    )
    assert len(mp_reaction_fp) == 1 and mp_mol_fp == {}


def test_build_similarity_matrix(az_path):
    # Test the workflow for computing the similarity matrix
    graph_az = json.loads(open(az_path).read())
    mp_syngraphs = [
        translator("az_retro", g, "syngraph", out_data_model="monopartite_reactions")
        for g in graph_az
    ]
    reaction_fp1, molecule_fp1 = compute_nodes_fingerprints(
        mp_syngraphs[1],
        reaction_fingerprints="structure_fp",
        molecular_fingerprint="rdkit",
    )
    reaction_fp2, molecule_fp2 = compute_nodes_fingerprints(
        mp_syngraphs[2],
        reaction_fingerprints="structure_fp",
        molecular_fingerprint="rdkit",
    )
    matrix = build_similarity_matrix(reaction_fp1, reaction_fp2)
    assert type(matrix) == pd.DataFrame
    assert matrix.shape == (3, 2)


def test_similarity_matrix(az_path):
    # Test similarity with oneself using the similarity matrix workflow
    graph_az = json.loads(open(az_path).read())
    mp_syngraphs = [
        translator("az_retro", g, "syngraph", out_data_model="monopartite_reactions")
        for g in graph_az
    ]

    ged_identical = graph_distance_factory(
        mp_syngraphs[0], mp_syngraphs[0], ged_method="nx_ged_matrix"
    )
    assert ged_identical == 0.0

    ged = graph_distance_factory(
        mp_syngraphs[0], mp_syngraphs[3], ged_method="nx_ged_matrix"
    )
    assert ged != 0.0


def test_compute_distance_matrix(az_path, ibm1_path):
    graph = json.loads(open(az_path).read())
    mp_syngraphs = [
        translator("az_retro", g, "syngraph", out_data_model="monopartite_reactions")
        for g in graph
    ]
    m = compute_distance_matrix(mp_syngraphs, ged_method="nx_ged")
    diag = pd.Series([m.iat[n, n] for n in range(len(m))], index=[m.index, m.columns])
    routes = [graph.source for graph in mp_syngraphs]
    proof_diag = pd.Series(np.zeros(len(routes)), index=[m.index, m.columns])
    assert diag.all() == proof_diag.all()

    routes = json.loads(open(ibm1_path).read())
    mp_syngraphs = [
        translator("ibm_retro", r, "syngraph", out_data_model="monopartite_reactions")
        for r in routes
    ]
    m2 = compute_distance_matrix(mp_syngraphs, ged_method="nx_optimized_ged")
    diag2 = pd.Series(
        [m2.iat[n, n] for n in range(len(m2))], index=[m2.index, m2.columns]
    )
    r2 = [graph.source for graph in mp_syngraphs]
    proof_diag = pd.Series(np.zeros(len(r2)), index=[m2.index, m2.columns])
    assert diag2.all() == proof_diag.all()

    matrix_parallelization = compute_distance_matrix(
        mp_syngraphs, ged_method="nx_ged", parallelization=True, n_cpu=8
    )
    diag_p = pd.Series(
        [matrix_parallelization.iat[n, n] for n in range(len(matrix_parallelization))],
        index=[matrix_parallelization.index, matrix_parallelization.columns],
    )
    assert diag_p.all() == proof_diag.all()
    assert m2.equals(matrix_parallelization)


def test_optimized_ged(az_path):
    graph_az = json.loads(open(az_path).read())
    syngraphs = [
        translator("az_retro", g, "syngraph", out_data_model="bipartite")
        for g in graph_az
    ]
    nx_graphs = [
        translator("syngraph", s, "networkx", out_data_model="bipartite")
        for s in syngraphs
    ]
    ged = nx.graph_edit_distance(
        nx_graphs[0], nx_graphs[1], node_match=lambda x, y: x == y
    )
    # nx.optimize_graph_edit_distance returns a generator
    opt_ged = nx.optimize_graph_edit_distance(
        nx_graphs[0], nx_graphs[1], node_match=lambda x, y: x == y
    )
    for g in opt_ged:
        min_g = g
    # The minimum value generated by optimize_graph_edit_distance is equal to the value returned by graph_edit_distance
    assert min_g == ged


def test_optimized_ged_workflow(az_path):
    graph_az = json.loads(open(az_path).read())
    syngraphs = [
        translator("az_retro", g, "syngraph", out_data_model="bipartite")
        for g in graph_az
    ]
    ged_bp = graph_distance_factory(
        syngraphs[0], syngraphs[3], ged_method="nx_optimized_ged"
    )
    ged_bp_std = graph_distance_factory(syngraphs[0], syngraphs[3], ged_method="nx_ged")
    assert ged_bp == ged_bp_std
    mp_syngraphs = [
        translator("az_retro", g, "syngraph", out_data_model="monopartite_reactions")
        for g in graph_az
    ]
    ged_mp = graph_distance_factory(
        mp_syngraphs[0], mp_syngraphs[3], ged_method="nx_optimized_ged"
    )
    ged_mp_std = graph_distance_factory(
        mp_syngraphs[0], mp_syngraphs[3], ged_method="nx_ged"
    )
    assert ged_mp == ged_mp_std


def test_get_available_ged():
    assert (
        type(get_available_ged_algorithms()) == dict
        and "nx_ged" in get_available_ged_algorithms()
    )


def test_get_default():
    assert (
        type(get_ged_default_parameters()) == dict
        and "reaction_similarity_name" in get_ged_default_parameters()
    )


def test_ged_params():
    assert (
        type(get_ged_parameters()) == dict
        and "reaction_fp_params" in get_ged_default_parameters()
    )
