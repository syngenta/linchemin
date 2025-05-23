import json
import math

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from rdkit.Chem import rdChemReactions

from linchemin.cgu.convert import converter
from linchemin.cgu.translate import translator
from linchemin.rem.graph_distance import (
    ChemicalSimilarityParameters,
    GedNx,
    GraphDistanceError,
    build_similarity_matrix,
    compute_distance_matrix,
    get_available_ged_algorithms,
    get_ged_default_parameters,
    get_ged_parameters,
    get_mol_fp_dict,
    get_reactions_fp_dict,
    graph_distance_factory,
)


def test_similarity_factory(az_as_dict, mpr_syngraph_instance):
    # Test similarity with oneself with monopartite syngraphs and factory workflow
    params = ChemicalSimilarityParameters()
    ged = GedNx()
    ged_identical = ged.compute_ged(
        mpr_syngraph_instance, mpr_syngraph_instance, params
    )
    assert math.isclose(ged_identical, 0.0, rel_tol=1e-9)

    mp_syngraphs = [
        translator("az_retro", g, "syngraph", out_data_model="monopartite_reactions")
        for g in az_as_dict
    ]
    ged_default_params = ged.compute_ged(mp_syngraphs[0], mp_syngraphs[3], params)

    other_params = ChemicalSimilarityParameters(
        reaction_fingerprint="structure_fp",
        reaction_fp_params={
            "fpSize": 1024,
            "fpType": rdChemReactions.FingerprintType.MorganFP,
        },
    )

    ged_other_params = ged.compute_ged(mp_syngraphs[0], mp_syngraphs[3], other_params)
    assert ged_default_params not in [0.0, ged_other_params]

    ged_matrix = graph_distance_factory(
        mp_syngraphs[0], mp_syngraphs[3], ged_method="nx_ged_matrix"
    )
    assert ged_matrix == ged_default_params


def test_graph_distance_factory_wrong_type(mpr_syngraph_instance, bp_syngraph_instance):
    # Test that passing two different types of syngraphs raises an error "
    with pytest.raises(GraphDistanceError) as ke:
        graph_distance_factory(
            mpr_syngraph_instance, bp_syngraph_instance, ged_method="nx_ged"
        )
    assert "MismatchingGraph" in str(ke.type)


def test_graph_distance_factory_unavailable_ged(mpr_syngraph_instance):
    # Test that asking for a ged method not available raises an error
    with pytest.raises(GraphDistanceError) as ke:
        graph_distance_factory(
            mpr_syngraph_instance,
            mpr_syngraph_instance,
            ged_method="non_available_method",
        )
    assert "UnavailableGED" in str(ke.type)


def test_bipartite_graph(az_as_dict):
    # Test similarity with oneself using bipartite syngraphs
    syngraphs = [
        translator("az_retro", g, "syngraph", out_data_model="bipartite")
        for g in az_as_dict
    ]

    ged_identical = graph_distance_factory(
        syngraphs[0], syngraphs[0], ged_method="nx_ged"
    )
    assert math.isclose(ged_identical, 0.0, rel_tol=1e-9)

    ged = graph_distance_factory(syngraphs[0], syngraphs[3], ged_method="nx_ged")
    assert not math.isclose(ged, 0.0, rel_tol=1e-9)


def test_compute_fingerprints(ibm1_as_dict, mpr_syngraph_instance):
    syngraphs = [
        translator("ibm_retro", g, "syngraph", out_data_model="bipartite")
        for g in ibm1_as_dict
    ]

    reaction_fp = get_reactions_fp_dict(
        syngraphs[1],
        reaction_fingerprints="structure_fp",
    )
    molecule_fp = get_mol_fp_dict(syngraphs[1], molecular_fingerprint="rdkit")
    assert len(reaction_fp) == 1 and len(molecule_fp) == 4

    mp_syngraphs = [converter(g, "monopartite_reactions") for g in syngraphs]
    mp_reaction_fp = get_reactions_fp_dict(
        mp_syngraphs[1],
        reaction_fingerprints="structure_fp",
    )
    assert len(mp_reaction_fp) == 1


def test_build_similarity_matrix(az_as_dict):
    # Test the workflow for computing the similarity matrix
    mp_syngraphs = [
        translator("az_retro", g, "syngraph", out_data_model="monopartite_reactions")
        for g in az_as_dict
    ]
    reaction_fp1 = get_reactions_fp_dict(
        mp_syngraphs[1],
        reaction_fingerprints="structure_fp",
    )
    reaction_fp2 = get_reactions_fp_dict(
        mp_syngraphs[2],
        reaction_fingerprints="structure_fp",
    )
    matrix = build_similarity_matrix(reaction_fp1, reaction_fp2)
    assert isinstance(matrix, pd.DataFrame)
    assert matrix.shape == (3, 2)


def test_similarity_matrix(az_as_dict):
    # Test similarity with oneself using the similarity matrix workflow
    mp_syngraphs = [
        translator("az_retro", g, "syngraph", out_data_model="monopartite_reactions")
        for g in az_as_dict
    ]

    ged_identical = graph_distance_factory(
        mp_syngraphs[0], mp_syngraphs[0], ged_method="nx_ged_matrix"
    )
    assert math.isclose(ged_identical, 0.0, rel_tol=1e-9)

    ged = graph_distance_factory(
        mp_syngraphs[0], mp_syngraphs[3], ged_method="nx_ged_matrix"
    )
    assert not math.isclose(ged, 0.0, rel_tol=1e-9)


def test_compute_distance_matrix(az_as_dict, ibm1_as_dict):
    mp_syngraphs = [
        translator("az_retro", g, "syngraph", out_data_model="monopartite_reactions")
        for g in az_as_dict
    ]
    m = compute_distance_matrix(mp_syngraphs, ged_method="nx_ged")
    diag = pd.Series([m.iat[n, n] for n in range(len(m))], index=[m.index, m.columns])
    routes = [graph.name for graph in mp_syngraphs]
    proof_diag = pd.Series(np.zeros(len(routes)), index=[m.index, m.columns])
    assert diag.all() == proof_diag.all()

    mp_syngraphs = [
        translator("ibm_retro", r, "syngraph", out_data_model="monopartite_reactions")
        for r in ibm1_as_dict
    ]
    m2 = compute_distance_matrix(mp_syngraphs, ged_method="nx_optimized_ged")
    diag2 = pd.Series(
        [m2.iat[n, n] for n in range(len(m2))], index=[m2.index, m2.columns]
    )
    r2 = [graph.name for graph in mp_syngraphs]
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
    # The minimum value generated by
    # optimize_graph_edit_distance is equal to the value
    # returned by graph_edit_distance
    assert min_g == ged


def test_optimized_ged_workflow(az_as_dict):
    syngraphs = [
        translator("az_retro", g, "syngraph", out_data_model="bipartite")
        for g in az_as_dict
    ]
    ged_bp = graph_distance_factory(
        syngraphs[0], syngraphs[3], ged_method="nx_optimized_ged"
    )
    ged_bp_std = graph_distance_factory(syngraphs[0], syngraphs[3], ged_method="nx_ged")
    assert ged_bp == ged_bp_std
    mp_syngraphs = [
        translator("az_retro", g, "syngraph", out_data_model="monopartite_reactions")
        for g in az_as_dict
    ]
    ged_mp = graph_distance_factory(
        mp_syngraphs[0], mp_syngraphs[3], ged_method="nx_optimized_ged"
    )
    ged_mp_std = graph_distance_factory(
        mp_syngraphs[0], mp_syngraphs[3], ged_method="nx_ged"
    )
    assert ged_mp == ged_mp_std


def test_get_available_ged():
    available_algorithm = get_available_ged_algorithms()
    assert isinstance(available_algorithm, dict)
    assert "nx_ged" in available_algorithm


def test_get_default():
    params = get_ged_default_parameters()
    assert isinstance(params, dict)
    assert "reaction_similarity_name" in params


def test_ged_params():
    params = get_ged_parameters()
    assert isinstance(params, dict)
    assert "reaction_fp_params" in params
