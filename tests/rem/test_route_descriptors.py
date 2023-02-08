import json

import pytest

from linchemin.cgu.convert import converter
from linchemin.cgu.syngraph import MonopartiteReacSynGraph, SynGraph, merge_syngraph
from linchemin.cgu.translate import translator
from linchemin.rem.route_descriptors import (
    descriptor_calculator,
    find_duplicates,
    find_path,
    get_available_descriptors,
    get_nodes_consensus,
    is_subset,
)


def test_unavailable_metrics(ibm1_path):
    """To test that a KeyError is raised if an unavailable metrics is requested."""
    with pytest.raises(Exception) as ke:
        graph = json.loads(open(ibm1_path).read())
        syngraph = translator(
            "ibm_retro", graph[3], "syngraph", out_data_model="bipartite"
        )
        descriptor_calculator(syngraph, "wrong_metrics")
    assert "KeyError" in str(ke.type)


def test_find_path(ibm1_path):
    """To test that find_path function returns the expected path."""
    graph = json.loads(open(ibm1_path).read())
    syngraph = translator("ibm_retro", graph[0], "syngraph", out_data_model="bipartite")
    root = syngraph.get_roots()
    leaves = syngraph.get_leaves()
    path = find_path(syngraph, leaves[0], root[0])
    assert [item.smiles for item in path] == [
        "CCN",
        "CCN.CCOC(=O)CC>>CCNC(=O)CC",
        "CCNC(=O)CC",
    ]


def test_longest_sequence(az_path):
    """To test that the LongestSequence object is returned as expected."""
    graph = json.loads(open(az_path).read())
    syngraph = translator("az_retro", graph[3], "syngraph", out_data_model="bipartite")
    longest_seq = descriptor_calculator(syngraph, "longest_seq")
    assert longest_seq == 3

    mp_syngraph = translator(
        "az_retro", graph[3], "syngraph", out_data_model="monopartite_reactions"
    )
    longest_seq_mp = descriptor_calculator(mp_syngraph, "longest_seq")
    assert longest_seq_mp == longest_seq


def test_metric_selector_nr_steps(az_path):
    """To test that the NrReactionSteps object is returned as expected."""
    graph = json.loads(open(az_path).read())
    syngraph = translator("az_retro", graph[4], "syngraph", out_data_model="bipartite")
    n_steps = descriptor_calculator(syngraph, "nr_steps")
    assert n_steps == 3


def test_metric_selector_paths(az_path):
    """To test that the PathFinder object is returned as expected."""
    graph = json.loads(open(az_path).read())
    syngraph = translator("az_retro", graph[2], "syngraph", out_data_model="bipartite")
    paths = descriptor_calculator(syngraph, "all_paths")
    assert len(paths) == 3


def test_metric_selector_branching_factor(az_path):
    """To test that the AvgBranchingFactor object is returned as expected."""
    graph = json.loads(open(az_path).read())
    syngraphs = [
        translator("az_retro", g, "syngraph", out_data_model="bipartite") for g in graph
    ]

    tree = merge_syngraph(syngraphs)
    avg_branch = descriptor_calculator(tree, "branching_factor")
    assert avg_branch == 1.1428571428571428


def test_nr_branches(az_path):
    graph = json.loads(open(az_path).read())
    syngraphs = translator("az_retro", graph[2], "syngraph", out_data_model="bipartite")
    # The expected number of branches is returned
    nr_b = descriptor_calculator(syngraphs, "nr_branches")
    assert nr_b == 0
    with pytest.raises(Exception) as e:
        graph = None
        descriptor_calculator(graph, "nr_branches")
    assert "NoneInput" in str(e.type)


def test_subset(az_path):
    graph = json.loads(open(az_path).read())
    mp_syngraphs = translator(
        "az_retro", graph[2], "syngraph", out_data_model="monopartite_reactions"
    )
    # A route is subset of another when: (i) the Syngraph dictionary is subset, (ii) the two routes have the same
    # target (iii) the two routes have different leaves
    reaction_leaf = mp_syngraphs.get_leaves()[0]
    subset = MonopartiteReacSynGraph()
    for r, conn in mp_syngraphs.graph.items():
        if r != reaction_leaf:
            subset.add_node((r, list(conn)))
    # Two types of subset are correctly identified
    assert is_subset(subset, mp_syngraphs)
    assert subset.get_roots() == mp_syngraphs.get_roots()
    assert len(subset.graph) < len(mp_syngraphs.graph)
    assert subset.get_leaves() != mp_syngraphs.get_roots()

    root_reaction = mp_syngraphs.get_roots()[0]
    subset2 = MonopartiteReacSynGraph()
    for r, conn in mp_syngraphs.graph.items():
        if r == root_reaction:
            subset2.add_node((r, list(conn)))

    assert is_subset(subset2, mp_syngraphs)
    g2 = translator(
        "az_retro", graph[0], "syngraph", out_data_model="monopartite_reactions"
    )
    # A non-subset is correctly identified as such
    assert not is_subset(g2, mp_syngraphs)


def test_find_duplicates(ibm1_path, az_path):
    graph1 = json.loads(open(ibm1_path).read())
    ibm_routes = [
        translator("ibm_retro", g, "syngraph", out_data_model="bipartite")
        for g in graph1
    ]
    graph2 = json.loads(open(az_path).read())
    az_routes = [
        translator("az_retro", g, "syngraph", out_data_model="bipartite")
        for g in graph2
    ]
    # No duplicates are found
    d = find_duplicates(ibm_routes, az_routes)
    assert d is None

    # A duplicate is found
    ibm_routes.append(az_routes[0])
    d2 = find_duplicates(ibm_routes, az_routes)
    assert d2 is not None
    assert len(d2) == 1

    # Exception raised when the two provided syngraphs have different types (MonopartiteSynGraph and SynGraph)
    with pytest.raises(Exception) as ke:
        ibm_routes_mp = [
            translator(
                "ibm_retro", g, "syngraph", out_data_model="monopartite_reactions"
            )
            for g in graph1
        ]
        find_duplicates(ibm_routes_mp, az_routes)
    assert "Exception" in str(ke.type)


def test_get_node_consensus(az_path):
    graph2 = json.loads(open(az_path).read())
    az_routes = [
        translator("az_retro", g, "syngraph", out_data_model="bipartite")
        for g in graph2
    ]
    # Check consensus for bipartite SynGraphs
    nodes_consensus = get_nodes_consensus(az_routes)
    roots = az_routes[0].get_roots()
    leaves = az_routes[0].get_leaves()
    # total number of diverse nodes
    assert len(nodes_consensus) == 24
    # the root is among the nodes and it is shared among all routes
    assert roots[0] in nodes_consensus
    assert len(nodes_consensus[roots[0]]) == len(az_routes)
    # the leaves are among the nodes, but a single leaf is not shared among all rotues
    assert leaves[0] in nodes_consensus
    assert len(nodes_consensus[leaves[0]]) != len(az_routes)

    # Check consensus for MonopartiteSynGraphs
    az_routes_mp = [
        translator("az_retro", g, "syngraph", out_data_model="monopartite_reactions")
        for g in graph2
    ]
    nodes_consensus_mp = get_nodes_consensus(az_routes_mp)
    reaction_roots = az_routes_mp[0].get_roots()
    reaction_leaves = az_routes_mp[0].get_roots()
    assert reaction_roots[0] in nodes_consensus_mp
    assert len(nodes_consensus_mp[reaction_leaves[0]]) != len(az_routes_mp)


def test_get_available_routes():
    assert (
        type(get_available_descriptors()) == dict
        and "nr_steps" in get_available_descriptors()
    )


def test_convergence(az_path):
    graph = json.loads(open(az_path).read())
    az_routes_mp = translator(
        "az_retro", graph[-1], "syngraph", out_data_model="monopartite_reactions"
    )
    lls = descriptor_calculator(az_routes_mp, "longest_seq")
    n_steps = descriptor_calculator(az_routes_mp, "nr_steps")
    convergence = descriptor_calculator(az_routes_mp, "convergence")
    assert convergence == lls / n_steps


def test_cdscores(az_path):
    graph = json.loads(open(az_path).read())
    az_routes_mp = translator(
        "az_retro", graph[0], "syngraph", out_data_model="monopartite_reactions"
    )
    cds = descriptor_calculator(az_routes_mp, "cdscore")
    assert 0 < cds < 1

    with pytest.raises(TypeError) as te:
        route = translator(
            "az_retro", graph[0], "iron", out_data_model="monopartite_reactions"
        )
        descriptor_calculator(route, "cdscore")
    assert "TypeError" in str(te.type)


def test_branchedness(ibm2_path):
    f = json.loads(open(ibm2_path).read())
    routes = [
        translator("ibm_retro", g, "syngraph", out_data_model="monopartite_reactions")
        for g in f[:3]
    ]
    assert descriptor_calculator(routes[0], "branchedness") == 0.0
    assert descriptor_calculator(routes[2], "branchedness") == 0.5
