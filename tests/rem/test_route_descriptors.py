import json
from unittest.mock import patch

import pytest

from linchemin.cgu.syngraph import MonopartiteReacSynGraph
from linchemin.cgu.translate import translator
from linchemin.rem.route_descriptors import (
    Branchedness,
    CDScore,
    Convergence,
    DescriptorError,
    DescriptorsCalculatorFactory,
    LongestSequence,
    NrReactionSteps,
    RouteDescriptor,
    SimplifiedAtomEffectiveness,
    UnavailableDescriptor,
    WrongGraphType,
    descriptor_calculator,
    find_duplicates,
    get_available_descriptors,
    get_configuration,
    get_nodes_consensus,
    is_subset,
    validate_input_graph,
)


@pytest.fixture
def branched_syngraph():
    reactions = [
        "Cc1cccc(C)c1NCC(=O)Cl.O=C(O)C1CCS(=O)(=O)CC1>>Cc1cccc(C)c1N(CC(=O)Cl)C(=O)C1CCS(=O)(=O)CC1",
        "Cc1cccc(C)c1N(CC(=O)Cl)C(=O)C1CCS(=O)(=O)CC1.Nc1ccc(-c2ncon2)cc1>>Cc1cccc(C)c1N(CC(=O)Nc1ccc(-c2ncon2)cc1)C(=O)C1CCS(=O)(=O)CC1",
        "Cc1cccc(C)c1NCC(=O)O>>Cc1cccc(C)c1NCC(=O)Cl",
        "Cc1cccc(C)c1NCC(=O)Cl.Nc1ccc(-c2ncon2)cc1>>Cc1cccc(C)c1NCC(=O)Nc1ccc(-c2ncon2)cc1",
        "Cc1cccc(C)c1NCC(=O)Cl.O=C(Cl)C1CCS(=O)(=O)CC1>>Cc1cccc(C)c1N(CC(=O)Cl)C(=O)C1CCS(=O)(=O)CC1",
        "Cc1cccc(C)c1NCC(=O)Nc1ccc(-c2ncon2)cc1.O=C(O)C1CCS(=O)(=O)CC1>>Cc1cccc(C)c1N(CC(=O)Nc1ccc(-c2ncon2)cc1)C(=O)C1CCS(=O)(=O)CC1",
    ]
    return MonopartiteReacSynGraph(
        [{"query_id": n, "output_string": s} for n, s in enumerate(reactions)]
    )


@DescriptorsCalculatorFactory.register_descriptors("mock_descriptor")
class MockDescriptor(RouteDescriptor):
    info = "A mocked descriptor"

    def compute_descriptor(self, graph: MonopartiteReacSynGraph) -> int:
        pass


def test_register_descriptors():
    assert "mock_descriptor" in DescriptorsCalculatorFactory._registered_descriptors
    assert (
        DescriptorsCalculatorFactory._registered_descriptors["mock_descriptor"]
        is MockDescriptor
    )


# Test retrieval of a registered descriptor
def test_get_descriptor():
    name = "mock_descriptor"
    descriptor_instance = DescriptorsCalculatorFactory.get_descriptor(name)
    assert isinstance(descriptor_instance, RouteDescriptor)


# Test retrieval of a non-existent descriptor
def test_get_unavailable_descriptor():
    with pytest.raises(UnavailableDescriptor):
        DescriptorsCalculatorFactory.get_descriptor("non_existent_descriptor")


# Test listing of descriptors
def test_list_route_descriptors():
    descriptors_list = DescriptorsCalculatorFactory.list_route_descriptors()
    assert "mock_descriptor" in descriptors_list


# Test configuration retrieval
def test_get_descriptor_configuration():
    name = "mock_descriptor"
    mock_config = {"config_key": "config_value"}

    with patch.object(RouteDescriptor, "get_configuration", return_value=mock_config):
        config = DescriptorsCalculatorFactory().get_descriptor_configuration(name)
        assert config == mock_config


def test_validate_input(bp_syngraph_instance, mpr_syngraph_instance, iron_w_smiles):
    g = validate_input_graph(mpr_syngraph_instance)
    assert g.graph == mpr_syngraph_instance.graph
    g2 = validate_input_graph(bp_syngraph_instance)
    assert isinstance(g2, MonopartiteReacSynGraph)
    with pytest.raises(WrongGraphType):
        validate_input_graph(iron_w_smiles)


def test_longest_sequence(mpr_syngraph_instance):
    """To test that the LongestSequence object is returned as expected."""
    longest_sequence = LongestSequence()
    lls = longest_sequence.compute_descriptor(mpr_syngraph_instance)
    assert lls == 3

    r = [
        {
            "query_id": 0,
            "output_string": "CC(=O)O.NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1>>CC(=O)NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1",
        }
    ]
    single_reaction_syngraph = MonopartiteReacSynGraph(r)
    assert descriptor_calculator(single_reaction_syngraph, "longest_seq") == 1
    assert {"title": "Longest Linear Sequence"}.items() <= get_configuration(
        "longest_seq"
    ).items()


def test_nr_steps(mpr_syngraph_instance):
    """To test that the NrReactionSteps object is returned as expected."""
    nr_steps = NrReactionSteps()
    n_steps = nr_steps.compute_descriptor(mpr_syngraph_instance)
    assert n_steps == 3
    assert {
        "title": "Total N of Steps",
        "order": 10,
    }.items() <= NrReactionSteps().get_configuration().items()


def test_branching_factor(branched_syngraph):
    """To test that the AvgBranchingFactor object is returned as expected."""
    avg_branch = descriptor_calculator(branched_syngraph, "branching_factor")
    assert avg_branch > 0
    assert {"title": "Avg Branching Factor"}.items() <= get_configuration(
        "branching_factor"
    ).items()


def test_nr_branches(mpr_syngraph_instance, branched_syngraph):
    # The expected number of branches is returned
    nr_b = descriptor_calculator(mpr_syngraph_instance, "nr_branches")
    assert nr_b == 0
    with pytest.raises(DescriptorError) as e:
        graph = None
        descriptor_calculator(graph, "nr_branches")
    assert "InvalidInput" in str(e.type)
    assert {"title": "N of Branches"}.items() <= get_configuration(
        "nr_branches"
    ).items()
    nr_b = descriptor_calculator(branched_syngraph, "nr_branches")
    assert nr_b == 2


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


def test_find_duplicates(
    bp_syngraph_instance, mpr_syngraph_instance, branched_syngraph
):
    l1 = [mpr_syngraph_instance]
    l2 = [branched_syngraph]
    # No duplicates are found
    d = find_duplicates(l1, l2)
    assert d is None

    # A duplicate is found
    l2.append(mpr_syngraph_instance)
    d2 = find_duplicates(l1, l2)
    assert d2 is not None
    assert len(d2) == 1

    # Exception raised when the two provided syngraphs have different types (MonopartiteSynGraph and BipartiteSynGraph)
    with pytest.raises(DescriptorError) as ke:
        find_duplicates(l2, [bp_syngraph_instance])
    assert "MismatchingGraphType" in str(ke.type)


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
    # the root is among the nodes, and it is shared among all routes
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


def test_get_available_route_descriptors():
    assert (
        isinstance(get_available_descriptors(), dict)
        and "nr_steps" in get_available_descriptors()
    )


def test_convergence(branched_syngraph):
    convergence = Convergence()
    convergence_value = convergence.compute_descriptor(branched_syngraph)
    assert convergence_value == round(3.0 / 6.0, 2)
    assert {"title": "Convergence"}.items() <= get_configuration("convergence").items()


def test_cdscores(branched_syngraph):
    cdscore = CDScore()
    cds = cdscore.compute_descriptor(branched_syngraph)
    assert 0 < cds < 1

    assert {"title": "Convergent Disconnection Score"}.items() <= get_configuration(
        "cdscore"
    ).items()


def test_branchedness(mpr_syngraph_instance, branched_syngraph):
    branchedness = Branchedness()
    assert branchedness.compute_descriptor(mpr_syngraph_instance) == 0.0
    assert branchedness.compute_descriptor(branched_syngraph) == 2.0
    assert {"title": "Branchedness"}.items() <= get_configuration(
        "branchedness"
    ).items()


def test_simplified_atom_effectiveness(mpr_syngraph_instance):
    sae = SimplifiedAtomEffectiveness()
    ae = sae.compute_descriptor(mpr_syngraph_instance)
    assert ae == round(30.0 / 50.0, 2)
    assert {"title": "Simplified Atom Effectiveness"}.items() <= get_configuration(
        "simplified_atom_effectiveness"
    ).items()
