import unittest

import pytest

from linchemin.cgu.graph_transformations.data_model_converters import (
    Bipartite,
    DataModelCatalog,
    MonopartiteMolecules,
    MonopartiteReactions,
)
from linchemin.cgu.graph_transformations.exceptions import UnavailableDataModel
from linchemin.cgu.iron import Direction, Edge, Iron, Node
from linchemin.cgu.syngraph import (
    BipartiteSynGraph,
    MonopartiteMolSynGraph,
    MonopartiteReacSynGraph,
)


def test_iron_bp_syngraph(iron_w_smiles):
    """To test the Iron -> SynGraph transformation (mainly tested in the test_syngraph)."""
    converter = Bipartite()
    bp_syngraph = converter.iron_to_syngraph(iron_w_smiles)
    assert isinstance(bp_syngraph, BipartiteSynGraph)
    assert len(bp_syngraph.graph) == 4
    new_iron = converter.syngraph_to_iron(bp_syngraph)
    assert new_iron.i_node_number() == 4
    assert new_iron.i_edge_number() == 3


def test_failing_iron_bp_syngraph():
    """To test that a warning is raised if an empty route is encountered"""
    # iron to syngraph
    iron = None
    converter = Bipartite()
    with unittest.TestCase().assertLogs(
        "linchemin.cgu.graph_transformations.data_model_converters", level="WARNING"
    ) as cm:
        syngraph = converter.iron_to_syngraph(iron)
    assert syngraph is None
    unittest.TestCase().assertEqual(len(cm.records), 1)
    unittest.TestCase().assertIn(
        "While converting from Iron to bipartite", cm.records[0].getMessage()
    )

    # syngraph to iron
    with unittest.TestCase().assertLogs(
        "linchemin.cgu.graph_transformations.data_model_converters", level="WARNING"
    ) as cm:
        new_iron = converter.syngraph_to_iron(syngraph)
    assert new_iron is None
    unittest.TestCase().assertEqual(len(cm.records), 1)
    unittest.TestCase().assertIn(
        "While converting from a bipartite SynGraph to Iron", cm.records[0].getMessage()
    )


def test_iron_to_mpm_syngraph(iron_w_smiles):
    """To test the Iron -> SynGraph transformation (mainly tested in the test_syngraph)."""
    converter = MonopartiteMolecules()
    mpm_syngraph = converter.iron_to_syngraph(iron_w_smiles)
    assert isinstance(mpm_syngraph, MonopartiteMolSynGraph)
    assert len(mpm_syngraph.graph) == 3
    new_iron = converter.syngraph_to_iron(mpm_syngraph)
    assert new_iron.i_node_number() == 3
    assert new_iron.i_edge_number() == 2


def test_failing_iron_mpm_syngraph():
    """To test that a warning is raised if an empty route is encountered"""
    iron = None
    converter = MonopartiteMolecules()
    with unittest.TestCase().assertLogs(
        "linchemin.cgu.graph_transformations.data_model_converters", level="WARNING"
    ) as cm:
        syngraph = converter.iron_to_syngraph(iron)
    assert syngraph is None
    unittest.TestCase().assertEqual(len(cm.records), 1)
    unittest.TestCase().assertIn(
        "While converting from Iron to a monopartite-molecules",
        cm.records[0].getMessage(),
    )

    # syngraph to iron
    with unittest.TestCase().assertLogs(
        "linchemin.cgu.graph_transformations.data_model_converters", level="WARNING"
    ) as cm:
        new_iron = converter.syngraph_to_iron(syngraph)
    assert new_iron is None
    unittest.TestCase().assertEqual(len(cm.records), 1)
    unittest.TestCase().assertIn(
        "While converting from a monopartite-molecules SynGraph to Iron",
        cm.records[0].getMessage(),
    )


def test_iron_to_mpr_syngraph(iron_w_smiles):
    """To test the Iron -> SynGraph transformation (mainly tested in the test_syngraph)."""
    converter = MonopartiteReactions()
    mpr_syngraph = converter.iron_to_syngraph(iron_w_smiles)
    assert isinstance(mpr_syngraph, MonopartiteReacSynGraph)
    assert len(mpr_syngraph.graph) == 1
    new_iron = converter.syngraph_to_iron(mpr_syngraph)
    assert new_iron.i_node_number() == 1
    assert new_iron.i_edge_number() == 0


def test_failing_iron_mpr_syngraph():
    """To test that a warning is raised if an empty route is encountered"""
    iron = None
    converter = MonopartiteReactions()
    with unittest.TestCase().assertLogs(
        "linchemin.cgu.graph_transformations.data_model_converters", level="WARNING"
    ) as cm:
        syngraph = converter.iron_to_syngraph(iron)
    assert syngraph is None
    unittest.TestCase().assertEqual(len(cm.records), 1)
    unittest.TestCase().assertIn(
        "While converting from Iron to monopartite-reactions",
        cm.records[0].getMessage(),
    )

    # syngraph to iron
    with unittest.TestCase().assertLogs(
        "linchemin.cgu.graph_transformations.data_model_converters", level="WARNING"
    ) as cm:
        new_iron = converter.syngraph_to_iron(syngraph)
    assert new_iron is None
    unittest.TestCase().assertEqual(len(cm.records), 1)
    unittest.TestCase().assertIn(
        "While converting from a monopartite-reactions SynGraph to Iron",
        cm.records[0].getMessage(),
    )


def test_unavailable_conversion():
    """To test that an error is raised when a data model not available is selected"""
    catalog = DataModelCatalog
    with pytest.raises(UnavailableDataModel):
        catalog.get_data_model("unknown_data_model")


def test_list_data_models():
    catalog = DataModelCatalog
    d = catalog.list_data_models()
    assert "bipartite" in d
    assert "monopartite_reactions" in d
    assert "monopartite_molecules" in d
