from linchemin.rem.route_metrics import (
    route_metric_calculator,
    distance_function_calculator,
)
from linchemin.cgu.syngraph import (
    MonopartiteReacSynGraph,
    BipartiteSynGraph,
    MonopartiteMolSynGraph,
)
import pytest
from linchemin.cheminfo.models import Molecule

# route from the test file of askcos
d = [
    {
        "output_string": "CC(=O)O.NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1>>CC(=O)NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1",
        "query_id": 0,
    },
    {
        "output_string": "Nc1ccc(N2CCOCC2)cc1.O=C1c2ccccc2C(=O)N1CC1CO1>>O=C1c2ccccc2C(=O)N1CC(O)CNc1ccc(N2CCOCC2)cc1",
        "query_id": 1,
    },
    {
        "output_string": "O=C(n1ccnc1)n1ccnc1.O=C1c2ccccc2C(=O)N1CC(O)CNc1ccc(N2CCOCC2)cc1>>O=C1c2ccccc2C(=O)N1CC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1",
        "query_id": 2,
    },
    {
        "output_string": "O=C1c2ccccc2C(=O)N1CC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1>>NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1",
        "query_id": 3,
    },
]


def test_basic_factory():
    route = MonopartiteReacSynGraph()
    external_data = {}
    with pytest.raises(KeyError) as ke:
        route_metric_calculator("wrong_metric", route, external_data)
    assert "KeyError" in str(ke.type)
    with pytest.raises(TypeError) as ke:
        route_metric_calculator("wrong_metric", [], external_data)
    assert "TypeError" in str(ke.type)


def test_distance_strategy():
    route = BipartiteSynGraph()
    node = Molecule()
    root = Molecule()
    with pytest.raises(KeyError) as ke:
        distance_function_calculator("wrong_function", route, node, root)
    assert "KeyError" in str(ke.type)


def test_reactant_availability():
    syngraph = MonopartiteReacSynGraph(d)
    starting_materials = [
        "CC(=O)O",
        "Nc1ccc(N2CCOCC2)cc1",
        "O=C(n1ccnc1)n1ccnc1",
        "O=C1c2ccccc2C(=O)N1CC1CO1",
    ]
    # best case scenario: all starting materials are available internally
    external_data = {s: "syngenta" for s in starting_materials}
    metric_score = route_metric_calculator(
        "reactant_availability", syngraph, external_data
    )
    assert metric_score == 1.0
    # worst case scenario: none of the starting materials is available
    syngraph = MonopartiteMolSynGraph(d)
    external_data = {s: "none" for s in starting_materials}
    metric_score = route_metric_calculator(
        "reactant_availability", syngraph, external_data
    )
    assert metric_score == 0.0
    # medium case: all starting materials are available at vendors
    external_data = {s: "vendor" for s in starting_materials}
    assert (
        route_metric_calculator("reactant_availability", syngraph, external_data) == 0.5
    )
    # medium-good case: the starting materials for the leaf reactions are available internally; those for the root no
    external_data = {
        "CC(=O)O": "none",
        "O=C(n1ccnc1)n1ccnc1": "vendor",
        "Nc1ccc(N2CCOCC2)cc1": "syngenta",
        "O=C1c2ccccc2C(=O)N1CC1CO1": "syngenta",
    }
    assert (
        route_metric_calculator("reactant_availability", syngraph, external_data)
        == 0.79
    )
    syngraph = BipartiteSynGraph(d)
    # medium-bad case: the starting materials for the leaf reactions are not available internally; those for the root are
    external_data = {
        "CC(=O)O": "syngenta",
        "O=C(n1ccnc1)n1ccnc1": "vendor",
        "Nc1ccc(N2CCOCC2)cc1": "none",
        "O=C1c2ccccc2C(=O)N1CC1CO1": "none",
    }
    assert (
        route_metric_calculator("reactant_availability", syngraph, external_data)
        == 0.21
    )


def test_yield_score():
    syngraph = BipartiteSynGraph(d)
    # best case scenario:all involved reactions have 100% yield
    external_data = {step["output_string"]: 1.0 for step in d}
    assert route_metric_calculator("yield", syngraph, external_data) == 1.0
    # worst case scenario: all involved reactions have 0% yield
    external_data = {step["output_string"]: 0.0 for step in d}
    assert route_metric_calculator("yield", syngraph, external_data) == 0.0
    # medium-good case: reactions close to the root have higher yield than those close to the leaves
    external_data = {
        "CC(=O)O.NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1>>CC(=O)NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1": 0.95,
        "O=C1c2ccccc2C(=O)N1CC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1>>NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1": 0.65,
        "O=C(n1ccnc1)n1ccnc1.O=C1c2ccccc2C(=O)N1CC(O)CNc1ccc(N2CCOCC2)cc1>>O=C1c2ccccc2C(=O)N1CC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1": 0.45,
        "Nc1ccc(N2CCOCC2)cc1.O=C1c2ccccc2C(=O)N1CC1CO1>>O=C1c2ccccc2C(=O)N1CC(O)CNc1ccc(N2CCOCC2)cc1": 0.25,
    }
    yield_score_mg = route_metric_calculator("yield", syngraph, external_data)
    assert yield_score_mg == 0.69
    # medium-bad case: reactions close to the root have lower yield than those close to the leaves
    external_data = {
        "CC(=O)O.NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1>>CC(=O)NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1": 0.25,
        "O=C1c2ccccc2C(=O)N1CC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1>>NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1": 0.45,
        "O=C(n1ccnc1)n1ccnc1.O=C1c2ccccc2C(=O)N1CC(O)CNc1ccc(N2CCOCC2)cc1>>O=C1c2ccccc2C(=O)N1CC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1": 0.65,
        "Nc1ccc(N2CCOCC2)cc1.O=C1c2ccccc2C(=O)N1CC1CO1>>O=C1c2ccccc2C(=O)N1CC(O)CNc1ccc(N2CCOCC2)cc1": 0.95,
    }
    yield_score_mb = route_metric_calculator("yield", syngraph, external_data)
    assert yield_score_mb == 0.46
    assert yield_score_mg > yield_score_mb
