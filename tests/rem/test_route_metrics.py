import math

import pytest

from linchemin.cgu.syngraph import (
    BipartiteSynGraph,
    MonopartiteMolSynGraph,
    MonopartiteReacSynGraph,
)
from linchemin.cheminfo.models import Molecule
from linchemin.rem.route_metrics import (
    StartingMaterialsAmount,
    YieldMetric,
    distance_function_calculator,
    route_metric_calculator,
)

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
    # if an unavailable metric is chosen, a KeyError is raised
    with pytest.raises(KeyError) as ke:
        route_metric_calculator("wrong_metric", route, external_data)
    assert "KeyError" in str(ke.type)
    # if the provided route is of the wrong type, a TypeError is raised
    with pytest.raises(TypeError) as ke:
        route_metric_calculator("reactant_availability", [], external_data)
    assert "TypeError" in str(ke.type)


def test_distance_strategy():
    route = BipartiteSynGraph()
    node = Molecule()
    root = Molecule()
    # if an unavailable distance function is chosen, a KeyError is raised
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
    output = route_metric_calculator("reactant_availability", syngraph, external_data)
    assert math.isclose(output.metric_value, 1.0, rel_tol=1e-9)
    assert output.raw_data
    assert "distance_function" in output.raw_data
    # worst case scenario: none of the starting materials is available
    syngraph = MonopartiteMolSynGraph(d)
    external_data = {s: "none" for s in starting_materials}
    output = route_metric_calculator("reactant_availability", syngraph, external_data)
    assert math.isclose(output.metric_value, 0.0, rel_tol=1e-9)
    # medium case: all starting materials are available at vendors
    external_data = {s: "vendor" for s in starting_materials}
    assert math.isclose(
        route_metric_calculator(
            "reactant_availability", syngraph, external_data
        ).metric_value,
        0.5,
        rel_tol=1e-9,
    )

    # medium-good case: the starting materials for the leaf
    # reactions are available internally; those for the root no
    external_data = {
        "CC(=O)O": "none",
        "O=C(n1ccnc1)n1ccnc1": "vendor",
        "Nc1ccc(N2CCOCC2)cc1": "syngenta",
        "O=C1c2ccccc2C(=O)N1CC1CO1": "syngenta",
    }
    assert math.isclose(
        route_metric_calculator(
            "reactant_availability", syngraph, external_data
        ).metric_value,
        0.79,
        rel_tol=1e-9,
    )

    syngraph = BipartiteSynGraph(d)
    # medium-bad case: the starting materials for the leaf reactions are not available internally; those for the root are
    external_data = {
        "CC(=O)O": "syngenta",
        "O=C(n1ccnc1)n1ccnc1": "vendor",
        "Nc1ccc(N2CCOCC2)cc1": "none",
        "O=C1c2ccccc2C(=O)N1CC1CO1": "none",
    }
    assert math.isclose(
        route_metric_calculator(
            "reactant_availability", syngraph, external_data
        ).metric_value,
        0.21,
        rel_tol=1e-9,
    )


def test_yield_score():
    syngraph = BipartiteSynGraph(d)
    # best case scenario:all involved reactions have 100% yield
    external_data = {step["output_string"]: 1.0 for step in d}
    metric = YieldMetric()
    assert math.isclose(
        metric.compute_metric(syngraph, external_data).metric_value,
        1.0,
        rel_tol=1e-9,
    )
    # worst case scenario: all involved reactions have 0% yield
    external_data = {step["output_string"]: 0.0 for step in d}
    assert math.isclose(
        metric.compute_metric(syngraph, external_data).metric_value,
        0.0,
        rel_tol=1e-9,
    )
    # medium-good case: reactions close to the root have higher yield than those close to the leaves
    external_data = {
        "CC(=O)O.NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1>>CC(=O)NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1": 0.95,
        "O=C1c2ccccc2C(=O)N1CC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1>>NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1": 0.65,
        "O=C(n1ccnc1)n1ccnc1.O=C1c2ccccc2C(=O)N1CC(O)CNc1ccc(N2CCOCC2)cc1>>O=C1c2ccccc2C(=O)N1CC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1": 0.45,
        "Nc1ccc(N2CCOCC2)cc1.O=C1c2ccccc2C(=O)N1CC1CO1>>O=C1c2ccccc2C(=O)N1CC(O)CNc1ccc(N2CCOCC2)cc1": 0.25,
    }
    output_mg = metric.compute_metric(syngraph, external_data)
    assert output_mg.raw_data
    assert math.isclose(
        output_mg.metric_value,
        0.71,
        rel_tol=1e-9,
    )
    # medium-bad case: reactions close to the root have lower yield than those close to the leaves
    external_data = {
        "CC(=O)O.NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1>>CC(=O)NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1": 0.25,
        "O=C1c2ccccc2C(=O)N1CC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1>>NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1": 0.45,
        "O=C(n1ccnc1)n1ccnc1.O=C1c2ccccc2C(=O)N1CC(O)CNc1ccc(N2CCOCC2)cc1>>O=C1c2ccccc2C(=O)N1CC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1": 0.65,
        "Nc1ccc(N2CCOCC2)cc1.O=C1c2ccccc2C(=O)N1CC1CO1>>O=C1c2ccccc2C(=O)N1CC(O)CNc1ccc(N2CCOCC2)cc1": 0.95,
    }
    output_mb = metric.compute_metric(syngraph, external_data)
    assert math.isclose(
        output_mb.metric_value,
        0.44,
        rel_tol=1e-9,
    )
    assert output_mg.metric_value > output_mb.metric_value


def test_starting_materials_amount():
    syngraph = MonopartiteReacSynGraph(d)
    external_info = {
        "target_amount": 319.15320615199994,  # 1 mol
        "yield": {item["output_string"]: 0.5 for item in d},
    }
    sm_amount = StartingMaterialsAmount()
    out = sm_amount.compute_metric(syngraph, external_info)
    assert out.raw_data
    assert out.raw_data["quantities"]["intermediates"] == {
        "NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1": {"grams": 554.285, "moles": 2.0},
        "O=C1c2ccccc2C(=O)N1CC(O)CNc1ccc(N2CCOCC2)cc1": {
            "grams": 3049.351,
            "moles": 8.0,
        },
        "O=C1c2ccccc2C(=O)N1CC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1": {
            "grams": 1628.592,
            "moles": 4.0,
        },
    }
    assert out.raw_data["quantities"]["starting_materials"] == {
        "CC(=O)O": {"grams": 120.042, "moles": 2.0},
        "Nc1ccc(N2CCOCC2)cc1": {"grams": 2849.77, "moles": 16.0},
        "O=C(n1ccnc1)n1ccnc1": {"grams": 1296.433, "moles": 8.0},
        "O=C1c2ccccc2C(=O)N1CC1CO1": {"grams": 3248.932, "moles": 16.0},
    }
    assert out.raw_data["target_amount"] == {"grams": 319.15320615199994, "moles": 1.0}
