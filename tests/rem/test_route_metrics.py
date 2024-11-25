import math

import pytest

from linchemin.cgu.syngraph import (
    BipartiteSynGraph,
    MonopartiteMolSynGraph,
    MonopartiteReacSynGraph,
)
from linchemin.cheminfo.models import Molecule
from linchemin.rem.route_metrics import (
    NotFullyMappedRouteError,
    RenewableCarbonMetric,
    StartingMaterialsAmount,
    YieldMetric,
    distance_function_calculator,
    route_metric_calculator,
    MissingDataError,
)


@pytest.fixture
def route():
    return [
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


def test_reactant_availability(route):
    syngraph = MonopartiteReacSynGraph(route)
    starting_materials = [
        "CC(=O)O",
        "Nc1ccc(N2CCOCC2)cc1",
        "O=C(n1ccnc1)n1ccnc1",
        "O=C1c2ccccc2C(=O)N1CC1CO1",
    ]
    categories = [
        {"name": "best", "criterion": "syngenta", "score": 1.0},
        {"name": "medium", "criterion": "vendor", "score": 0.5},
        {"name": "worst", "criterion": "none", "score": 0.0},
    ]
    # best case scenario: all starting materials are available internally
    external_data = {s: "syngenta" for s in starting_materials}
    output = route_metric_calculator(
        "reactant_availability", syngraph, external_data, categories
    )
    assert math.isclose(output.metric_value, 1.0, rel_tol=1e-9)
    assert output.raw_data
    assert "distance_function" in output.raw_data
    # worst case scenario: none of the starting materials is available
    syngraph = MonopartiteMolSynGraph(route)
    external_data = {s: "none" for s in starting_materials}
    output = route_metric_calculator(
        "reactant_availability", syngraph, external_data, categories
    )
    assert math.isclose(output.metric_value, 0.0, rel_tol=1e-9)
    # medium case: all starting materials are available at vendors
    external_data = {s: "vendor" for s in starting_materials}
    assert math.isclose(
        route_metric_calculator(
            "reactant_availability", syngraph, external_data, categories
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
            "reactant_availability", syngraph, external_data, categories
        ).metric_value,
        0.79,
        rel_tol=1e-9,
    )

    syngraph = BipartiteSynGraph(route)
    # medium-bad case: the starting materials for the leaf reactions are not available internally; those for the root are
    external_data = {
        "CC(=O)O": "syngenta",
        "O=C(n1ccnc1)n1ccnc1": "vendor",
        "Nc1ccc(N2CCOCC2)cc1": "none",
        "O=C1c2ccccc2C(=O)N1CC1CO1": "none",
    }
    assert math.isclose(
        route_metric_calculator(
            "reactant_availability", syngraph, external_data, categories
        ).metric_value,
        0.21,
        rel_tol=1e-9,
    )


def test_reactant_availability_no_cat(route):
    syngraph = MonopartiteReacSynGraph(route)
    starting_materials = [
        "CC(=O)O",
        "Nc1ccc(N2CCOCC2)cc1",
        "O=C(n1ccnc1)n1ccnc1",
        "O=C1c2ccccc2C(=O)N1CC1CO1",
    ]
    external_data = {s: "syngenta" for s in starting_materials}
    with pytest.raises(MissingDataError):
        route_metric_calculator("reactant_availability", syngraph, external_data)


def test_yield_score(route):
    syngraph = BipartiteSynGraph(route)
    # best case scenario:all involved reactions have 100% yield
    external_data = {step["output_string"]: 1.0 for step in route}
    metric = YieldMetric()
    assert math.isclose(
        metric.compute_metric(syngraph, external_data).metric_value,
        1.0,
        rel_tol=1e-9,
    )
    # worst case scenario: all involved reactions have 0% yield
    external_data = {step["output_string"]: 0.0 for step in route}
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


def test_starting_materials_amount(route):
    syngraph = MonopartiteReacSynGraph(route)
    external_info = {
        "target_amount": 319.15320615199994,  # 1 mol
        "yield": {item["output_string"]: 0.5 for item in route},
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


def test_renewable_carbon(route):
    syngraph = MonopartiteReacSynGraph(route)
    starting_materials = [
        "CC(=O)O",
        "Nc1ccc(N2CCOCC2)cc1",
        "O=C(n1ccnc1)n1ccnc1",
    ]
    external_data = {"building_blocks": starting_materials}
    metric = RenewableCarbonMetric()

    # a fully mapped route is needed
    with pytest.raises(NotFullyMappedRouteError):
        metric.compute_metric(syngraph, external_data=external_data)
    mapped_route = [
        {
            "output_string": "O[C:2]([CH3:1])=[O:3].[NH2:4][CH2:5][CH:6]1[CH2:7][N:8]([c:9]2[cH:10][cH:11][c:12]([N:13]3[CH2:14][CH2:15][O:16][CH2:17][CH2:18]3)[cH:19][cH:20]2)[C:21](=[O:22])[O:23]1>>[CH3:1][C:2](=[O:3])[NH:4][CH2:5][CH:6]1[CH2:7][N:8]([c:9]2[cH:10][cH:11][c:12]([N:13]3[CH2:14][CH2:15][O:16][CH2:17][CH2:18]3)[cH:19][cH:20]2)[C:21](=[O:22])[O:23]1",
            "query_id": "0",
        },
        {
            "output_string": "[NH2:1][c:2]1[cH:3][cH:4][c:5]([N:6]2[CH2:7][CH2:8][O:9][CH2:10][CH2:11]2)[cH:12][cH:13]1.[O:14]=[C:15]1[c:16]2[cH:17][cH:18][cH:19][cH:20][c:21]2[C:22](=[O:23])[N:24]1[CH2:25][CH:26]1[CH2:27][O:28]1>>[NH:1]([c:2]1[cH:3][cH:4][c:5]([N:6]2[CH2:7][CH2:8][O:9][CH2:10][CH2:11]2)[cH:12][cH:13]1)[CH2:27][CH:26]([CH2:25][N:24]1[C:15](=[O:14])[c:16]2[cH:17][cH:18][cH:19][cH:20][c:21]2[C:22]1=[O:23])[OH:28]",
            "query_id": "1",
        },
        {
            "output_string": "C1=CN([C:2](N2C=CN=C2)=[O:1])C=N1.[O:3]=[C:4]1[c:5]2[cH:6][cH:7][cH:8][cH:9][c:10]2[C:11](=[O:12])[N:13]1[CH2:14][CH:15]([OH:16])[CH2:17][NH:18][c:19]1[cH:20][cH:21][c:22]([N:23]2[CH2:24][CH2:25][O:26][CH2:27][CH2:28]2)[cH:29][cH:30]1>>[O:1]=[C:2]1[O:16][CH:15]([CH2:14][N:13]2[C:4](=[O:3])[c:5]3[cH:6][cH:7][cH:8][cH:9][c:10]3[C:11]2=[O:12])[CH2:17][N:18]1[c:19]1[cH:20][cH:21][c:22]([N:23]2[CH2:24][CH2:25][O:26][CH2:27][CH2:28]2)[cH:29][cH:30]1",
            "query_id": "2",
        },
        {
            "output_string": "O=C1c2ccccc2C(=O)[N:1]1[CH2:2][CH:3]1[CH2:4][N:5]([c:6]2[cH:7][cH:8][c:9]([N:10]3[CH2:11][CH2:12][O:13][CH2:14][CH2:15]3)[cH:16][cH:17]2)[C:18](=[O:19])[O:20]1>>[NH2:1][CH2:2][CH:3]1[CH2:4][N:5]([c:6]2[cH:7][cH:8][c:9]([N:10]3[CH2:11][CH2:12][O:13][CH2:14][CH2:15]3)[cH:16][cH:17]2)[C:18](=[O:19])[O:20]1",
            "query_id": "3",
        },
    ]
    syngraph = MonopartiteReacSynGraph(mapped_route)
    out = metric.compute_metric(syngraph, external_data=external_data)
    assert out.metric_value == round(13.0 / 16.0, 2)
    assert out.additional_info

    # none of the provided building blocks appears in the route
    external_data = {"building_blocks": ["CCC(O)=O"]}
    out = metric.compute_metric(syngraph, external_data=external_data)
    assert math.isclose(out.metric_value, 0.0, rel_tol=1e-9)
