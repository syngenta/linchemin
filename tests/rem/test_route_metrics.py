import math

import pytest
from linchemin.cheminfo.models import ChemicalEquation, Molecule
from linchemin.cgu.syngraph import (
    BipartiteSynGraph,
    MonopartiteMolSynGraph,
    MonopartiteReacSynGraph,
)
from linchemin.rem.route_metrics import (
    NotFullyMappedRouteError,
    ReactantAvailability,
    RouteMetric,
    RenewableCarbonMetric,
    StartingMaterialsAmount,
    YieldMetric,
    route_metric_calculator,
    MissingDataError,
    RouteComponentType,
    RouteComponents,
    InvalidComponentTypeError,
    UnavailableMetricError,
    UnavailableMoleculeFormat,
)

from unittest.mock import patch, Mock


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


def test_route_component_type_enum():
    assert RouteComponentType.MOLECULES.value == "molecules"
    assert RouteComponentType.CHEMICAL_EQUATIONS.value == "chemical_equations"


def test_route_components():
    components = RouteComponents(component_type=RouteComponentType.MOLECULES)
    components.structural_format = "smiles"
    components.uid_structure_map = {"m1": "smiles1", "m2": "smiles1"}
    assert components

    components = RouteComponents(component_type="chemical_equations")
    components.structural_format = "smiles"
    components.uid_structure_map = {"ce1": "smiles1", "ce": "smiles1"}
    assert components


def test_route_components_init_with_invalid_string():
    with pytest.raises(InvalidComponentTypeError, match="Invalid component_type"):
        RouteComponents("reactions")


def test_route_components_init_with_invalid_type():
    with pytest.raises(
        InvalidComponentTypeError,
        match="component_type must be RouteComponentType or str",
    ):
        RouteComponents(123)


@pytest.mark.parametrize("fmt", ["smarts", "mol_blockV3K", "mol_blockV2K"])
def test_validate_format_valid(fmt):
    RouteComponents.validate_format(fmt)


def test_validate_format_invalid():
    with pytest.raises(UnavailableMoleculeFormat):
        RouteComponents.validate_format("invalid_format")


@pytest.mark.parametrize("fmt", ["smarts", "mol_blockV3K", "mol_blockV2K"])
def test_get_mol_to_string_function(fmt):
    func = RouteComponents.get_mol_to_string_function(fmt)
    assert callable(func)
    assert func == RouteComponents.molecule_func_map[fmt]


def test_basic_factory():
    route = MonopartiteReacSynGraph()
    external_data = {}
    # if an unavailable metric is chosen, a KeyError is raised
    with pytest.raises(UnavailableMetricError) as ke:
        route_metric_calculator("wrong_metric", route, external_data)
    # if the provided route is of the wrong type, a TypeError is raised
    with pytest.raises(TypeError) as ke:
        route_metric_calculator("reactant_availability", [], external_data)
    assert "TypeError" in str(ke.type)


@pytest.fixture
def mock_route():
    return Mock(spec=BipartiteSynGraph)


@pytest.fixture
def reactant_availability(mock_route):
    return ReactantAvailability(mock_route)


def test_init(mock_route):
    metric = ReactantAvailability(mock_route)
    assert metric.route == mock_route
    assert metric.name == "reactant_availability"


def test_check_route_format_bipartite():
    bipartite_route = Mock(spec=BipartiteSynGraph)
    assert ReactantAvailability._check_route_format(bipartite_route) == bipartite_route


@patch("linchemin.rem.route_metrics.converter")
def test_check_route_format_conversion(mock_converter):
    monopartite_route = Mock()
    mock_converter.return_value = Mock(spec=BipartiteSynGraph)
    result = ReactantAvailability._check_route_format(monopartite_route)
    mock_converter.assert_called_once_with(monopartite_route, "bipartite")
    assert isinstance(result, BipartiteSynGraph)


@patch.object(RouteMetric, "_get_molecules_map")
def test_get_route_components_for_metric(mock_get_molecules_map, reactant_availability):
    mock_leaves = [Mock(spec=Molecule) for _ in range(3)]
    reactant_availability.route.get_leaves.return_value = mock_leaves
    uid_map = {"uid1": "SMILES1", "uid2": "SMILES2", "uid3": "SMILES3"}
    mock_get_molecules_map.return_value = uid_map

    result = reactant_availability.get_route_components_for_metric("smiles")

    assert isinstance(result, RouteComponents)
    assert result.component_type == RouteComponentType.MOLECULES
    assert result.structural_format == "smiles"
    assert result.uid_structure_map == uid_map
    mock_get_molecules_map.assert_called_once_with(
        molecule_list=mock_leaves, structural_format="smiles"
    )


def test_reactant_availability(route):
    syngraph = MonopartiteReacSynGraph(route)
    ra = ReactantAvailability(syngraph)

    leaves = syngraph.get_molecule_leaves()
    starting_materials = [l.uid for l in leaves]
    categories = [
        {"name": "best", "criterion": "syngenta", "score": 1.0},
        {"name": "medium", "criterion": "vendor", "score": 0.5},
        {"name": "worst", "criterion": "none", "score": 0.0},
    ]
    # best case scenario: all starting materials are available internally
    external_data = {s: "syngenta" for s in starting_materials}
    output = ra.compute_metric(external_data, categories)
    assert math.isclose(output.metric_value, 1.0, rel_tol=1e-9)
    assert output.raw_data
    assert "distance_function" in output.raw_data

    # worst case scenario: none of the starting materials is available
    external_data = {s: "none" for s in starting_materials}
    output = ra.compute_metric(external_data, categories)
    assert math.isclose(output.metric_value, 0.0, rel_tol=1e-9)

    # medium case: all starting materials are available at vendors
    external_data = {s: "vendor" for s in starting_materials}
    output = ra.compute_metric(external_data, categories)
    assert math.isclose(
        output.metric_value,
        0.5,
        rel_tol=1e-9,
    )

    # medium-good case: the starting materials for the leaf
    external_data = {
        m.uid: "syngenta"
        for m in leaves
        if m.smiles in ["Nc1ccc(N2CCOCC2)cc1", "O=C1c2ccccc2C(=O)N1CC1CO1"]
    }
    external_data.update(
        {m.uid: "vendor" for m in leaves if m.smiles == "O=C(n1ccnc1)n1ccnc1"}
    )
    external_data.update({m.uid: "none" for m in leaves if m.smiles == "CC(=O)O"})
    output = ra.compute_metric(external_data, categories)
    assert math.isclose(
        output.metric_value,
        0.79,
        rel_tol=1e-9,
    )

    # medium-bad case: the starting materials for the leaf reactions are not available internally; those for the root are
    external_data = {
        m.uid: "none"
        for m in leaves
        if m.smiles in ["Nc1ccc(N2CCOCC2)cc1", "O=C1c2ccccc2C(=O)N1CC1CO1"]
    }
    external_data.update(
        {m.uid: "vendor" for m in leaves if m.smiles == "O=C(n1ccnc1)n1ccnc1"}
    )
    external_data.update({m.uid: "syngenta" for m in leaves if m.smiles == "CC(=O)O"})
    output = ra.compute_metric(external_data, categories)
    assert math.isclose(
        output.metric_value,
        0.21,
        rel_tol=1e-9,
    )


@patch.object(ReactantAvailability, "_check_route_format")
def test_reactant_availability_no_cat(mock_fmt_check):
    route = Mock(specc=MonopartiteReacSynGraph)
    mock_fmt_check.return_value = route
    ra = ReactantAvailability(route)
    external_data = {}

    with pytest.raises(MissingDataError):
        ra.compute_metric(external_data, categories=None)


def test_yield_score(route):
    syngraph = MonopartiteReacSynGraph(route)
    all_steps = syngraph.get_unique_nodes()
    steps_uid = [ce.uid for ce in all_steps]
    yield_sc = YieldMetric(syngraph)

    # best case scenario:all involved reactions have 100% yield
    external_data = {uid: 1.0 for uid in steps_uid}
    output = yield_sc.compute_metric(external_data)
    assert math.isclose(
        output.metric_value,
        1.0,
        rel_tol=1e-9,
    )
    # worst case scenario: all involved reactions have 0% yield
    external_data = {uid: 0.0 for uid in steps_uid}
    output = yield_sc.compute_metric(external_data)
    assert math.isclose(
        output.metric_value,
        0.0,
        rel_tol=1e-9,
    )
    # medium-good case: reactions close to the root have higher yield than those close to the leaves
    external_data = {
        ce.uid: 0.95
        for ce in all_steps
        if ce.smiles
        == "CC(=O)O.NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1>>CC(=O)NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1"
    }
    external_data.update(
        {
            ce.uid: 0.65
            for ce in all_steps
            if ce.smiles
            == "O=C1c2ccccc2C(=O)N1CC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1>>NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1"
        }
    )
    external_data.update(
        {
            ce.uid: 0.45
            for ce in all_steps
            if ce.smiles
            == "O=C(n1ccnc1)n1ccnc1.O=C1c2ccccc2C(=O)N1CC(O)CNc1ccc(N2CCOCC2)cc1>>O=C1c2ccccc2C(=O)N1CC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1"
        }
    )
    external_data.update(
        {
            ce.uid: 0.25
            for ce in all_steps
            if ce.smiles
            == "Nc1ccc(N2CCOCC2)cc1.O=C1c2ccccc2C(=O)N1CC1CO1>>O=C1c2ccccc2C(=O)N1CC(O)CNc1ccc(N2CCOCC2)cc1"
        }
    )

    output_mg = yield_sc.compute_metric(external_data)
    assert math.isclose(
        output_mg.metric_value,
        0.71,
        rel_tol=1e-9,
    )
    # medium-bad case: reactions close to the root have lower yield than those close to the leaves
    external_data = {
        ce.uid: 0.25
        for ce in all_steps
        if ce.smiles
        == "CC(=O)O.NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1>>CC(=O)NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1"
    }
    external_data.update(
        {
            ce.uid: 0.45
            for ce in all_steps
            if ce.smiles
            == "O=C1c2ccccc2C(=O)N1CC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1>>NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1"
        }
    )
    external_data.update(
        {
            ce.uid: 0.65
            for ce in all_steps
            if ce.smiles
            == "O=C(n1ccnc1)n1ccnc1.O=C1c2ccccc2C(=O)N1CC(O)CNc1ccc(N2CCOCC2)cc1>>O=C1c2ccccc2C(=O)N1CC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1"
        }
    )
    external_data.update(
        {
            ce.uid: 0.95
            for ce in all_steps
            if ce.smiles
            == "Nc1ccc(N2CCOCC2)cc1.O=C1c2ccccc2C(=O)N1CC1CO1>>O=C1c2ccccc2C(=O)N1CC(O)CNc1ccc(N2CCOCC2)cc1"
        }
    )

    output_mb = yield_sc.compute_metric(external_data)
    assert math.isclose(
        output_mb.metric_value,
        0.44,
        rel_tol=1e-9,
    )
    assert output_mg.metric_value > output_mb.metric_value


def test_starting_materials_amount(route):
    syngraph = MonopartiteReacSynGraph(route)
    sm_amount = StartingMaterialsAmount(syngraph)

    route_steps = syngraph.get_unique_nodes()
    steps_uid = [ce.uid for ce in route_steps]
    external_info = {
        "target_amount": 319.15320615199994,  # 1 mol
        "yield": {uid: 0.5 for uid in steps_uid},
    }
    out = sm_amount.compute_metric(external_info)
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

    leaves_uid = sorted([m.uid for m in syngraph.get_molecule_leaves()])
    external_data = {uid: True for uid in leaves_uid[:-1]}
    external_data.update({leaves_uid[-1]: False})
    rc = RenewableCarbonMetric(syngraph)

    # a fully mapped route is needed
    with pytest.raises(NotFullyMappedRouteError):
        rc.compute_metric(external_data=external_data)
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
    rc = RenewableCarbonMetric(syngraph)
    out = rc.compute_metric(external_data=external_data)
    assert out.metric_value == round(14.0 / 16.0, 2)
    assert out.additional_info

    # none of the provided building blocks appears in the route
    external_data = {uid: False for uid in leaves_uid}
    out = rc.compute_metric(external_data=external_data)
    assert math.isclose(out.metric_value, 0.0, rel_tol=1e-9)
