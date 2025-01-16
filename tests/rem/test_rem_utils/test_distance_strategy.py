import math
from unittest.mock import Mock, patch

import pytest

from linchemin.cgu.syngraph import BipartiteSynGraph
from linchemin.cheminfo.models import ChemicalEquation, Molecule
from linchemin.rem.rem_utils.distance_strategy import (
    DistanceContext,
    LongestLinearSequenceWeightedDistanceStrategy,
    SimpleDistanceStrategy,
    distance_function_calculator,
)


def test_distance_strategy():
    route = BipartiteSynGraph()
    node = Molecule()
    root = Molecule()
    # if an unavailable distance function is chosen, a KeyError is raised
    with pytest.raises(KeyError) as ke:
        distance_function_calculator("wrong_function", route, node, root)
    assert "KeyError" in str(ke.type)


@patch("linchemin.rem.rem_utils.distance_strategy.find_path")
def test_simple_distance_strategy(mock_find_path):
    # Create mock objects
    mock_route = Mock(spec=BipartiteSynGraph)
    mock_node = Mock(spec=Molecule)
    mock_root = Mock(spec=Molecule)
    mock_chem_eq1 = Mock(spec=ChemicalEquation)
    mock_chem_eq2 = Mock(spec=ChemicalEquation)
    mock_molecule = Mock(spec=Molecule)
    mock_path = [mock_node, mock_chem_eq1, mock_molecule, mock_chem_eq2, mock_root]
    mock_find_path.return_value = mock_path

    strategy = SimpleDistanceStrategy()
    distance = strategy.distance(mock_route, mock_node, mock_root)

    mock_find_path.assert_called_once_with(mock_route, mock_node, mock_root)

    assert distance == 2


@patch("linchemin.rem.rem_utils.distance_strategy.find_path")
def test_simple_distance_strategy_no_chemical_equations(mock_find_path):
    mock_route = Mock(spec=BipartiteSynGraph)
    mock_node = Mock(spec=Molecule)
    mock_root = Mock(spec=Molecule)
    # the path does not contain any ChemicalEquation
    mock_path = [mock_node, mock_root]
    mock_find_path.return_value = mock_path

    strategy = SimpleDistanceStrategy()
    distance = strategy.distance(mock_route, mock_node, mock_root)

    mock_find_path.assert_called_once_with(mock_route, mock_node, mock_root)

    assert distance == 0


@patch("linchemin.rem.rem_utils.distance_strategy.find_path")
def test_simple_distance_strategy_empty_path(mock_find_path):
    mock_route = Mock(spec=BipartiteSynGraph)
    mock_node = Mock(spec=Molecule)
    mock_root = Mock(spec=Molecule)

    # Set up an empty mock path
    mock_find_path.return_value = []

    strategy = SimpleDistanceStrategy()
    distance = strategy.distance(mock_route, mock_node, mock_root)
    mock_find_path.assert_called_once_with(mock_route, mock_node, mock_root)

    assert distance == 0


@patch("linchemin.rem.rem_utils.distance_strategy.descriptor_calculator")
@patch("linchemin.rem.rem_utils.distance_strategy.find_path")
def test_lls_distance_strategy(mock_find_path, mock_lls):
    # Create mock objects
    mock_route = Mock(spec=BipartiteSynGraph)
    mock_node = Mock(spec=Molecule)
    mock_root = Mock(spec=Molecule)
    mock_chem_eq1 = Mock(spec=ChemicalEquation)
    mock_chem_eq2 = Mock(spec=ChemicalEquation)
    mock_molecule = Mock(spec=Molecule)
    mock_path = [mock_node, mock_chem_eq1, mock_molecule, mock_chem_eq2, mock_root]
    mock_find_path.return_value = mock_path
    mock_lls.return_value = 2

    strategy = LongestLinearSequenceWeightedDistanceStrategy()
    assert math.isclose(
        strategy.distance(mock_route, mock_node, mock_root),
        0.333,
        rel_tol=1e-9,
    )
    mock_find_path.assert_called()


@patch("linchemin.rem.rem_utils.distance_strategy.descriptor_calculator")
@patch("linchemin.rem.rem_utils.distance_strategy.find_path")
def test_lls_distance_strategy_empty_path(mock_find_path, mock_lls):
    mock_route = Mock(spec=BipartiteSynGraph)
    mock_node = Mock(spec=Molecule)
    mock_root = Mock(spec=Molecule)

    # Set up an empty mock path
    mock_find_path.return_value = []
    mock_lls.return_value = 0
    strategy = LongestLinearSequenceWeightedDistanceStrategy()

    assert math.isclose(
        strategy.distance(mock_route, mock_node, mock_root),
        1.0,
        rel_tol=1e-9,
    )


@pytest.fixture
def distance_context():
    return DistanceContext()


def test_set_strategy_simple(distance_context):
    distance_context.set_strategy("simple")
    assert isinstance(distance_context.strategy, SimpleDistanceStrategy)


def test_set_strategy_longest_sequence(distance_context):
    distance_context.set_strategy("longest_sequence")
    assert isinstance(
        distance_context.strategy, LongestLinearSequenceWeightedDistanceStrategy
    )


def test_set_strategy_invalid(distance_context):
    with pytest.raises(KeyError):
        distance_context.set_strategy("invalid_strategy")


@patch.object(SimpleDistanceStrategy, "distance")
def test_calculate_distance_simple(mock_distance, distance_context):
    mock_route = Mock()
    mock_node = Mock()
    mock_root = Mock()
    mock_distance.return_value = 5

    distance_context.set_strategy("simple")
    distance_context.calculate_distance(mock_route, mock_node, mock_root)

    mock_distance.assert_called_once_with(mock_route, mock_node, mock_root)
    assert distance_context.get_distance() == 5
