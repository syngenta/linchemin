from linchemin.rem.rem_utils.classification_schema import (
    Criterion,
    CriterionTypeError,
    Category,
    ClassificationSchema,
    CriterionOperatorError,
)
import operator
import pytest

from unittest.mock import Mock


def test_criterion_categorical():
    criterion_categorical = "category1"
    criterion = Criterion(criterion_categorical)
    assert criterion.type == "categorical"
    assert criterion.value == "category1"


def test_criterion_numerical():
    criterion_numerical = {"gt": 0}
    criterion = Criterion(criterion_numerical)
    assert criterion.type == "numerical"
    assert criterion.conditions == [(operator.gt, 0)]

    criterion_numerical = {"gt": 0, "lt": 1}
    criterion = Criterion(criterion_numerical)
    assert criterion.type == "numerical"
    assert criterion.conditions == [(operator.gt, 0), (operator.lt, 1)]


def test_criterion_wrong_type():
    with pytest.raises(CriterionTypeError):
        Criterion(1)


def test_criterion_wrong_operator():
    with pytest.raises(CriterionOperatorError):
        Criterion({"operator": 2})


@pytest.mark.parametrize(
    "criterion_value, test_value, expected_result",
    [
        ({"eq": 6}, 6, True),
        ({"eq": 6}, 0.5, False),
        ({"lt": 10}, 9, True),
        ({"lt": 0.5}, 10, False),
        ({"gt": 5}, 6, True),
        ({"gt": 5}, 5, False),
        ("cat1", "cat1", True),
        ("cat1", "cat2", False),
        ("special", "special", True),
        ("special", "normal", False),
    ],
)
def test_criterion_check(criterion_value, test_value, expected_result):
    criterion = Criterion(criterion_value)
    assert criterion.check(test_value) is expected_result


def test_category():
    category_name = "test_category_categorical"
    category_criterion = "cat1"
    category_score = 0.5
    category = Category(
        name=category_name, criterion=category_criterion, score=category_score
    )
    assert category.name == category_name
    assert category.criterion == Criterion(category_criterion)
    assert category.score


@pytest.fixture
def test_categories():
    return [
        {"name": "Low", "criterion": {"lt": 50}, "score": 1},
        {"name": "Medium", "criterion": {"ge": 50, "lt": 80}, "score": 2},
        {"name": "High", "criterion": {"ge": 80}, "score": 3},
        {"name": "Special", "criterion": "VIP", "score": 4},
    ]


@pytest.fixture
def schema_test(test_categories):
    return ClassificationSchema(categories=test_categories)


def test_classification_schema_init(schema_test, test_categories):
    assert schema_test.categories
    assert len(schema_test.categories) == len(test_categories)
    for cat, category in zip(schema_test.categories, test_categories):
        assert isinstance(cat, Category)
        assert cat.name == category["name"]
        assert cat.score == category["score"]
        assert isinstance(cat.criterion, Criterion)


@pytest.mark.parametrize(
    "value, expected_category",
    [
        (30, "Low"),
        (60, "Medium"),
        (90, "High"),
        ("VIP", "Special"),
        (100, "High"),
        (0, "Low"),
        ("Regular", None),
    ],
)
def test_categorize(schema_test, value, expected_category):
    result = schema_test.categorize(value)
    if expected_category is None:
        assert result is None
    else:
        assert result.name == expected_category


def test_categorize_calls_check(schema_test):
    for category in schema_test.categories:
        category.criterion.check = Mock(return_value=False)

    schema_test.categorize(50)

    for category in schema_test.categories:
        category.criterion.check.assert_called_once_with(50)


def test_repr(schema_test):
    repr_string = repr(schema_test)
    assert repr_string.startswith("ClassificationSchema(categories=[")
    assert repr_string.endswith("])")
    for category in schema_test.categories:
        assert str(category) in repr_string
