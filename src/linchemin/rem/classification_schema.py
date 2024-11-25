from typing import Union, List, Dict
import operator
from linchemin.utilities import console_logger

logger = console_logger(__name__)


class CriterionTypeError(ValueError):
    """To be raised when the input type for a criterion is incorrect"""


class CriterionOperatorError(ValueError):
    """To be raised when the input type for a criterion is incorrect"""


class Criterion:
    """Class representing a criterion of classification"""

    operators = {
        "gt": operator.gt,
        "ge": operator.ge,
        "lt": operator.lt,
        "le": operator.le,
        "eq": operator.eq,
        "ne": operator.ne,
    }

    def __init__(self, value: Union[str, Dict[str, float]]):
        if isinstance(value, str):
            self.type = "categorical"
            self.value = value
        elif isinstance(value, dict):
            self.type = "numerical"
            self.conditions = self._parse_conditions(value)
        else:
            logger.error("Invalid criterion type")
            raise CriterionTypeError

    def _parse_conditions(self, value: Dict[str, float]) -> List[tuple]:
        """To parse conditions composed of more than one operator"""
        conditions = []
        for op_string, threshold in value.items():
            op = self._get_operator(op_string)
            if op is None:
                raise CriterionOperatorError(f"Invalid operator: {op_string}")
            conditions.append((op, threshold))
        return conditions

    def _get_operator(self, op_string: str):
        return self.operators.get(op_string, None)

    def check(self, value: Union[str, float]) -> bool:
        if self.type == "categorical":
            return value == self.value
        if not isinstance(value, (int, float)):
            return False
        return all(op(value, threshold) for op, threshold in self.conditions)

    def __eq__(self, other):
        if not isinstance(other, Criterion):
            return NotImplemented

        if self.type != other.type:
            return False

        if self.type == "categorical":
            return self.value == other.value
        else:
            return self.conditions == other.conditions

    def __repr__(self):
        if self.type == "categorical":
            return f"Criterion('{self.value}')"
        else:
            conditions_str = ", ".join(
                f"{op.__name__}: {threshold}" for op, threshold in self.conditions
            )
            return f"Criterion({{{conditions_str}}})"


class Category:
    """Class representing a category"""

    def __init__(
        self, name: str, criterion: Union[str, Dict[str, float]], score: float
    ):
        self.name = name
        self.criterion = Criterion(criterion)
        self.score = score

    def _get_criterion_repr(self) -> str:
        """To generate a string representing the criterion"""
        if self.criterion.type == "categorical":
            return f"'{self.criterion.value}'"
        conditions = []
        for op, threshold in self.criterion.conditions:
            op_name = op.__name__
            conditions.append(f"'{op_name}': {threshold}")
        return "{" + ", ".join(conditions) + "}"

    def __repr__(self):
        criterion_str = self._get_criterion_repr()
        return f"Category(name='{self.name}', criterion={criterion_str}, score={self.score})"


class ClassificationSchema:
    def __init__(
        self, categories: List[Dict[str, Union[str, Dict[str, float], float]]]
    ):
        self.categories = [
            Category(cat["name"], cat["criterion"], cat["score"]) for cat in categories
        ]

    def categorize(self, value: Union[str, float]) -> Union[Category, None]:
        """To categorize a value"""
        for category in self.categories:
            if category.criterion.check(value):
                return category
        return None

    def compute_score(self, value: Union[str, float]) -> float:
        """To compute the score associated with a value"""
        category = self.categorize(value)
        if category is not None:
            return category.score

    def __repr__(self):
        return f"ClassificationSchema(categories={self.categories})"
