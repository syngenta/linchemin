class TranslationError(Exception):
    """Base class for exceptions leading to unsuccessful translation."""

    pass


class EmptyRoute(TranslationError):
    """Raised if an empty route is found"""

    pass


class InvalidRoute(TranslationError):
    """Raised if the route does not contain at least two molecules connected by an edge"""


class UnavailableFormat(TranslationError):
    """Raised if the selected format is not among the available ones"""

    pass


class UnavailableDataModel(TranslationError):
    """Raised if the selected output format is not among the available ones"""

    pass


class UnavailableTranslation(TranslationError):
    """Raised if the required translation cannot be performed"""

    pass
