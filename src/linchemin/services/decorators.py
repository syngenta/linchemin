import logging
from functools import partial, wraps
from typing import Any, Callable, Optional

import requests

from .callbacks import default_on_success
from .response_handler import ResponseHandler

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def response_handling(
    function: Optional[Callable] = None,
    success_status_code: int = 200,
    on_success: Callable[[requests.models.Response], Any] = default_on_success,
) -> Callable:
    """
    Decorator to handle request responses.
    Args:
        function (Callable, optional): function to decorate.
        success_status_code (int): status expected on success.
        on_success (Callable): function to call on success.
    Returns:
        Callable: a function wrapped with the decorator.
    """
    if function is None:
        return partial(
            response_handling,
            success_status_code=success_status_code,
            on_success=on_success,
        )

    @wraps(function)
    def _wrapper(*args, **kwargs):
        logger.debug(
            f"request {function.__name__} with args={args} and kwargs={kwargs}"
        )
        response = function(*args, **kwargs)
        logger.debug(f"response {response.text}")

        response_handler = ResponseHandler(
            response=response,
            success_status_code=success_status_code,
            on_success=on_success,
        )
        return response_handler.handle()

    return _wrapper
