import requests


def default_on_success(response: requests.models.Response) -> dict:
    """
    Process the successful response.

    Parameters:
        response (requests.models.Response): response from an API request.
    Returns:
        dict: dictionary representing the response.
    """
    response_dict = response.json()
    return {"response": response_dict}
