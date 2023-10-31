from abc import ABC, abstractmethod
from typing import Optional

import requests

from linchemin.services.callbacks import default_on_success
from linchemin.services.decorators import response_handling


class ServiceRoute:
    def __init__(self, base_url: Optional[str] = None, api_version: str = "v1") -> None:
        """
        Initialize the routes of the service.
        Parameters:
            base_url (str): base url for the service. If not provided it will default to
            the environment variable provided in input.
            api_version (str, optional): api version. If not provided it will default to v1.
        """
        self._base_url = base_url
        self._api_version = api_version
        self._update_routes()

    @abstractmethod
    def _update_routes(self) -> None:
        """Update all the routes."""

    @property
    def base_url(self) -> str:
        """
        Get the base url for the service.
        Returns:
            str: base url for the service.
        """
        return self._base_url

    @base_url.setter
    def base_url(self, value: str) -> None:
        """
        Set the base url for the service.
        Parameters:
            value (str): base url to set.
        """
        self._base_url = value
        self._update_routes()


class NamerxnServiceSDK:
    """
    Python wrapper  to access the REST API requests of the namerxn service.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        NamerxnServiceSDK constructor.
        Parameters:
            api_key (str): an API key to access the service.
            project_id (str, optional): project identifier. Defaults to None.
            base_url (str, optional): base url for the service. If not provided it will default to
                the environment variable RXN4CHEMISTRY_BASE_URL or https://rxn.res.ibm.com.
        Examples:
            Initialize the wrapper by simply providing an API key:
            #>>> from rxn4chemistry import RXN4ChemistryWrapper
            #>>> rxn4chemistry_wrapper = RXN4ChemistryWrapper(api_key=api_key)
        """
        self._api_key = api_key
        self.project_id = project_id
        self.headers = self._construct_headers()
        self.routes = ServiceRoute(base_url)

    def set_base_url(self, base_url: str) -> None:
        """
        Set base url for the namerxn service.
        Args:
            base_url (str): base url for the service to set.
        """
        self.routes.base_url = base_url

    def _construct_headers(self) -> dict:
        """
        Construct header, required for all requests.
        Returns:
            dict: dictionary containing the "Content-Type" and the
                "Authorization".
        """
        return {"Content-Type": "application/json"}

    @response_handling(success_status_code=201, on_success=default_on_success)
    def submit(self):
        headers = {
            "accept": "application/json",
            # Already added when you pass json= but not when you pass data=
            # 'Content-Type': 'application/json',
        }

        params = {
            "classification_code": "namerxn",
            "inp_fmt": "smiles",
            "out_fmts": "smiles",
            "mapping_style": "matching",
        }

        json_data = [
            {
                "input_string": "CC(Cl)=O.CN>>CNC(C)=O",
                "query_id": "1",
            },
        ]

        response = requests.post(
            "http://127.0.0.1:8002/run_batch/",
            params=params,
            headers=headers,
            json=json_data,
        )
        print(response.json())
        print(response.status_code)
        return response


class ServiceEndPoint(ABC):
    @property
    @abstractmethod
    def input(self):
        ...

    pass

    def output(self):
        ...

    pass


class Service(ABC):
    pass


if __name__ == "__main__":
    sdk = NamerxnServiceSDK(base_url="http://127.0.0.1:8002")
    sdk.submit()
