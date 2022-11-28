#############SDK####################
from urllib.parse import urljoin
from pydantic import BaseModel, ValidationError

import requests
from . import schemas


class RxnMapperService:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.endpoint_map = {k: EndPoint(base_url=self.base_url, **v) for k, v in schemas.endpoint_info_map.items()}


class EndPoint:
    def __init__(self, base_url: str, ep_url: str, request_method: str, input_schema, output_schema, **garbage):
        self.url = urljoin(base_url, ep_url)
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.request_method = request_method

        if input_schema:
            self.input_example = self.input_schema.Config().schema_extra.get('example')
        else:
            self.input_example = None
        self.output_example = self.output_schema.Config().schema_extra.get('example')

    def validate_input(self, data):
        try:
            self.input_schema(**data)
        except ValidationError as e:
            print(e)

    def validate_output(self, data):
        try:
            self.output_schema(**data)
        except ValidationError as e:
            print(e)

    def submit(self, request_input: dict = None):
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
        }
        if request_input:
            self.validate_input(data=request_input)

        if self.input_schema:
            if qdf := self.input_schema.query_data_field:
                query_data = request_input.get(qdf)
            else:
                query_data = request_input
            if pf := self.input_schema.parameter_fields:
                parameters = {k: v for k, v in request_input.items() if k in pf}
            else:
                parameters = {}
        else:
            query_data = None
            parameters = {}

        response = requests.request(self.request_method, self.url, headers=headers, params=parameters, json=query_data)
        request_results = response.json()

        self.validate_output(data=request_results)

        return request_results

