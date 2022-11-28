import string
from typing import Optional, List, ClassVar
from enum import Enum
from pydantic import BaseModel


class ReactionFormat(Enum):
    SMILES = 'smiles'
    # RXNBLOCK = 'rxnblock'


class Software(BaseModel):
    name: str
    version: str


class QueryReactionString(BaseModel):
    query_id: str
    input_string: str

    class Config:
        schema_extra = {
            "example": {
                'input_string': 'CC(Cl)=O.CN>>CNC(C)=O',
                'query_id': '1',
            }
        }


class ResultsReactionString(BaseModel):
    query_id: str
    output_string: Optional[str] = None
    confidence: Optional[float] = None
    notes: Optional[dict] = None
    success: bool

    class Config:
        schema_extra = {
            "example": {
                "reaction_smiles_mapped": "Cl[C:3]([CH3:4])=[O:5].[CH3:1][NH2:2]>>[CH3:1][NH:2][C:3]([CH3:4])=[O:5]",
                "query_id": "1",
                "confidence": 0.981,
                "success": True,
                "notes": {}
            }
        }


## endpoints I/O
class Metadata(BaseModel):
    ci_toolkit: Software
    a2a_mapper: Software

    class Config:
        schema_extra = {
            "example": {
                "ci_toolkit": {
                    "name": "RDKit",
                    "version": "2022.09.1"
                },
                "a2a_mapper": {
                    "name": "rxnmapper",
                    "version": "0.2.4"
                }
            }
        }


class RunBatchInp(BaseModel):
    query_data: List[QueryReactionString]
    inp_fmt: str
    out_fmt: str

    query_data_field: ClassVar[str] = 'query_data'
    parameter_fields: ClassVar[List[str]] = ['inp_fmt', 'out_fmt', ]

    class Config:
        schema_extra = {
            "example": {
                'classification_code': 'namerxn',
                'inp_fmt': 'smiles',
                'out_fmt': 'smiles',
                'mapping_style': 'matching',
                'query_data': [
                    {
                        "input_string": "CC(Cl)=O.CN.CC#N>>CNC(C)=O.O",
                        "query_id": "1"
                    }
                ]
            }
        }


class RunBatchOut(BaseModel):
    metadata: Metadata
    query_parameters: dict
    output: dict
    # success_list: List[ResultsReactionString]
    # failure_list: List[QueryReactionString]
    outcome: dict

    class Config:
        schema_extra = {
            "example": {
                'metadata': {
                    'ci_toolkit': {'name': 'RDKit', 'version': '2022.09.1'},
                    'a2a_mapper': {'name': 'rxnmapper', 'version': '0.2.4'}},
                'query_parameters': {'inp_fmt': 'smiles', 'out_fmt': 'smiles'},
                'output': {'successes_list':
                               [{'query_id': '1',
                                 'output_string': 'CC#N.Cl[C:3]([CH3:4])=[O:5].[CH3:1][NH2:2]>>[CH3:1][NH:2][C:3]([CH3:4])=[O:5].[OH2:6]',
                                 'confidence': 0.981, 'notes': {}, 'success': True}], 'failure_list': []},
                'outcome': {}
            }
        }


endpoint_info_map = {
    'metadata': {'ep_url': 'metadata',
                 'input_schema': None,
                 'output_schema': Metadata,
                 'request_method': 'get', },

    'run_batch': {'ep_url': 'run_batch',
                  'input_schema': RunBatchInp,
                  'output_schema': RunBatchOut,
                  'request_method': 'post'}
}
