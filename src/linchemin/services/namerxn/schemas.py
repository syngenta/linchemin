from enum import Enum
from typing import ClassVar, Dict, List, Optional, Union

from pydantic import BaseModel


class ReactionFormat(Enum):
    SMILES = "smiles"
    RXNBLOCK = "rxnblock"


class ReactionMappingStyle(Enum):
    MATCHING = "matching"
    COMPLETE = "complete"
    COMPLETER = "completer"
    NOMAP = "nomap"


class ClassificationCode(Enum):
    # FILBERT = 'filbert'
    NAMERXN = "namerxn"
    # NAMERXN_FILBERT = 'namerxn_filbert'


class Software(BaseModel):
    name: str
    version: str


class ReactionClassInfo(BaseModel):
    namerxn_class_L1_name: str
    namerxn_class_L1_number: str
    namerxn_class_L2_name: str
    namerxn_class_L2_number: str
    namerxn_class_L3_name: str
    namerxn_class_L3_number: Optional[str] = ""
    namerxn_class_name: str
    namerxn_class_number: str
    namerxn_version: str
    rxno_link: Optional[str] = ""
    rxno_number: Optional[str] = ""
    wiki: Optional[str] = ""


class QueryReactionString(BaseModel):
    query_id: str
    input_string: str

    class Config:
        schema_extra = {
            "example": {
                "input_string": "CC(Cl)=O.CN>>CNC(C)=O",
                "query_id": "1",
            }
        }


class ResultsReactionString(BaseModel):
    query_id: str
    output_string: Optional[str] = None
    confidence: Optional[float] = None
    reaction_class_id: Optional[str] = None
    notes: Optional[dict] = None
    success: bool

    class Config:
        schema_extra = {
            "example": {
                "reaction_smiles_mapped": "CC#N.Cl[C:2]([CH3:1])=[O:3].[CH3:4][NH2:5]>>O.[CH3:1][C:2](=[O:3])[NH:5][CH3:4]",
                "query_id": "1",
                "confidence": None,
                "success": True,
                "notes": {},
            }
        }


## endpoints I/O
class Metadata(BaseModel):
    ci_toolkit: Software
    a2a_mapper: Software
    reaction_classification_library: Software

    class Config:
        schema_extra = {
            "example": {
                "ci_toolkit": {"name": "RDKit", "version": "2022.09.1"},
                "a2a_mapper": {"name": "namrxn", "version": "3.4.0"},
                "reaction_classification_library": {
                    "name": "namrxn",
                    "version": "3.4.0",
                },
            }
        }


class AllReactionClassInfoOut(BaseModel):
    metadata: Metadata
    output: List[ReactionClassInfo]
    outcome: Dict


class ReactionClassInfoInp(BaseModel):
    namerxn_class_number: str
    query_data_field: ClassVar[str] = None
    parameter_fields: ClassVar[List[str]] = None

    class Config:
        schema_extra = {"example": {"namerxn_class_number": "2.3.1"}}


class ReactionClassInfoOut(BaseModel):
    metadata: Metadata
    output: Union[ReactionClassInfo, None]
    outcome: Dict

    class Config:
        schema_extra = {
            "example": {
                "metadata": {
                    "ci_toolkit": {"name": "RDKit", "version": "2022.09.1"},
                    "a2a_mapper": {"name": "namrxn", "version": "3.4.0"},
                    "reaction_classification_library": {
                        "name": "namrxn",
                        "version": "3.4.0",
                    },
                },
                "output": {
                    "namerxn_class_L1_name": "Acylation and related processes",
                    "namerxn_class_L1_number": "2",
                    "namerxn_class_L2_name": "N-acylation to urea",
                    "namerxn_class_L2_number": "3",
                    "namerxn_class_L3_name": "Isocyanate + amine urea coupling",
                    "namerxn_class_L3_number": "1",
                    "namerxn_class_name": "Isocyanate + amine urea coupling",
                    "namerxn_class_number": "2.3.1",
                    "namerxn_version": "3.4.0",
                    "rxno_link": "",
                    "rxno_number": "",
                    "wiki": "",
                },
                "outcome": {},
            }
        }


class RunBatchInp(BaseModel):
    query_data: List[QueryReactionString]
    classification_code: str
    inp_fmt: str
    out_fmt: str
    mapping_style: str

    query_data_field: ClassVar[str] = "query_data"
    parameter_fields: ClassVar[List[str]] = [
        "classification_code",
        "inp_fmt",
        "out_fmt",
        "mapping_style",
    ]

    class Config:
        schema_extra = {
            "example": {
                "inp_fmt": "smiles",
                "out_fmt": "smiles",
                "classification_code": "namerxn",
                "mapping_style": "matching",
                "query_data": [
                    {"input_string": "CC(Cl)=O.CN.CC#N>>CNC(C)=O.O", "query_id": "1"}
                ],
            }
        }


class RunBatchOut(BaseModel):
    metadata: Metadata
    query_parameters: dict
    output: dict
    outcome: dict

    class Config:
        schema_extra = {
            "example": {
                "metadata": {
                    "ci_toolkit": {"name": "RDKit", "version": "2022.09.1"},
                    "a2a_mapper": {"name": "namrxn", "version": "3.4.0"},
                    "reaction_classification_library": {
                        "name": "namrxn",
                        "version": "3.4.0",
                    },
                },
                "query_parameters": {
                    "classification_code": "namerxn",
                    "inp_fmt": "smiles",
                    "out_fmt": "smiles",
                    "mapping_style": "matching",
                },
                "output": {
                    "successes_list": [
                        {
                            "query_id": "1",
                            "reaction_class_id": "2.1.1",
                            "output_string": "CC#N.Cl[C:2]([CH3:1])=[O:3].[CH3:4][NH2:5]>>O.[CH3:1][C:2](=[O:3])[NH:5][CH3:4]",
                            "success": True,
                            "confidence": None,
                            "notes": {},
                        }
                    ],
                    "failure_list": [],
                },
                "outcome": {},
            }
        }


endpoint_info_map = {
    "metadata": {
        "ep_url": "metadata",
        "input_schema": None,
        "output_schema": Metadata,
        "request_method": "get",
    },
    "all_class_info": {
        "ep_url": "all_class_info",
        "input_schema": None,
        "output_schema": AllReactionClassInfoOut,
        "request_method": "get",
    },
    "class_info": {
        "ep_url": "class_info",
        "input_schema": ReactionClassInfoInp,
        "output_schema": ReactionClassInfoOut,
        "request_method": "get",
    },
    "run_batch": {
        "ep_url": "run_batch",
        "input_schema": RunBatchInp,
        "output_schema": RunBatchOut,
        "request_method": "post",
    },
}
