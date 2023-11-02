from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Union

from linchemin.services.namerxn import service
from linchemin.services.rxnmapper import service as rxn_service
from linchemin.utilities import console_logger

"""
Module containing classes and functions for the pipeline of atom-2-atom mapping of chemical equations
"""

logger = console_logger(__name__)


class UnavailableMapper(KeyError):
    """Raised if the selected mapper is not among the available ones."""

    pass


@dataclass
class MappingOutput:
    """Dataclass to store the results of an atom-to-atom mapping"""

    mapped_reactions: list = field(default_factory=list)
    """ A list of dictionaries with the successfully mapped reactions"""
    unmapped_reactions: list = field(default_factory=list)
    """ A list of dictionaries with the reactions that could not be mapped"""
    pipeline_success_rate: dict = field(default_factory=dict)
    """ a dictionary in the form {'mapper_name': success_rate} to store the success rates of single mappers
        within a pipeline"""

    @property
    def success_rate(self) -> Union[float, int]:
        """A float between 0.0 and 1 indicating the success rate of the mapping"""
        if self.mapped_reactions:
            return len(self.mapped_reactions) / (
                len(self.mapped_reactions) + len(self.unmapped_reactions)
            )
        else:
            return 0.0


# Mappers factory


class Mapper(ABC):
    """Abstract class for the atom mappers"""

    @abstractmethod
    def map_chemical_equations(self, reactions_list: List[dict]) -> MappingOutput:
        pass


class NameRxnMapper(Mapper):
    """Class for the NameRxn atom mapper"""

    info = "NextMove reaction classifier. Needs credentials"

    def map_chemical_equations(self, reactions_list: List[dict]):
        # print('NameRxn mapper is called')
        out = MappingOutput()
        namerxn_service = service.NamerxnService(base_url="http://127.0.0.1:8004/")
        input_dict = {
            "inp_fmt": "smiles",
            "out_fmt": "smiles",
            "classification_code": "namerxn",
            "mapping_style": "matching",
            "query_data": reactions_list,
        }
        endpoint = namerxn_service.endpoint_map.get("run_batch")
        out_request = endpoint.submit(request_input=input_dict)
        # if the mapper is not available, raise an error MapperUnavailableError.

        out.mapped_reactions = [
            {"query_id": d["query_id"], "output_string": d["output_string"]}
            for d in out_request["output"]["successes_list"]
        ]
        out.unmapped_reactions = out_request["output"]["failure_list"]
        # To check the reaction classification
        # for d in out_request['output']['successes_list']:
        #     print(d['output_string'])
        #     print(d['reaction_class_id'])
        return out


# class ChematicaMapper(Mapper):
#     """ Class for the Chematica atom mapper """
#     info = 'Atom mapper developed in the Chematica software'
#
#     def map_chemical_equations(self, reactions_list: List[dict]):
#         # print('Chematica mapper is called')
#         out = MappingOutput()
#         out.unmapped_reactions = reactions_list
#         # response = namerxn_sdk_wrapper(reactions_list)
#         # if the mapper is not available, raise an error MapperUnavailableError.
#         # out.mapped_reactions = response['success_list]
#         # out.unmapped_reactions = response['failure_list']
#         return out


class RxnMapper(Mapper):
    """Class for the IbmRxn atom mapper"""

    info = "Atom mapper developed by IBM"

    def map_chemical_equations(self, reactions_list: List[dict]):
        # print('RxnMapper mapper is called')
        out = MappingOutput()
        rxnmapper_service = rxn_service.RxnMapperService(
            base_url="http://127.0.0.1:8002/"
        )
        input_dict = {
            "classification_code": "namerxn",
            "inp_fmt": "smiles",
            "out_fmt": "smiles",
            "mapping_style": "matching",
            "query_data": reactions_list,
        }
        endpoint = rxnmapper_service.endpoint_map.get("run_batch")
        out_request = endpoint.submit(request_input=input_dict)
        # if the mapper is not available, raise an error MapperUnavailableError.
        out.mapped_reactions = [
            {"query_id": d["query_id"], "output_string": d["output_string"]}
            for d in out_request["output"]["successes_list"]
        ]
        out.unmapped_reactions = out_request["output"]["failure_list"]
        return out


class MapperFactory:
    mappers = {
        "namerxn": {"value": NameRxnMapper, "info": NameRxnMapper.info},
        # 'chematica': {'value': ChematicaMapper,
        #               'info': ChematicaMapper.info},
        "rxnmapper": {"value": RxnMapper, "info": RxnMapper.info},
    }

    def call_mapper(self, mapper_name, reactions_list):
        """Takes a string indicating a mapper and calls it"""
        if mapper_name not in self.mappers:
            logger.error(
                f"'{mapper_name}' is not a valid mapper. Available mappers are: {list(self.mappers.keys())}"
            )
            raise UnavailableMapper

        mapper = self.mappers[mapper_name]["value"]
        return mapper().map_chemical_equations(reactions_list)


def perform_atom_mapping(reactions_list: List[dict], mapper_name: str) -> MappingOutput:
    """Gives access to the mapper factory.

    :param:
        reactions_list: a list of dictionaries with the reaction strings to be mapped

        mapper_name: a string indicating the name of the mapper to be used
    """
    factory = MapperFactory()
    return factory.call_mapper(mapper_name, reactions_list)


# Mapping chain


class MappingStep(ABC):
    """Abstract handler for the concrete handlers of consecutive atom mappers"""

    @abstractmethod
    def mapping(self, out: MappingOutput):
        pass


class FirstMapping(MappingStep):
    """Concrete handler to call the first mapper"""

    def mapping(self, out: MappingOutput):
        mapper = "namerxn"
        # try:
        mapper_output = perform_atom_mapping(out.unmapped_reactions, mapper)
        out.mapped_reactions = mapper_output.mapped_reactions
        # print(out.mapped_reactions)
        out.unmapped_reactions = mapper_output.unmapped_reactions
        out.pipeline_success_rate[mapper] = mapper_output.success_rate
        if out.success_rate == 1.0:
            return out
        else:
            return ThirdMapping().mapping(out)
            # return ThirdMapping().mapping(out)
        # except: MapperUnavailableError
        #   return SecondMapping().mapping(out)


# class SecondMapping(MappingStep):
#     """ Concrete handler to call the second mapper """
#
#     def mapping(self, out: MappingOutput):
#         mapper = 'chematica'
#         # try:
#         mapper_output = perform_atom_mapping(out.unmapped_reactions, mapper)
#         if mapper_output.mapped_reactions is not []:
#             out.mapped_reactions.extend(mapper_output.mapped_reactions)
#         out.unmapped_reactions = mapper_output.unmapped_reactions
#         out.pipeline_success_rate[mapper] = mapper_output.success_rate
#         if out.success_rate == 1.0:
#             return out
#         else:
#             return ThirdMapping().mapping(out)
#         # except: MapperUnavailableError
#         #   return ThirdMapping().mapping(out)


class ThirdMapping(MappingStep):
    """Concrete handler to call the third mapper"""

    def mapping(self, out):
        mapper = "rxnmapper"
        # try:
        mapper_output = perform_atom_mapping(out.unmapped_reactions, mapper)
        if mapper_output.mapped_reactions is not []:
            out.mapped_reactions.extend(mapper_output.mapped_reactions)
        out.unmapped_reactions = mapper_output.unmapped_reactions
        out.pipeline_success_rate[mapper] = mapper_output.success_rate
        if out.success_rate != 1.0:
            logger.warning("Some reactions remain unmapped at the end of the pipeline")
        return out
        # except: MapperUnavailableError
        #   return out


class MappingBuilder:
    """Class to start the chain calling the handler of the first mapper"""

    @staticmethod
    def initiate_mapping(reactions_list):
        out = MappingOutput()
        out.unmapped_reactions = reactions_list
        return FirstMapping().mapping(out)


def pipeline_atom_mapping(reactions_list: List[dict] = None) -> MappingOutput:
    """Facade function to start the atom-to-atom mapping pipeline.

    :param:
         reactions_list: a list of dictionaries containing the reaction strings to be mapped and their id in the
                         form [{'query_id': n, 'input_string': unmapped_reaction_string}]

    :return:
        out: a MappingOutput instance
    """
    return MappingBuilder().initiate_mapping(reactions_list)


def get_available_mappers():
    """Returns a dictionary with the available mappers and some info"""
    return {
        f: additional_info["info"]
        for f, additional_info in MapperFactory.mappers.items()
    }
