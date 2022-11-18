from abc import ABC, abstractmethod
from dataclasses import dataclass, field

"""
Module containing classes and functions for the pipeline of atom-2-atom mapping of chemical equations
"""

@dataclass
class MappingOutput:
    """ Class to store the results of an atom-to-atom mapping.

        Attributes:
            mapped_reactions: a list of dictionaries with the sucessfully mapped reactions

            unmapped_reactions: a list of dictionaries with the reactions that could not be mapped

            success_rate: a float between 0.0 and 1 indicating the success rate of the mapping

            pipeline_success_rate: a dictionary in the form {'mapper_name': success_rate} to store the success rates
                                   of single mappers within a pipeline
    """
    mapped_reactions: list = field(default_factory=list)
    unmapped_reactions: list = field(default_factory=list)
    pipeline_success_rate: dict = field(default_factory=dict)

    @property
    def success_rate(self) -> dict:
        if self.mapped_reactions:
            return len(self.mapped_reactions) / (len(self.mapped_reactions) + len(self.unmapped_reactions))
        else:
            return 0



# Mappers factory

class Mapper(ABC):
    """ Abstract class for the atom mappers """

    @abstractmethod
    def map_chemical_equations(self, reactions_list: list[dict]):
        pass


class NameRxnMapper(Mapper):
    """ Class for the NameRxn atom mapper """
    info = 'NextMove reaction classifier. Needs credentials'

    def map_chemical_equations(self, reactions_list: list[dict]):
        print('NameRxn mapper is called')
        out = MappingOutput()
        # response = namerxn_sdk_wrapper(reactions_list)
        # if the mapper is not available, raise an error MapperUnavailableError.
        # out.mapped_reactions = response['success_list]
        # out.unmapped_reactions = response['failure_list']
        return out


class ChematicaMapper(Mapper):
    """ Class for the Chematica atom mapper """
    info = 'Atom mapper developed in the Chematica software'

    def map_chemical_equations(self, reactions_list: list[dict]):
        print('Chematica mapper is called')
        out = MappingOutput()
        # response = namerxn_sdk_wrapper(reactions_list)
        # if the mapper is not available, raise an error MapperUnavailableError.
        # out.mapped_reactions = response['success_list]
        # out.unmapped_reactions = response['failure_list']
        return out


class IbmRxnMapper(Mapper):
    """ Class for the IbmRxn atom mapper """
    info = 'Atom mapper developed by IBM'

    def map_chemical_equations(self, reactions_list: list[dict]):
        print('RxnMapper mapper is called')
        out = MappingOutput()
        # response = namerxn_sdk_wrapper(reactions_list)
        # if the mapper is not available, raise an error MapperUnavailableError.
        # out.mapped_reactions = response['success_list]
        # out.unmapped_reactions = response['failure_list']
        return out


class MapperFactory:
    mappers = {'namerxn': {'value': NameRxnMapper,
                           'info': NameRxnMapper.info},
               'chematica': {'value': ChematicaMapper,
                             'info': ChematicaMapper.info},
               'rxnmapper': {'value': IbmRxnMapper,
                             'info': IbmRxnMapper.info}
               }

    def call_mapper(self, mapper_name, reactions_list):
        """ Takes a string indicating a mapper and calls it """
        if mapper_name not in self.mappers:
            raise KeyError(f"'{mapper_name}' is not a valid mapper."
                           f"Available mappers are: {self.mappers.keys()}")

        mapper = self.mappers[mapper_name]['value']
        return mapper().map_chemical_equations(reactions_list)


def perform_atom_mapping(mapper_name: str, reactions_list: list[dict]) -> MappingOutput:
    """ Gives access to the mapper factory.

        Parameters:
            mapper_name: a string indicating the name of the mapper to be used

            reactions_list: a list of dictionaries with the reaction strings to be mapped
    """
    factory = MapperFactory()
    return factory.call_mapper(mapper_name, reactions_list)


# Mapping chain

class MappingStep(ABC):
    """ Abstract handler for the concrete handlers of consecutive atom mappers """

    @abstractmethod
    def mapping(self):
        pass


class FirstMapping(MappingStep):
    """ Concrete handler to call the first mapper """

    def mapping(self, out):
        mapper = 'namerxn'
        # try:
        mapper_output = perform_atom_mapping(mapper, out.unmapped_reactions)
        out.mapped_reactions = mapper_output.mapped_reactions
        out.unmapped_reactions = mapper_output.unmapped_reactions
        out.pipeline_success_rate[mapper] = mapper_output.success_rate
        if out.success_rate == 1.0:
            return out
        else:
            return SecondMapping().mapping(out)
        # except: MapperUnavailableError
        #   return SecondMapping().mapping(out)


class SecondMapping(MappingStep):
    """ Concrete handler to call the second mapper """

    def mapping(self, out):
        mapper = 'chematica'
        # try:
        mapper_output = perform_atom_mapping(mapper, out.unmapped_reactions)
        out.mapped_reactions = mapper_output.mapped_reactions
        out.unmapped_reactions = mapper_output.unmapped_reactions
        out.pipeline_success_rate[mapper] = mapper_output.success_rate
        if out.success_rate == 1.0:
            return out
        else:
            return ThirdMapping().mapping(out)
        # except: MapperUnavailableError
        #   return ThirdMapping().mapping(out)


class ThirdMapping(MappingStep):
    """ Concrete handler to call the third mapper """

    def mapping(self, out):
        mapper = 'rxnmapper'
        # try:
        mapper_output = perform_atom_mapping(mapper, out.unmapped_reactions)
        out.mapped_reactions = mapper_output.mapped_reactions
        out.unmapped_reactions = mapper_output.unmapped_reactions
        out.pipeline_success_rate[mapper] = mapper_output.success_rate
        return out
        # except: MapperUnavailableError
        #   return out


class MappingBuilder:
    """ Class to start the chain calling the handler of the first mapper """

    @staticmethod
    def initiate_mapping(reactions_list):
        out = MappingOutput()
        out.unmapped_reactions = reactions_list
        return FirstMapping().mapping(out)


def pipeline_atom_mapping(reactions_list: list[dict]=None):
    """ Facade function to start the atom-to-atom mapping pipeline.

        Parameters:
             reactions_list: a list of dictionaries containing the reaction strings to be mapped and their id

        Returns:
            out: a MappingOutput instance
    """
    return MappingBuilder().initiate_mapping(reactions_list)


def get_available_mappers():
    """ Returns a dictionary with the available mappers and some info"""
    return {f: additional_info['info'] for f, additional_info in MapperFactory.mappers.items()}
