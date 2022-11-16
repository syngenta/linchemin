from abc import ABC, abstractmethod

"""
Module containing classes and functions for the pipeline of atom-2-atom mapping of chemical equations
"""


# Mappers factory

class Mapper(ABC):
    """ Abstract class for the atom mappers """

    @abstractmethod
    def map_chemical_equations(self):
        pass


class NameRxnMapper(Mapper):
    """ Class for the NameRxn atom mapper """

    def map_chemical_equations(self):
        print('NameRxn mapper is called')


class ChematicaMapper(Mapper):
    """ Class for the Chematica atom mapper """

    def map_chemical_equations(self):
        print('Chematica mapper is called')


class IbmRxnMapper(Mapper):
    """ Class for the IbmRxn atom mapper """

    def map_chemical_equations(self):
        print('RxnMapper mapper is called')


class MapperFactory:
    mappers = {'namerxn': NameRxnMapper,
               'chematica': ChematicaMapper,
               'rxnmapper': IbmRxnMapper}

    def call_mapper(self, mapper_name):
        """ Takes a string indicating a mapper and calls it """
        if mapper_name not in self.mappers:
            raise KeyError(f"'{mapper_name}' is not a valid mapper."
                           f"Available mappers are: {self.mappers.keys()}")

        mapper = self.mappers[mapper_name]
        return mapper().map_chemical_equations()


def perform_atom_mapping(mapper_name):
    """ Gives access to the mapper factory, which wraps each of available atom mapper."""
    factory = MapperFactory()
    return factory.call_mapper(mapper_name)


# Mapping chain

class MappingStep(ABC):
    """ Abstract handler for the concrete handlers of consecutive atom mappers """

    @abstractmethod
    def mapping(self):
        pass


class FirstMapping(MappingStep):
    """ Concrete handler to call the first mapper """

    def mapping(self):
        mapped_ce = perform_atom_mapping('namerxn')
        return SecondMapping().mapping()


class SecondMapping(MappingStep):
    """ Concrete handler to call the second mapper """

    def mapping(self):
        mapped_ce = perform_atom_mapping('chematica')
        return ThirdMapping().mapping()


class ThirdMapping(MappingStep):
    """ Concrete handler to call the third mapper """

    def mapping(self):
        mapped_ce = perform_atom_mapping('rxnmapper')
        return mapped_ce


class MappingBuilder:
    """ Class to start the chain calling the handler of the first mapper """

    @staticmethod
    def initiate_mapping():
        return FirstMapping().mapping()


def pipeline_atom_mapping():
    """ Facade function to start the atom-to-atom mapping pipeline """
    return MappingBuilder().initiate_mapping()
