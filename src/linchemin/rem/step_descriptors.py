import abc
from dataclasses import dataclass, field
from typing import Type, Union
from linchemin.utilities import console_logger
import linchemin.cheminfo.functions as cif
from linchemin.cgu.syngraph import MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph
from linchemin.cgu.convert import converter

"""
Module containing all classes and functions to compute route's step descriptors 
"""

logger = console_logger(__name__)


@dataclass
class DescriptorCalculationOutput:
    """ Class to store the outcome of the calculation of descriptors.

            Attributes:
            ------------
                descriptor_value: a float corresponding to the descriptor value

                additional_info: a dictionary possibly containing additional information about the descriptor
    """
    descriptor_value: Union[float, None] = None
    additional_info: dict = field(default_factory=dict)


class StepDescriptor(metaclass=abc.ABCMeta):
    """ Abstract class for StepDescriptorCalculators """

    @abc.abstractmethod
    def compute_step_descriptor(self, unique_reactions, all_transformations, target,
                                step) -> DescriptorCalculationOutput:
        pass

    @staticmethod
    def extract_all_atomic_paths(desired_product, all_transformations: list) -> list:
        """ To identify the atomic paths starting from the desired product."""
        # identify all the atom transformations that involve atoms of the desired product
        target_transformations = [at for at in all_transformations if at.product_uid == desired_product.uid]
        all_atomic_paths = []
        # One path is created from each mapped atom in the desired product to the corresponding atom in a leaf
        for t in target_transformations:
            path = find_atom_path(t, all_transformations)
            all_atomic_paths.append(path)
        return all_atomic_paths


class StepDescriptorsFactory:
    """A factory class for accessing the calculation of StepDescriptors.

        The factory uses a registry of available StepDescriptors. To
        register a new descriptor, the class should be decorated
        with the `register_step_descriptor` decorator.
    """

    _step_descriptors = {}

    @classmethod
    def register_step_descriptor(cls, name: str):
        """Decorator for registering a new step descriptor.

        :param:
            name (str): The name of the step descriptor to be used as a key in the registry.

        :return:
            function: The decorator function.
        """

        def decorator(step_descriptor_class: Type[StepDescriptor]):
            cls._step_descriptors[name.lower()] = step_descriptor_class
            return step_descriptor_class

        return decorator

    @classmethod
    def list_step_descriptors(cls):
        """List the names of all available step descriptors.

        :return:
            list: The names of the step descriptors.
        """
        return list(cls._step_descriptors.keys())

    @classmethod
    def get_step_descriptor_instance(cls, name: str) -> StepDescriptor:
        """Get an instance of the specified StepDescriptor.

        :param:
            name (str): The name of the step descriptor.

        :return:
            StepDescriptor: An instance of the specified step descriptor.

        :raise:
            KeyError: If the specified descriptor is not registered.
        """
        step_descriptor = cls._step_descriptors.get(name.lower())
        if step_descriptor is None:
            logger.error(f"Step descriptor '{name}' not found")
            raise KeyError
        return step_descriptor()


@StepDescriptorsFactory.register_step_descriptor("step_effectiveness")
class StepEffectiveness(StepDescriptor):
    """ Subclass to compute the atom effectiveness of the step """

    def compute_step_descriptor(self, unique_reactions: set,
                                all_transformations: list,
                                target,
                                step) -> DescriptorCalculationOutput:
        out = DescriptorCalculationOutput()
        n_atoms_prod = target.rdmol.GetNumAtoms()
        all_atomic_paths = self.extract_all_atomic_paths(target, all_transformations)
        contributing_atoms = sum(1 for ap in all_atomic_paths if
                                 list(filter(lambda at: at.reactant_uid in list(step.role_map['reactants']), ap)))
        out.descriptor_value = contributing_atoms / n_atoms_prod

        out.additional_info['contributing_atoms'] = contributing_atoms
        return out


@StepDescriptorsFactory.register_step_descriptor("step_hypsicity")
class StepHypsicity(StepDescriptor):
    """ Subclass to compute the hypsicty of the step """

    def compute_step_descriptor(self, unique_reactions, all_transformations, target, step):
        """ It initializes the calculation of the step hypsicity """
        out = DescriptorCalculationOutput()
        cif.compute_oxidation_numbers(target.rdmol_mapped)
        # all the atomic paths along the route for the mapped atoms in the target are extracted
        all_atomic_paths = self.extract_all_atomic_paths(target, all_transformations)
        delta = 0.0
        ox_nrs = []
        # for each atomic path
        for ap in all_atomic_paths:
            # if the path passes through the considered step...
            if leaf_transformation := list(filter(lambda at: at.reactant_uid in list(step.role_map['reactants']), ap)):
                # the variation of the oxidation number for atom is computed
                d, ox_nr = self.hypsicity_calculation(ap[0], leaf_transformation[0], target, step)
                delta += d
                ox_nrs.append(ox_nr)
        out.descriptor_value = delta
        out.additional_info['oxidation_numbers'] = ox_nrs
        return out

    @staticmethod
    def hypsicity_calculation(target_transformation, step_transformation, target, step):
        """ It computes the difference between the oxidation number of an atom in the target and the same atom in the
            considered step """
        leaf = [m for uid, m in step.catalog.items() if uid == step_transformation.reactant_uid][0]
        # the oxidation
        cif.compute_oxidation_numbers(leaf.rdmol_mapped)
        target_atom_ox = [atom.GetIntProp('_OxidationNumber') for atom in target.rdmol_mapped.GetAtoms()
                          if atom.GetIdx() == target_transformation.prod_atom_id][0]
        leaf_atom_ox = [atom.GetIntProp('_OxidationNumber') for atom in leaf.rdmol_mapped.GetAtoms()
                        if atom.GetIdx() == step_transformation.react_atom_id][0]
        delta = abs(target_atom_ox - leaf_atom_ox)
        ox_nrs = (target_atom_ox, leaf_atom_ox)
        return delta, ox_nrs


def step_descriptor_calculator(name: str,
                               route: Union[MonopartiteReacSynGraph, BipartiteSynGraph],
                               step) -> DescriptorCalculationOutput:
    """ Facade function to expose the StepDescriptor factory.

        :param:
            name: a string indicating the name of the descriptor to be computed

            route: a SynGraph object

            step: a ChemicalEquation corresponding to a step in the route

        :return:
            a DescriptorCalculationOutput instance
    """
    if isinstance(route, BipartiteSynGraph):
        route = converter(route, 'monopartite_reactions')
    elif not isinstance(route, MonopartiteReacSynGraph):
        logger.error(
            f"Step descriptors can be computed only with MonopartiteReacSynGraph or BipartiteSynGraph objects. "
            f"{type(route)} cannot be processed.")
        raise TypeError
    step_descriptor = StepDescriptorsFactory.get_step_descriptor_instance(name)
    unique_reactions = extract_unique_ce(route)
    all_transformations = [at for ce in unique_reactions for at in ce.mapping.atom_transformations]
    target = route.get_molecule_roots()[0]
    return step_descriptor.compute_step_descriptor(unique_reactions, all_transformations, target, step)


def get_available_step_descriptors():
    """ It returns all the available step descriptor names """
    return StepDescriptorsFactory.list_step_descriptors()


def extract_unique_ce(route):
    unique_ce = set()
    for parent, children in route.graph.items():
        unique_ce.add(parent)
        for child in children:
            unique_ce.add(child)
    return unique_ce


def find_atom_path(current_transformation, all_transformations, path=None):
    """ To find an 'atomic path' from one of the target atoms to the starting material from which the atom arrives"""
    if path is None:
        path = []
    path += [current_transformation]
    if not (next_t := [t for t in all_transformations if t.prod_atom_id == current_transformation.react_atom_id
                                                         and t.product_uid == current_transformation.reactant_uid]):
        return path
    for t in next_t:
        if new_path := find_atom_path(t, all_transformations, path):
            return new_path
