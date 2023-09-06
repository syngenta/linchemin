import abc
from dataclasses import dataclass, field
from typing import Type, Union

import linchemin.cheminfo.functions as cif
from linchemin.cgu.convert import converter
from linchemin.cgu.syngraph import (BipartiteSynGraph, MonopartiteMolSynGraph,
                                    MonopartiteReacSynGraph)
from linchemin.cheminfo.constructors import ChemicalEquationConstructor
from linchemin.cheminfo.functions import Descriptors
from linchemin.utilities import console_logger

"""
Module containing all classes and functions to compute route's step descriptors 
"""

logger = console_logger(__name__)


class NodeNotPresent(KeyError):
    """Raised when the selected node is not present in the input route"""

    pass


class WrongSmilesType(TypeError):
    """Raised when a smiles od^f the wrong typeis given"""

    pass


@dataclass
class DescriptorCalculationOutput:
    """Class to store the outcome of the calculation of descriptors.

    Attributes:
    ------------
    descriptor_value: a float
        It corresponds to the descriptor value

    additional_info: a dictionary possibly containing additional information about the descriptor
    """

    descriptor_value: Union[float, None] = None
    additional_info: dict = field(default_factory=dict)


class StepDescriptor(metaclass=abc.ABCMeta):
    """Abstract class for StepDescriptorCalculators"""

    @abc.abstractmethod
    def compute_step_descriptor(
        self, unique_reactions, all_transformations, target, step
    ) -> DescriptorCalculationOutput:
        pass

    @staticmethod
    def extract_all_atomic_paths(desired_product, all_transformations: list) -> list:
        """To identify the atomic paths starting from the desired product."""
        # identify all the atom transformations that involve atoms of the desired product
        target_transformations = set(
            at for at in all_transformations if at.product_uid == desired_product.uid
        )
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
        """
        Decorator for registering a new step descriptor.

        Parameters:
        ------------
        name: str
            The name of the step descriptor to be used as a key in the registry.

        Returns:
        ----------
        function: The decorator function.
        """

        def decorator(step_descriptor_class: Type[StepDescriptor]):
            cls._step_descriptors[name.lower()] = step_descriptor_class
            return step_descriptor_class

        return decorator

    @classmethod
    def list_step_descriptors(cls):
        """List the names of all available step descriptors.

        Returns:
        ---------
        list: The names of the step descriptors.
        """
        return list(cls._step_descriptors.keys())

    @classmethod
    def get_step_descriptor_instance(cls, name: str) -> StepDescriptor:
        """Get an instance of the specified StepDescriptor.

        Parameters:
        ------------
        name: str
            The name of the step descriptor.

        Returns:
        ---------
        StepDescriptor: An instance of the specified step descriptor.

        Raises:
        --------
        KeyError: If the specified descriptor is not registered.
        """
        step_descriptor = cls._step_descriptors.get(name.lower())
        if step_descriptor is None:
            logger.error(f"Step descriptor '{name}' not found")
            raise KeyError
        return step_descriptor()


@StepDescriptorsFactory.register_step_descriptor("step_effectiveness")
class StepEffectiveness(StepDescriptor):
    """Subclass to compute the atom effectiveness of the step. Currently computed as the ratio between the number
    of atoms in the step reactants that contribute to the final target and the total number of atoms in the step reactants
    """

    def compute_step_descriptor(
        self, unique_reactions: set, all_transformations: list, target, step
    ) -> DescriptorCalculationOutput:
        out = DescriptorCalculationOutput()
        all_atomic_paths = self.extract_all_atomic_paths(target, all_transformations)

        contributing_atoms = sum(
            1
            for ap in all_atomic_paths
            if list(
                filter(
                    lambda at: at.reactant_uid in list(step.role_map["reactants"]), ap
                )
            )
        )
        reactants = [
            reac for h, reac in step.catalog.items() if h in step.role_map["reactants"]
        ]
        # n_atoms_reactants = sum(r.GetNumAtoms() for r.rdmol_mapped in reactants)
        n_atoms_reactants = 0
        for reactant in reactants:
            stoich = next(
                n
                for h, n in step.stoichiometry_coefficients["reactants"].items()
                if h == reactant.uid
            )
            n_atoms_reactants += reactant.rdmol_mapped.GetNumAtoms() * stoich
        # out.descriptor_value = contributing_atoms / n_atoms_prod
        out.descriptor_value = contributing_atoms / n_atoms_reactants

        out.additional_info["contributing_atoms"] = contributing_atoms
        return out


@StepDescriptorsFactory.register_step_descriptor("step_hypsicity")
class StepHypsicity(StepDescriptor):
    """Subclass to compute the hypsicty of the step"""

    def compute_step_descriptor(
        self, unique_reactions, all_transformations, target, step
    ):
        """It initializes the calculation of the step hypsicity"""
        out = DescriptorCalculationOutput()
        cif.compute_oxidation_numbers(target.rdmol_mapped)
        # all the atomic paths along the route for the mapped atoms in the target are extracted
        all_atomic_paths = self.extract_all_atomic_paths(target, all_transformations)
        delta = 0.0
        ox_nrs = []
        # for each atomic path
        for ap in all_atomic_paths:
            # if the path passes through the considered step...
            if leaf_transformation := list(
                filter(
                    lambda at: at.reactant_uid in list(step.role_map["reactants"]), ap
                )
            ):
                # the variation of the oxidation number for atom is computed
                d, ox_nr = self.hypsicity_calculation(
                    ap[0], leaf_transformation[0], target, step
                )
                delta += d
                ox_nrs.append(ox_nr)
        out.descriptor_value = delta
        out.additional_info["oxidation_numbers"] = ox_nrs
        return out

    @staticmethod
    def hypsicity_calculation(target_transformation, step_transformation, target, step):
        """It computes the difference between the oxidation number of an atom in the reactants of considered step
        and the same atom in the target (aligned with Andraos' definition)"""
        leaf = next(
            m
            for uid, m in step.catalog.items()
            if uid == step_transformation.reactant_uid
        )
        # the oxidation
        cif.compute_oxidation_numbers(leaf.rdmol_mapped)
        target_atom_ox = next(
            atom.GetIntProp("_OxidationNumber")
            for atom in target.rdmol_mapped.GetAtoms()
            if atom.GetIdx() == target_transformation.prod_atom_id
        )
        leaf_atom_ox = next(
            atom.GetIntProp("_OxidationNumber")
            for atom in leaf.rdmol_mapped.GetAtoms()
            if atom.GetIdx() == step_transformation.react_atom_id
        )
        # delta = abs(target_atom_ox - leaf_atom_ox)
        delta = leaf_atom_ox - target_atom_ox
        ox_nrs = (leaf_atom_ox, target_atom_ox)
        return delta, ox_nrs


def step_descriptor_calculator(
    descriptor_name: str,
    route: Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph],
    step: str,
) -> DescriptorCalculationOutput:
    """
    To compute a step descriptor.

    Parameters:
    ------------
    descriptor_name: str
        The name of the descriptor to be computed
    route: Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
        The SynGraph corresponding to the route of interest
    step: str
        The smiles of the reaction step for which the descriptor should be computed

    Returns:
    ---------
    out: DescriptorCalculationOutput
        The output of the calculation

    Raises:
    --------
    TypeError: if the input graph is not a SynGraph

    Example:
    ---------
    >>> out = step_descriptor_calculator('step_effectiveness', route_syngraph, step_ce)
    """
    if isinstance(route, (BipartiteSynGraph, MonopartiteMolSynGraph)):
        route = converter(route, "monopartite_reactions")
    elif not isinstance(route, MonopartiteReacSynGraph):
        logger.error(
            f"Step descriptors can be computed only with MonopartiteReacSynGraph or BipartiteSynGraph objects. "
            f"{type(route)} cannot be processed."
        )
        raise TypeError

    ce = build_step_ce(step)
    if ce not in route.graph:
        logger.error("The selected step is not present in the input route")
        raise NodeNotPresent

    step_descriptor = StepDescriptorsFactory.get_step_descriptor_instance(
        descriptor_name
    )
    unique_reactions = route.get_unique_nodes()
    all_transformations = [
        at for ce in unique_reactions for at in ce.mapping.atom_transformations
    ]
    d_roots = {
        mol: Descriptors.ExactMolWt(mol.rdmol) for mol in route.get_molecule_roots()
    }
    target = max(d_roots, key=d_roots.get)
    # target = route.get_molecule_roots()[0]
    return step_descriptor.compute_step_descriptor(
        unique_reactions, all_transformations, target, ce
    )


def get_available_step_descriptors():
    """It returns all the available step descriptor names"""
    return StepDescriptorsFactory.list_step_descriptors()


def build_step_ce(step_smiles: str):
    """To build the ChemicalEquation corresponding to the input step smiles"""
    if step_smiles.count(">") != 2:
        logger.error(
            "The provided smiles does not represent a reaction. Please provide a valid reaction smiles"
        )
        raise WrongSmilesType

    return ChemicalEquationConstructor().build_from_reaction_string(
        step_smiles, "smiles"
    )


def find_atom_path(
    current_transformation, all_transformations: list, path: Union[list, None] = None
):
    """To find an 'atomic path' from one of the target atoms to the starting material from which the atom arrives"""
    if path is None:
        path = []
    path += [current_transformation]
    if not (
        next_t := [
            t
            for t in all_transformations
            if t.prod_atom_id == current_transformation.react_atom_id
            and t.product_uid == current_transformation.reactant_uid
        ]
    ):
        return path
    for t in next_t:
        if new_path := find_atom_path(t, all_transformations, path):
            return new_path
