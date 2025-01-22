import abc
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Type, Union

import linchemin.cheminfo.functions as cif
from linchemin.cgu.convert import converter
from linchemin.cgu.syngraph import (
    BipartiteSynGraph,
    MonopartiteMolSynGraph,
    MonopartiteReacSynGraph,
)
from linchemin.cheminfo.constructors import ChemicalEquationConstructor
from linchemin.cheminfo.functions import Descriptors
from linchemin.cheminfo.models import AtomTransformation, ChemicalEquation, Molecule
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

    descriptor_value: float = 0.0
    additional_info: dict = field(default_factory=dict)


@dataclass
class BondInfo:
    step_bond_order: float
    target_bond_order: float
    target_atoms_id: Tuple[int, int]


class StepDescriptor(metaclass=abc.ABCMeta):
    """Abstract class for StepDescriptorCalculators"""

    @abc.abstractmethod
    def compute_step_descriptor(
        self,
        unique_reactions: Set[ChemicalEquation],
        all_transformations: List[AtomTransformation],
        target: Molecule,
        step: ChemicalEquation,
    ) -> DescriptorCalculationOutput:
        pass

    @staticmethod
    def extract_all_atomic_paths(
        desired_product: Molecule, all_transformations: List[AtomTransformation]
    ) -> List[List[AtomTransformation]]:
        """To identify the atomic paths starting from the desired product."""
        # identify all the atom transformations that involve atoms
        # of the desired product
        target_transformations = {
            at for at in all_transformations if at.product_uid == desired_product.uid
        }
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


@StepDescriptorsFactory.register_step_descriptor("step_bond_efficiency")
class StepBondEfficiency(StepDescriptor):
    """
    Class to compute the bond efficiency for a given step in the context of a route.

    The descriptor assumes value 0 in the best case scenario (all bonds formed in the step are
    present in the target molecule unchanged.)

    The higher the descriptor value, the worst.
    """

    bond_orders = {
        cif.Chem.BondType.SINGLE: 1.0,
        cif.Chem.BondType.DOUBLE: 2.0,
        cif.Chem.BondType.TRIPLE: 3.0,
        cif.Chem.BondType.AROMATIC: 1.5,
        cif.Chem.BondType.UNSPECIFIED: 0.0,
    }

    def __init__(self):
        self.target: Optional[Molecule] = None
        self.step: Optional[ChemicalEquation] = None
        self.all_transformations: Optional[List[AtomTransformation]] = None
        self.all_atomic_paths: Optional[List[List[AtomTransformation]]] = None
        self.out = DescriptorCalculationOutput()

    def compute_step_descriptor(
        self,
        unique_reactions: Set[ChemicalEquation],
        all_transformations: List[AtomTransformation],
        target: Molecule,
        step: ChemicalEquation,
    ) -> DescriptorCalculationOutput:
        self.target = target
        self.step = step
        self.all_transformations = all_transformations
        self.all_atomic_paths = self.extract_all_atomic_paths(
            target, all_transformations
        )

        disconnection_bonds = (
            self.step.disconnection.new_bonds + self.step.disconnection.modified_bonds
        )
        if not disconnection_bonds:
            # the step does not involve any bond formation/modification
            return self._handle_absence_of_disconnection_bonds()

        if self.target.uid in self.step.role_map["products"]:
            # the step corresponds to the final step of the route
            return self._process_final_step()

        return self._process_intermediate_step(disconnection_bonds)

    def _process_intermediate_step(
        self, disconnection_bonds: List[Tuple[int, int]]
    ) -> DescriptorCalculationOutput:
        step_desired_product = next(
            mol
            for h, mol in self.step.catalog.items()
            if h in self.step.role_map["products"]
        )

        for bond in disconnection_bonds:
            ap1, ap2 = self.find_bond_atomic_paths(bond, step_desired_product)
            if ap1 and ap2:
                self._process_bond(
                    bond,
                    ap1[0].prod_atom_id,
                    ap2[0].prod_atom_id,
                    step_desired_product,
                )
            else:
                self.out.additional_info[bond] = {
                    "target_bond": "atoms are not present in the target."
                }
                self.out.descriptor_value += 6

        return self.out

    def _process_bond(
        self,
        bond: Tuple[int, int],
        a1_target: int,
        a2_target: int,
        step_desired_product: Molecule,
    ) -> None:
        target_bond_order = self.get_bond_order(self.target, a1_target, a2_target)
        if target_bond_order:
            step_bond_order = self.get_bond_order(step_desired_product, *bond)
            bond_info = BondInfo(
                step_bond_order, target_bond_order, (a1_target, a2_target)
            )
            self._update_output(bond, bond_info)
        else:
            self.out.additional_info[bond] = {
                "target_bond": "not present in the target"
            }
            self.out.descriptor_value += 6

    def _process_final_step(self) -> DescriptorCalculationOutput:
        for bond in self.step.disconnection.modified_bonds:
            self._process_final_step_bond(bond)
        for bond in self.step.disconnection.new_bonds:
            self.out.additional_info[bond] = "new bond in the final step"
        return self.out

    def _process_final_step_bond(self, bond: Tuple[int, int]) -> None:
        a1_prod, a2_prod = bond
        a1_at = next(
            at
            for at in self.step.mapping.atom_transformations
            if at.prod_atom_id == a1_prod
        )
        reactant = next(
            mol for h, mol in self.step.catalog.items() if mol.uid == a1_at.reactant_uid
        )
        a2_reac = next(
            at.react_atom_id
            for at in self.step.mapping.atom_transformations
            if at.prod_atom_id == a2_prod
        )

        target_bond_order = self.get_bond_order(self.target, a1_prod, a2_prod)
        reactant_bond_order = self.get_bond_order(
            reactant, a1_at.react_atom_id, a2_reac
        )

        bond_info = BondInfo(reactant_bond_order, target_bond_order, (a1_prod, a2_prod))
        self._update_output(bond, bond_info)

    def _update_output(
        self,
        bond: Tuple[int, int],
        bond_info: BondInfo,
    ) -> None:
        self.out.additional_info[bond] = bond_info.__dict__
        self.out.descriptor_value += abs(
            bond_info.target_bond_order - bond_info.step_bond_order
        )

    def find_bond_atomic_paths(
        self,
        new_bond: Tuple[int, int],
        step_desired_product: Molecule,
    ) -> Tuple[Optional[List[AtomTransformation]], Optional[List[AtomTransformation]]]:
        a1, a2 = new_bond
        ap1 = next(
            (
                ap
                for ap in self.all_atomic_paths
                if self.atomic_path_contains_atom(ap, a1, step_desired_product.uid)
            ),
            None,
        )
        ap2 = next(
            (
                ap
                for ap in self.all_atomic_paths
                if self.atomic_path_contains_atom(ap, a2, step_desired_product.uid)
            ),
            None,
        )
        return ap1, ap2

    @staticmethod
    def atomic_path_contains_atom(
        atomic_path: List[AtomTransformation], atom_id: int, step_id: int
    ) -> bool:
        return any(
            at.prod_atom_id == atom_id and at.product_uid == step_id
            for at in atomic_path
        )

    def get_bond_order(self, molecule: Molecule, a1: int, a2: int) -> Optional[float]:
        bond = molecule.rdmol_mapped.GetBondBetweenAtoms(a1, a2)
        return self.bond_orders[bond.GetBondType()] if bond else None

    def _handle_absence_of_disconnection_bonds(self) -> DescriptorCalculationOutput:
        self.out.descriptor_value = 6
        self.out.additional_info["info"] = "no new/changed bonds in the selected step"
        return self.out


@StepDescriptorsFactory.register_step_descriptor("step_effectiveness")
class StepEffectiveness(StepDescriptor):
    """Subclass to compute the atom effectiveness of the step.
    Currently computed as the ratio between the number
    of atoms in the step reactants that contribute to the
    final target and the total number of atoms in the step reactants
    """

    def compute_step_descriptor(
        self,
        unique_reactions: Set[ChemicalEquation],
        all_transformations: List[AtomTransformation],
        target: Molecule,
        step: ChemicalEquation,
    ) -> DescriptorCalculationOutput:
        out = DescriptorCalculationOutput()
        all_atomic_paths = self.extract_all_atomic_paths(target, all_transformations)
        contributing_atoms = self.find_contributing_atoms(all_atomic_paths, step)

        n_atoms_reactants = self.get_reactants_atoms(step)

        n_contributing_atoms = sum(
            len(atoms) for mol_uid, atoms in contributing_atoms.items()
        )
        out.descriptor_value = n_contributing_atoms / n_atoms_reactants

        out.additional_info["contributing_atoms"] = contributing_atoms
        return out

    @staticmethod
    def get_reactants_atoms(step: ChemicalEquation) -> int:
        """To get the number of atoms in all the reactants of the considered step"""
        reactants = step.get_reactants()
        n_atoms_reactants = 0
        for reactant in reactants:
            stoich = next(
                n
                for h, n in step.stoichiometry_coefficients["reactants"].items()
                if h == reactant.uid
            )
            n_atoms_reactants += reactant.rdmol_mapped.GetNumAtoms() * stoich
        return n_atoms_reactants

    @staticmethod
    def find_contributing_atoms(
        all_atomic_paths: List[List[AtomTransformation]], step: ChemicalEquation
    ) -> dict:
        """To identify the atoms contributing to the target molecule"""
        contributing_atoms = {}
        for ap in all_atomic_paths:
            if atom_transformation := next(
                (at for at in ap if at.reactant_uid in step.role_map["reactants"]), None
            ):
                if atom_transformation.reactant_uid not in contributing_atoms:
                    contributing_atoms[atom_transformation.reactant_uid] = [
                        {
                            "step_atom": atom_transformation.react_atom_id,
                            "target_atom": ap[0].prod_atom_id,
                        }
                    ]
                else:
                    contributing_atoms[atom_transformation.reactant_uid].append(
                        {
                            "step_atom": atom_transformation.react_atom_id,
                            "target_atom": ap[0].prod_atom_id,
                        }
                    )
        return contributing_atoms


@StepDescriptorsFactory.register_step_descriptor("step_hypsicity")
class StepHypsicity(StepDescriptor):
    """Subclass to compute the hypsicity of the step"""

    def compute_step_descriptor(
        self,
        unique_reactions: Set[ChemicalEquation],
        all_transformations: List[AtomTransformation],
        target: Molecule,
        step: ChemicalEquation,
    ) -> DescriptorCalculationOutput:
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
        out.additional_info["contributing_oxidation_numbers"] = [
            ox_nr for ox_nr in ox_nrs if ox_nr[0] != ox_nr[1]
        ]
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

    NodeNotPresent: if the selected step does not appear in the input SynGraph

    Example:
    ---------
    >>> out = step_descriptor_calculator('step_effectiveness', route_syngraph, step_smiles)
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


def build_step_ce(step_smiles: str) -> ChemicalEquation:
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
    current_transformation: AtomTransformation,
    all_transformations: List[AtomTransformation],
    path: Union[list, None] = None,
) -> List:
    """To find an 'atomic path' from an atom in a starting material to the latest compound it ends up in"""
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
