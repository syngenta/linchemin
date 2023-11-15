import abc

import linchemin.cheminfo.functions as cif
from linchemin.cheminfo.models import ChemicalEquation, Molecule
from linchemin.utilities import console_logger
from typing import Type

"""
Module containing functions and classes for computing score and metrics of single nodes of a route.
"""

logger = console_logger(__name__)


class NoMapping(Exception):
    """Raised if the atom mapping is absent but needed to compute a descriptor"""

    pass


class BadMapping(Exception):
    """Raised if the atom mapping has some issues"""

    pass


class ChemicalEquationDescriptor(metaclass=abc.ABCMeta):
    """Definition of the abstract class for ChemicalEquationDescriptor."""

    @abc.abstractmethod
    def compute_ce_descriptor(self, reaction: ChemicalEquation):
        pass


class ChemicalEquationDescriptorsFactory:
    """A factory class for accessing the calculation of ChemicalEquationDescriptors.

    The factory uses a registry of available ChemicalEquationDescriptors. To
    register a new descriptor, the class should be decorated
    with the `register_ce_descriptor` decorator.
    """

    _ce_descriptors = {}

    @classmethod
    def register_ce_descriptor(cls, name: str):
        """
        Decorator for registering a new ChemicalEquation descriptor.

        Parameters:
        ------------
        name: str
            The name of the descriptor to be used as a key in the registry.

        Returns:
        ----------
        function: The decorator function.
        """

        def decorator(ce_descriptor_class: Type[ChemicalEquationDescriptor]):
            cls._ce_descriptors[name.lower()] = ce_descriptor_class
            return ce_descriptor_class

        return decorator

    @classmethod
    def list_ce_descriptors(cls):
        """List the names of all available ChemicalEquation descriptors.

        Returns:
        ---------
        list: The names of the available descriptors.
        """
        return list(cls._ce_descriptors.keys())

    @classmethod
    def get_ce_descriptor_instance(cls, name: str) -> ChemicalEquationDescriptor:
        """Get an instance of the specified ChemicalEquationDescriptor.

        Parameters:
        ------------
        name: str
            The name of the ChemicalEquation descriptor.

        Returns:
        ---------
        ChemicalEquationDescriptor: An instance of the specified ChemicalEquation descriptor.

        Raises:
        --------
        KeyError: If the specified descriptor is not registered.
        """
        ce_descriptor = cls._ce_descriptors.get(name.lower())
        if ce_descriptor is None:
            logger.error(f"ChemicalEquation descriptor '{name}' not found")
            raise KeyError
        return ce_descriptor()


@ChemicalEquationDescriptorsFactory.register_ce_descriptor("ce_hypsicity")
class CEHypsicity(ChemicalEquationDescriptor):
    """Subclass to compute the hypsicity of a ChemicalEquation as the absolute value of the change in the oxidation
    state of each atom.
    """

    def compute_ce_descriptor(self, reaction: ChemicalEquation) -> float:
        """Takes a ChemicalEquation instance and returns its hypsicity, i.e., the change in the oxidation state of
        the atoms"""

        if reaction.mapping is None:
            logger.error("No atom mapping: the descriptor cannot be computed")
            raise NoMapping

        # select the desired product and add the oxidation number property to its atoms
        desired_product = reaction.get_products()[0]
        cif.compute_oxidation_numbers(desired_product.rdmol_mapped)
        desired_product_at = [
            at
            for at in reaction.mapping.atom_transformations
            if at.product_uid == desired_product.uid
        ]
        # identify the involved reactants and add the oxidation number property to their atoms
        reactants_uid = [at.reactant_uid for at in desired_product_at]
        reactants = [
            mol for mol in reaction.get_reactants() if mol.uid in reactants_uid
        ]
        for r in reactants:
            cif.compute_oxidation_numbers(r.rdmol_mapped)
        return self.compute_oxidation_state_change(
            desired_product_at, desired_product, reactants
        )

    @staticmethod
    def compute_oxidation_state_change(
        atom_transformations: list, desired_product: Molecule, reactants: list
    ) -> float:
        """Computes the change in oxidation state for each mapped atom in the ChemicalEquation as the difference
        between the oxidation number of an atom in the desired product and the corresponding atom in the reactants.
        """
        delta = 0.0
        # ox_nrs = []
        for at in atom_transformations:
            p_atom = next(
                (
                    atom
                    for atom in desired_product.rdmol_mapped.GetAtoms()
                    if atom.GetIdx() == at.prod_atom_id
                ),
                None,
            )

            reactant = next(
                (reac for reac in reactants if reac.uid == at.reactant_uid), None
            )

            r_atom = next(
                (
                    atom
                    for atom in reactant.rdmol_mapped.GetAtoms()
                    if atom.GetIdx() == at.react_atom_id
                ),
                None,
            )
            if p_atom.GetSymbol() != r_atom.GetSymbol():
                logger.error("problem with mapping")
                raise BadMapping
            # ox_nrs.append((prod_ox, react_ox))
            delta += p_atom.GetIntProp("_OxidationNumber") - r_atom.GetIntProp(
                "_OxidationNumber"
            )

        # print(ox_nrs)
        return delta


@ChemicalEquationDescriptorsFactory.register_ce_descriptor("ce_atom_effectiveness")
class CEAtomEffectiveness(ChemicalEquationDescriptor):
    """Subclass of atom efficiency at ChemicalEquation level"""

    def compute_ce_descriptor(self, reaction: ChemicalEquation) -> float:
        """Takes a ChemicalEquation instance and computes the atom efficiency as the ratio between the number of
        mapped atoms in the desired product and the number of atom in the reactants.
        """
        if reaction.mapping is None:
            logger.error("No atom mapping: the descriptor cannot be computed")
            raise NoMapping

        # compute the number of mapped atoms in the desired product
        desired_product = reaction.get_products()[0]
        n_atoms_prod = len(
            [
                a
                for a in desired_product.rdmol_mapped.GetAtoms()
                if a.GetAtomMapNum() not in [0, -1]
            ]
        )
        # compute the number of atoms in all the reactants
        reactants = reaction.get_reactants()
        n_atoms_reactants = 0
        for reactant in reactants:
            stoich = next(
                n
                for h, n in reaction.stoichiometry_coefficients["reactants"].items()
                if h == reactant.uid
            )
            n_atoms_reactants += reactant.rdmol_mapped.GetNumAtoms() * stoich
        return n_atoms_prod / n_atoms_reactants


@ChemicalEquationDescriptorsFactory.register_ce_descriptor("ce_convergence")
class CEConvergentDisconnection(ChemicalEquationDescriptor):
    """Subclass to compute the Convergent Disconnection Score.
    https://pubs.acs.org/doi/10.1021/acs.jcim.1c01074"""

    def compute_ce_descriptor(self, reaction: ChemicalEquation) -> float:
        """Takes a ChemicalEquation instance and compute the Convergent Disconnection Score [0, 1].
        The closer the score is to 1, the more balanced is the reaction.
        """

        # Retrieve list of products and reactants of the input reaction
        products = reaction.get_products()
        reactants = reaction.get_reactants()
        if len(reactants) == 1:
            return 1

        prod_n_atoms = [p.rdmol_mapped.GetNumAtoms() for p in products]
        reacs_n_atoms = [r.rdmol_mapped.GetNumAtoms() for r in reactants]
        scale_factor = prod_n_atoms[0] / len(reactants)
        abs_error = [abs(r - scale_factor) for r in reacs_n_atoms]
        return 1 / (1 + sum(abs_error) / len(abs_error))


def chemical_equation_descriptor_calculator(
    reaction: ChemicalEquation, descriptor: str
) -> float:
    """
    To compute a descriptor of a ChemicalEquation.

    Parameters:
    -------------
    node: ChemicalEquation
        The reaction for which the descriptor should be computed
    descriptor: str
        Which descriptor should be computed

    Returns:
    --------
    descriptor: float
        The descriptor value


    Raises:
    -------
    TypeError if the input reaction is not a ChemicalEquation object

    Example:
    --------
    >>> smile = '[CH3:1][C:2]([CH3])=[O:3]>>[CH3:1][C:2]([OH])=[O:3]'
    >>> chemical_equation_constructor = ChemicalEquationConstructor(molecular_identity_property_name='smiles')
    >>> reaction = chemical_equation_constructor2.build_from_reaction_string(reaction_string=smile, inp_fmt='smiles')
    >>> d = chemical_equation_descriptor_calculator(reaction, 'ce_atom_effectivness')
    """
    if type(reaction) != ChemicalEquation:
        raise TypeError(
            "ChemicalEquation descriptors can be computed only on ChemicalEquation instances."
        )

    ce_descriptor = ChemicalEquationDescriptorsFactory.get_ce_descriptor_instance(
        descriptor
    )

    return ce_descriptor.compute_ce_descriptor(reaction)
