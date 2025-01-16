from typing import Union

import linchemin.cheminfo.functions as cif
from linchemin import settings, utilities
from linchemin.cheminfo.chemical_hashes import (
    calculate_molecular_hash_map,
    calculate_reaction_like_hash_map,
    validate_molecular_identifier,
    validate_reaction_identifier,
)
from linchemin.cheminfo.model_constructors.disconnection_constructor import (
    create_disconnection,
)
from linchemin.cheminfo.model_constructors.ratam_constructor import create_ratam
from linchemin.cheminfo.models import ChemicalEquation, Molecule, ReactionComponents

logger = utilities.console_logger(__name__)


class UnparsableReaction(Exception):
    """To be raised if the input reaction string cannot be parsed"""


class UnparsableMolecule(Exception):
    """To be raised if the input molecule string cannot be parsed"""


class InvalidMoleculeInput(Exception):
    """To be raised if the input molecule is not an rdkit Mol object"""


# Molecule Constructor
class MoleculeConstructor:
    """
    Class implementing the constructor of the Molecule class

    Attributes:
    ------------
    molecular_identity_property_name: a string indicating which kind of input string determines the identity
                                      of the object (e.g. 'smiles')
    hash_list: a list containing the additional hash values to be computed

    """

    def __init__(
        self,
        molecular_identity_property_name: str = settings.CONSTRUCTORS.molecular_identity_property_name,
        hash_list: list = settings.CONSTRUCTORS.molecular_hash_list,
    ):
        validate_molecular_identifier(
            molecular_identifier=molecular_identity_property_name
        )

        self.molecular_identity_property_name = molecular_identity_property_name
        self.hash_list = set(hash_list + [self.molecular_identity_property_name])

    def build_from_molecule_string(
        self, molecule_string: str, inp_fmt: str
    ) -> Molecule:
        """
        Build a Molecule instance from a string representation.

        Args:
            molecule_string (str): String representation of the molecule
            inp_fmt (str): Input format of the molecule string

        Returns:
            Molecule: Constructed Molecule instance

        Raises:
            UnparsableMolecule: If the molecule string cannot be parsed
        """
        try:
            rdmol_input = cif.rdmol_from_string(
                input_string=molecule_string, inp_fmt=inp_fmt
            )
            return self.build_from_rdmol(rdmol_input)
        except ValueError as e:
            logger.error(f"Invalid input format: {inp_fmt}. Error: {e}")
            raise UnparsableMolecule(f"Invalid input format: {inp_fmt}")
        except Exception as e:
            logger.error(
                f"Failed to parse molecule string: {molecule_string}. Error: {e}"
            )
            raise UnparsableMolecule(f"Failed to parse molecule string: {str(e)}")

    def build_from_rdmol(self, rdmol: cif.Mol) -> Molecule:
        """
        Build a Molecule instance from a rdkit Mol instance.

        Args:
            rdmol (Chem.Mol): RDKit Mol object

        Returns:
            Molecule: Constructed Molecule instance

        Raises:
            InvalidMoleculeInput: If the input RDKit Mol object is invalid
        """
        if not isinstance(rdmol, cif.Mol):
            raise InvalidMoleculeInput("Input must be a valid RDKit Mol object")

        rdmol_mapped = rdmol
        rdmol_unmapped = cif.remove_rdmol_atom_mapping(rdmol=rdmol_mapped)

        rdmol_unmapped_canonical = cif.new_molecule_canonicalization(rdmol_unmapped)
        rdmol_mapped_canonical = cif.new_molecule_canonicalization(rdmol_mapped)

        hash_map = calculate_molecular_hash_map(
            rdmol=rdmol_unmapped_canonical,
            hash_list=self.hash_list,
        )
        identity_property = hash_map.get(self.molecular_identity_property_name)
        uid = utilities.create_hash(identity_property)
        smiles = cif.compute_mol_smiles(rdmol=rdmol_unmapped_canonical)

        return Molecule(
            rdmol=rdmol_unmapped_canonical,
            rdmol_mapped=rdmol_mapped_canonical,
            molecular_identity_property_name=self.molecular_identity_property_name,
            hash_map=hash_map,
            smiles=smiles,
            uid=uid,
            identity_property=identity_property,
        )


class ChemicalEquationConstructor:
    """
    Class implementing the constructor of the ChemicalEquation class

    Attributes:
    ---------------
    molecular_identity_property_name: a string indicating the property determining the identity
                                      of the molecules in the chemical equation (e.g. 'smiles')
    chemical_equation_identity_name: a string indicating the components of the chemical equation
                                     participating in the definition of its uid (e.g., 'r_p' to
                                     include only reactants and products; 'r_r_p' to include also
                                     reagents
    """

    desired_product: Molecule
    molecule_constructor: MoleculeConstructor
    original_reaction_components: ReactionComponents
    chemical_equation: ChemicalEquation

    def __init__(
        self,
        molecular_identity_property_name: str = settings.CONSTRUCTORS.molecular_identity_property_name,
        chemical_equation_identity_name: str = settings.CONSTRUCTORS.chemical_equation_identity_name,
    ) -> None:
        validate_molecular_identifier(
            molecular_identifier=molecular_identity_property_name
        )
        self.molecular_identity_property_name = molecular_identity_property_name
        validate_reaction_identifier(
            reaction_identifier=chemical_equation_identity_name
        )
        self.chemical_equation_identity_name = chemical_equation_identity_name
        self.molecule_constructor = MoleculeConstructor(
            self.molecular_identity_property_name
        )

    def build_from_rdrxn(
        self,
        rdrxn: cif.rdChemReactions.ChemicalReaction,
        desired_product: Union[cif.Mol, None] = settings.CONSTRUCTORS.desired_product,
    ) -> ChemicalEquation:
        """To build a ChemicalEquation instance from a rdkit ChemicalReaction object"""
        components = cif.rdrxn_to_molecule_catalog(rdrxn, self.molecule_constructor)
        self.original_reaction_components = ReactionComponents.from_dict(components)
        if not desired_product:
            self.desired_product = cif.get_heaviest_mol(
                self.original_reaction_components.products
            )
        else:
            self.desired_product = self.molecule_constructor.build_from_rdmol(
                desired_product
            )

        return self._unpack_rdrxn()

    def build_from_reaction_string(
        self,
        reaction_string: str,
        inp_fmt: str,
        desired_product: Union[str, None] = settings.CONSTRUCTORS.desired_product,
    ) -> ChemicalEquation:
        """To build a ChemicalEquation instance from a reaction string"""
        try:
            rdrxn = cif.rdrxn_from_string(input_string=reaction_string, inp_fmt=inp_fmt)
        except Exception as e:
            logger.error(f"String {reaction_string} could not be parsed. Error: {e}")
            raise UnparsableReaction
        if desired_product:
            desired_product = cif.rdmol_from_string(desired_product, inp_fmt=inp_fmt)

        return self.build_from_rdrxn(rdrxn=rdrxn, desired_product=desired_product)

    def _unpack_rdrxn(self) -> ChemicalEquation:
        if cif.is_mapped_molecule(self.desired_product.rdmol_mapped):
            return self._build_from_mapped_reaction()
        return self._build_from_unmapped_reaction()

    def _build_from_unmapped_reaction(self) -> ChemicalEquation:
        self.chemical_equation = ChemicalEquation()
        self.chemical_equation.catalog = (
            self.original_reaction_components.get_molecule_catalog()
        )
        self.chemical_equation.role_map = (
            self.original_reaction_components.get_role_map()
        )
        self.chemical_equation.stoichiometry_coefficients = (
            self.original_reaction_components.get_stoichiometry_coefficients()
        )
        self._populate_ce()
        self.chemical_equation.mapping = None
        self.chemical_equation.disconnection = None
        return self.chemical_equation

    def _build_from_mapped_reaction(self) -> ChemicalEquation:
        self.chemical_equation = ChemicalEquation()
        self.chemical_equation.catalog = (
            self.original_reaction_components.get_molecule_catalog()
        )
        self.chemical_equation.mapping = create_ratam(
            self.original_reaction_components, self.desired_product
        )
        self.chemical_equation.role_map = self.chemical_equation.mapping.get_role_map()
        self.chemical_equation.stoichiometry_coefficients = (
            self.chemical_equation.mapping.get_stoichiometry_coefficients()
        )
        self._populate_ce()
        self.chemical_equation.disconnection = create_disconnection(
            self.chemical_equation, self.desired_product
        )
        return self.chemical_equation

    def _populate_ce(self) -> None:
        use_reagents = self.chemical_equation_identity_name not in ["r_p", "u_r_p"]
        use_mapping = self.chemical_equation.mapping is not None
        self.chemical_equation.rdrxn = self._build_rdrxn(
            use_reagents=use_reagents, use_mapping=use_mapping
        )
        self.chemical_equation.smiles = self._get_reaction_smiles(use_mapping)
        self.chemical_equation.hash_map = calculate_reaction_like_hash_map(
            self.chemical_equation.catalog, self.chemical_equation.role_map
        )
        self.chemical_equation.uid = self.chemical_equation.hash_map.get(
            self.chemical_equation_identity_name
        )

    def _build_rdrxn(
        self, use_reagents: bool, use_mapping: bool
    ) -> cif.rdChemReactions.ChemicalReaction:
        """Build an RDKit ChemicalReaction object."""
        return cif.build_rdrxn(
            catalog=self.chemical_equation.catalog,
            role_map=self.chemical_equation.role_map,
            stoichiometry_coefficients=self.chemical_equation.stoichiometry_coefficients,
            use_reagents=use_reagents,
            use_smiles=False,
            use_atom_mapping=use_mapping,
            mapping=self.chemical_equation.mapping,
        )

    def _get_reaction_smiles(self, use_mapping: bool) -> str:
        """Get the SMILES representation of the reaction."""
        return cif.rdrxn_to_string(
            rdrxn=self.chemical_equation.rdrxn,
            out_fmt="smiles",
            use_atom_mapping=use_mapping,
        )
