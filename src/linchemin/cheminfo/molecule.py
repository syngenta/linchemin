# Standard library imports
from typing import Dict, List, Tuple

# Local imports
import linchemin.cheminfo.functions as cif
import linchemin.utilities as utilities

# Third party imports


class Molecule:
    """Class holding information of chemical compounds.

    Attributes:

        identity_property_name: a string indicating which kind of input string determines the identity
                            of the object (e.g. 'smiles')

        rdmol: an rdkit Mol object

        uid: the hash key identifying the Molecule instance

        smiles: the smiles string associated with the Molecule instance

        rdmol: the rdkit Mol object associated with the Molecule instance

    """

    def __init__(self, rdmol: cif.Mol, identity_property_name: str):
        # acceptable values for identity_property_name are smiles, inchikey_KET_15T, inchi_key
        # TODO: enforce by enum!
        rdmol_mapped = rdmol
        self.identity_property_name = identity_property_name

        rdmol_unmapped = cif.remove_rdmol_atom_mapping(rdmol=rdmol_mapped)
        # rdmol_unmapped_canonical, log = cif.canonicalize_rdmol(rdmol_unmapped)
        rdmol_unmapped_canonical = cif.canonicalize_rdmol_lite(
            rdmol=rdmol_unmapped, is_pattern=False
        )
        rdmol_mapped_canonical = cif.canonicalize_rdmol_lite(
            rdmol=rdmol_mapped, is_pattern=False
        )
        self.rdmol = rdmol_unmapped_canonical
        self.rdmol_mapped = rdmol_mapped_canonical
        self.hash_map = self.calculate_hash_values()

        self.smiles = cif.compute_mol_smiles(rdmol=self.rdmol)

        self.identity_property = self.hash_map.get(identity_property_name)
        self.uid = utilities.create_hash(
            self.identity_property
        )  # the hashed identity property

    def __hash__(self) -> int:
        return self.uid

    def __eq__(self, other) -> bool:
        return type(self) == type(other) and self.__hash__() == other.__hash__()

    def __str__(self) -> str:
        return f"{self.smiles}"

    def to_dict(self) -> dict:
        """To return a dictionary with all the attributes of the Molecule instance"""
        return {
            "type": "Molecule",
            "uid": self.uid,
            "smiles": self.smiles,
            "hash_map": self.hash_map,
        }

    def calculate_hash_values(self):
        """To compute the hash key of the Molecule instance"""
        return cif.calculate_molecular_hash_values(
            rdmol=self.rdmol,
            hash_list=["CanonicalSmiles", "inchi_key", "inchi_KET_15T"],
        )


class MoleculeConstructor:
    """Class implementing the constructor of the Molecule class

    Attributes:
        identity_property_name: a string indicating which kind of input string determines the identity
                                of the object (e.g. 'smiles')

    """

    def __init__(self, identity_property_name: str):
        self.identity_property_name = identity_property_name

    def build_from_molecule_string(
        self,
        molecule_string: str,
        inp_fmt: str,
    ) -> Molecule:
        """To build a Molecule instance from a string"""
        rdmol_input = cif.rdmol_from_string(
            input_string=molecule_string, inp_fmt=inp_fmt
        )
        return Molecule(
            rdmol=rdmol_input, identity_property_name=self.identity_property_name
        )

    def build_from_rdmol(self, rdmol: cif.Mol) -> Molecule:
        """To build a Molecule instance from an rdkit Mol instance"""
        return Molecule(rdmol=rdmol, identity_property_name=self.identity_property_name)
