from dataclasses import dataclass, field
from collections import namedtuple

import linchemin.cheminfo.functions as cif

"""
Module containing the definition of all relevant cheminformatics classes
"""


@dataclass
class Molecule:
    """ Class holding information of a chemical compound """
    smiles: str = field(default_factory=str)
    """ A string indicating the canonical smiles of the molecule"""
    identity_property_name: str = field(default_factory=str)
    """A string indicating the name of the property definying the identity of the molecule"""
    uid: int = field(default_factory=int)
    """ The integer obtained by hashing the identity property of the molecule"""
    hash_map: dict = field(default_factory=dict)
    """ A dictionary mapping various properties (smiles, inchi_key, ety..) with their values"""
    rdmol: cif.Mol | None = field(default=None)
    """ An unmapped RDKit Molecule object"""
    rdmol_mapped: cif.Mol | None = field(default=None)
    """ A mapped RDKit Molecule object (if any) """
    identity_property: str = field(default_factory=str)
    """ The identity property of the molecule"""

    def __hash__(self) -> int:
        return self.uid

    def __eq__(self, other) -> bool:
        return type(self) == type(other) and self.__hash__() == other.__hash__()

    def __str__(self) -> str:
        return f'{self.smiles}'

    def to_dict(self) -> dict:
        """ To return a dictionary with all the attributes of the Molecule instance """
        return {'type': 'Molecule', 'uid': self.uid, 'smiles': self.smiles, 'hash_map': self.hash_map}


@dataclass
class Disconnection:
    """ Class holding information of a disconnection """
    molecule: Molecule | None = None
    rdmol: cif.Mol | None = None
    rdmol_fragmented: cif.Mol | None = None
    reacting_atoms: list = field(default_factory=list)
    new_bonds: list = field(default_factory=list)
    modified_bonds: list = field(default_factory=list)
    hash_map: dict = field(default_factory=dict)
    uid: int = field(default_factory=int)
    identity_property: str = field(default_factory=str)

    def __hash__(self) -> int:
        return self.uid

    def __eq__(self, other) -> bool:
        return type(self) == type(other) and self.__hash__() == other.__hash__()

    def __str__(self) -> str:
        return f'{self.identity_property}'

    def to_dict(self) -> dict:
        return {'type': 'Disconnection', 'uid': self.uid,
                'hash_map': self.hash_map,
                }


@dataclass
class Pattern:
    """ Class holding information about a molecular pattern """
    rdmol_mapped: cif.Mol | None = None
    identity_property_name: str = 'smarts'
    rdmol: cif.Mol | None = None
    smarts: str = field(default_factory=str)
    hash_map: dict = field(default_factory=dict)
    uid: int = field(default_factory=int)
    identity_property: str = field(default_factory=str)

    def __hash__(self) -> int:
        return self.uid

    def __eq__(self, other) -> bool:
        return type(self) == type(other) and self.__hash__() == other.__hash__()

    def __str__(self) -> str:
        return f'{self.smarts}'

    def to_dict(self) -> dict:
        return {'type': 'Molecule', 'uid': self.uid, 'smarts': self.smarts, 'hash_map': self.hash_map}


@dataclass
class Template:
    """ Class holding information of a reaction template """
    pattern_catalog: dict = field(default_factory=dict)
    role_map: dict = field(default_factory=dict)
    stoichiometry_coefficients: dict = field(default_factory=dict)
    rdchiral_data: dict = field(default_factory=dict)
    hash_map: dict = field(default_factory=dict)
    uid: int = field(default_factory=int)
    rdrxn: cif.rdChemReactions.ChemicalReaction = field(default=cif.rdChemReactions.ChemicalReaction)
    smarts: str = field(default_factory=str)
    identity_property: str = field(default_factory=str)

    def __hash__(self) -> int:
        return self.uid

    def __eq__(self, other):
        return type(self) == type(other) and self.__hash__() == other.__hash__()

    def __str__(self) -> str:
        return f'{self.smarts}'

    def to_dict(self) -> dict:
        return {'type': 'Template', 'uid': self.uid,
                'smarts': self.smarts,
                'hash_map': self.hash_map,
                'role_map': self.role_map,
                'rdchiral_data': self.rdchiral_data,
                'stoichiometry_coefficients': self.stoichiometry_coefficients}

    def build_reaction_smiles(self) -> str:
        return '>'.join(['.'.join([self.role_map.get(uid).smiles for uid in self.role_map[role]]) for role in
                         ['reactants', 'reagents', 'products']])

    def build_rdrxn(self, use_atom_mapping=True):
        return cif.build_rdrxn(catalog=self.pattern_catalog,
                               role_map=self.role_map,
                               stoichiometry_coefficients=self.stoichiometry_coefficients,
                               use_smiles=False,
                               use_atom_mapping=use_atom_mapping)


@dataclass
class Ratam:
    """ Dataclass holding information of a reaction atom-to-atom mapping """
    full_map_info: dict = field(default_factory=dict)
    """ A dictionary mapping Molecule identifiers to a list of mapping dictionary in the form {atom_id: atom_map_number}"""
    atom_transformations: list = field(default_factory=list)
    """ a list of AtomTransformations namedtuples corresponding to the transformations of mapped atoms.
        AtomTransformation = namedtuple('AtomTransformation', ['product_uid', 'reactant_uid', 'prod_atom_id', 'react_atom_id',
                                                       'map_num']) """

    def diagnosis(self):
        """ To perform a diagnostic process on the atom mapping"""
        pass


@dataclass
class ChemicalEquation:
    """ Dataclass holding information of a chemical reaction """
    catalog: dict = field(default_factory=dict)
    """ A dictionary mapping the unique identifiers to the Molecule instances involved in the reaction """
    role_map: dict = field(default_factory=dict)
    """ A dictionary mapping each role (reactants, reagents, products) to a sorted list of Molecule's uids"""
    stoichiometry_coefficients: dict = field(default_factory=dict)
    """ A nested dictionary mapping each role to a dictionary containing the uids of the Molecules with that role
        and their stoichiometry coefficients {'role': {uid: n}}"""
    hash_map: dict = field(default_factory=dict)
    """ A dictionary mapping various properties with their values"""
    uid: int = field(default_factory=int)
    """ The integer obtained by hashing the identity property of the molecule"""
    rdrxn: cif.rdChemReactions.ChemicalReaction | None = None
    """ An RDKit ChemicalReaction object """
    smiles: str = field(default_factory=str)
    """ The standardized smiles of the reaction"""
    mapping: Ratam | None = field(default=None)
    """ A Ratam instance (if the ChemicalEquation is mapped)"""
    template: Template | None = field(default=None)
    """ A Template instance (if the ChemicalEquation is mapped)"""
    disconnection: Disconnection | None = field(default=None)
    """ A Disconnection instance (if the ChemicalEquation is mapped)"""

    def __hash__(self) -> int:
        return self.uid

    def __eq__(self, other):
        return type(self) == type(other) and self.__hash__() == other.__hash__()

    def __str__(self) -> str:
        return f'{self.smiles}'

    def build_reaction_smiles(self) -> str:
        """ To build the reaction smiles from the smiles of the involved Molecule instances """
        return '>'.join(['.'.join([self.catalog.get(uid).smiles for uid in self.role_map[role]]) for role in
                         ['reactants', 'reagents', 'products']])

    def to_dict(self) -> dict:
        """ To return a dictionary with all the attributes of the ChemicalEquation instance """
        return {'type': 'ChemicalEquation',
                'uid': self.uid, 'smiles': self.smiles,
                'hash_map': self.hash_map, 'roles': self.role_map,
                'stoichiometry_coefficients': self.stoichiometry_coefficients,
                'template': self.template.to_dict() if self.template else None,
                'disconnection': self.disconnection.to_dict() if self.disconnection else None}

    def build_rdrxn(self, use_atom_mapping=True):
        """ To build the rdkit ChemicalReaction object associated with the ChemicalEquation instance """
        return cif.build_rdrxn(catalog=self.catalog,
                               role_map=self.role_map,
                               stoichiometry_coefficients=self.stoichiometry_coefficients,
                               use_smiles=False,
                               use_atom_mapping=use_atom_mapping)
