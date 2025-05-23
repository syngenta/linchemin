from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import List, NamedTuple, Union

import linchemin.cheminfo.functions as cif

"""
Module containing the definition of all relevant cheminformatics classes
"""


@dataclass
class Molecule:
    """Class holding information of a chemical compound"""

    smiles: str = field(default_factory=str)
    """ (str) The canonical smiles of the molecule"""
    molecular_identity_property_name: str = field(default_factory=str)
    """(str) The name of the property definying the identity of the molecule"""
    uid: int = field(default_factory=int)
    """ (int) The unique identifier of the molecule """
    hash_map: dict = field(default_factory=dict)
    """ (dict) The dictionary mapping various properties (smiles, inchi_key, ety..) with their values"""
    rdmol: Union[cif.Mol, None] = field(default=None)
    """ (Union[cif.Mol, None]) The unmapped RDKit Molecule object"""
    rdmol_mapped: Union[cif.Mol, None] = field(default=None)
    """ (Union[cif.Mol, None]) The mapped RDKit Molecule object (if any) """
    identity_property: str = field(default_factory=str)
    """ (str) The identity property of the molecule"""

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


@dataclass
class ReactionComponents:
    reactants: List[Molecule] = field(default_factory=list)
    reagents: List[Molecule] = field(default_factory=list)
    products: List[Molecule] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def get_molecule_catalog(self):
        """To get a catalog of the unique Molecules involved in the reaction
        in the form {mol_uid: Molecule}"""
        return {mol.uid: mol for mol in self.reactants + self.reagents + self.products}

    def get_stoichiometry_coefficients(self):
        """To get the stoichiometry coefficients of each molecule involved
        the reaction in the form {'role': {uid: n}}"""
        return {
            "reactants": self._count_unique_items_by_attribute(self.reactants),
            "products": self._count_unique_items_by_attribute(self.products),
            "reagents": self._count_unique_items_by_attribute(self.reagents),
        }

    def get_role_map(self):
        """To get a dictionary mapping a list of Molecules onto their role"""
        return {
            "reactants": sorted(list({m.uid for m in set(self.reactants)})),
            "products": sorted(list({m.uid for m in set(self.products)})),
            "reagents": sorted(list({m.uid for m in set(self.reagents)})),
        }

    @staticmethod
    def _count_unique_items_by_attribute(items, attr="uid"):
        """To count the number of unique items in a list"""
        return dict(Counter(getattr(item, attr) for item in items))


class AtomTransformation(NamedTuple):
    product_uid: str
    reactant_uid: str
    prod_atom_id: int
    react_atom_id: int
    map_num: int


@dataclass
class Disconnection:
    """Class holding information of a disconnection"""

    molecule: Union[Molecule, None] = None
    rdmol: Union[cif.Mol, None] = None
    rdmol_fragmented: Union[cif.Mol, None] = None
    reacting_atoms: list = field(default_factory=list)
    hydrogenated_atoms: list = field(default_factory=list)
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
        return f"{self.identity_property}"

    def to_dict(self) -> dict:
        return {
            "type": "Disconnection",
            "uid": self.uid,
            "hash_map": self.hash_map,
        }

    def extract_info(self) -> dict:
        """To extract a dictionary containing the ids of atoms and bonds involved in the disconnection"""
        if bonds := [
            self.rdmol.GetBondBetweenAtoms(*atoms_pair).GetIdx()
            for atoms_pair in self.new_bonds + self.modified_bonds
        ]:
            return {
                "disconnection_bonds": bonds,
                "disconnection_atoms": self.reacting_atoms,
            }
        else:
            return {"disconnection_atoms": self.reacting_atoms}


@dataclass
class Pattern:
    """Class holding information about a molecular pattern"""

    rdmol_mapped: Union[cif.Mol, None] = None
    identity_property_name: str = "smarts"
    rdmol: Union[cif.Mol, None] = None
    smarts: str = field(default_factory=str)
    hash_map: dict = field(default_factory=dict)
    uid: int = field(default_factory=int)
    identity_property: str = field(default_factory=str)

    def __hash__(self) -> int:
        return self.uid

    def __eq__(self, other) -> bool:
        return type(self) == type(other) and self.__hash__() == other.__hash__()

    def __str__(self) -> str:
        return f"{self.smarts}"

    def to_dict(self) -> dict:
        return {
            "type": "Molecule",
            "uid": self.uid,
            "smarts": self.smarts,
            "hash_map": self.hash_map,
        }


@dataclass
class Template:
    """Class holding information of a reaction template"""

    pattern_catalog: dict = field(default_factory=dict)
    role_map: dict = field(default_factory=dict)
    stoichiometry_coefficients: dict = field(default_factory=dict)
    rdchiral_data: dict = field(default_factory=dict)
    hash_map: dict = field(default_factory=dict)
    uid: int = field(default_factory=int)
    rdrxn: cif.rdChemReactions.ChemicalReaction = field(
        default=cif.rdChemReactions.ChemicalReaction
    )
    smarts: str = field(default_factory=str)
    identity_property: str = field(default_factory=str)

    def __hash__(self) -> int:
        return self.uid

    def __eq__(self, other):
        return type(self) == type(other) and self.__hash__() == other.__hash__()

    def __str__(self) -> str:
        return f"{self.smarts}"

    def to_dict(self) -> dict:
        return {
            "type": "Template",
            "uid": self.uid,
            "smarts": self.smarts,
            "hash_map": self.hash_map,
            "role_map": self.role_map,
            "rdchiral_data": self.rdchiral_data,
            "stoichiometry_coefficients": self.stoichiometry_coefficients,
        }

    def build_reaction_smiles(self) -> str:
        return ">".join(
            [
                ".".join([self.role_map.get(uid).smiles for uid in self.role_map[role]])
                for role in ["reactants", "reagents", "products"]
            ]
        )

    def build_rdrxn(self, use_atom_mapping=True):
        return cif.build_rdrxn(
            catalog=self.pattern_catalog,
            role_map=self.role_map,
            stoichiometry_coefficients=self.stoichiometry_coefficients,
            use_reagents=True,
            use_smiles=False,
            use_atom_mapping=use_atom_mapping,
        )


@dataclass
class Ratam:
    """Dataclass holding information of a reaction atom-to-atom mapping"""

    full_map_info: dict = field(default_factory=dict)
    """ (dict) The dictionary mapping Molecule uids to a list of dictionaries mapping atom ids with map number"""
    atom_transformations: set = field(default_factory=set)
    """ (set) The set of AtomTransformations namedtuples corresponding to the transformations of mapped atoms.
        AtomTransformation = namedtuple('AtomTransformation', ['product_uid', 'reactant_uid', 'prod_atom_id',
        'react_atom_id', 'map_num']) """
    reactants_unmapped_atoms_info: dict = field(default_factory=dict)
    """ (dict) The dictionary containing the fraction of unmapped atoms in the reactants"""
    desired_product_unmapped_atoms_info: dict = field(default_factory=dict)
    """ (dict) The dictionary containing the fraction of unmapped atoms in the desired product and their indices"""

    def get_role_map(self) -> dict:
        return {
            role: sorted(list(map_info.keys()))
            for role, map_info in self.full_map_info.items()
        }

    def get_stoichiometry_coefficients(self) -> dict:
        stoichiometry_coefficients = {"reactants": {}, "reagents": {}, "products": {}}
        for role, mapping_info in self.full_map_info.items():
            for uid, map_list in mapping_info.items():
                stoichiometry_coefficients[role].update({uid: len(map_list)})
        return stoichiometry_coefficients

    def diagnosis(self):
        """To perform a diagnostic process on the atom mapping"""
        pass


@dataclass
class ChemicalEquation:
    """
    Dataclass holding information of a chemical reaction
    """

    catalog: dict = field(default_factory=dict)
    """(dict) The catalog of the Molecule instances involved in the chemical reaction."""
    role_map: dict = field(default_factory=dict)
    """ (dict) The dictionary mapping the uid of the involved Molecules to their roles (reactants, reagents and
    products)."""
    stoichiometry_coefficients: dict = field(default_factory=dict)
    """ (dict) The dictionary mapping each role to a dictionary containing the uids of the Molecules with that role
        and their stoichiometry coefficients {'role': {uid: n}}"""
    hash_map: dict = field(default_factory=dict)
    """ (dict) The dictionary mapping properties with hash value derived from them"""
    uid: int = field(default_factory=int)
    """ (int) The integer representing the unique identifier of the reaction."""
    rdrxn: Union[cif.rdChemReactions.ChemicalReaction, None] = None
    """ (Union[cif.rdChemReactions.ChemicalReaction, None]) The RDKit ChemicalReaction object corresponding to the
    ChemicalEquation (default None)"""
    smiles: str = field(default_factory=str)
    """ (str) The SMILES representation of the reaction."""
    mapping: Union[Ratam, None] = field(default=None)
    """ (Union[Ratam, None]) The Ratam object with the information about the atom mapping (default None, if the
    ChemicalEquation is not mapped)."""
    template: Union[Template, None] = field(default=None)
    """ (Union[Template, None]) The Template object (default None, if the ChemicalEquation is not mapped)."""
    disconnection: Union[Disconnection, None] = field(default=None)
    """ (Union[Disconnection, None]) The Disconnection object (default None, if the ChemicalEquation is not mapped)."""

    def __hash__(self) -> int:
        return self.uid

    def __eq__(self, other):
        return type(self) == type(other) and self.__hash__() == other.__hash__()

    def __str__(self) -> str:
        return f"{self.smiles}"

    def build_reaction_smiles(self, use_reagents: bool) -> str:
        """To build a reaction smiles from the smiles of the involved Molecule instances"""
        if use_reagents:
            return ">".join(
                [
                    ".".join(
                        [self.catalog.get(uid).smiles for uid in self.role_map[role]]
                    )
                    for role in ["reactants", "reagents", "products"]
                ]
            )
        else:
            return ">>".join(
                [
                    ".".join(
                        [self.catalog.get(uid).smiles for uid in self.role_map[role]]
                    )
                    for role in ["reactants", "products"]
                ]
            )

    def to_dict(self) -> dict:
        """To return a dictionary with all the attributes of the ChemicalEquation instance"""
        return {
            "type": "ChemicalEquation",
            "uid": self.uid,
            "smiles": self.smiles,
            "hash_map": self.hash_map,
            "roles": self.role_map,
            "stoichiometry_coefficients": self.stoichiometry_coefficients,
            "template": self.template.to_dict() if self.template else None,
            "disconnection": self.disconnection.to_dict()
            if self.disconnection
            else None,
        }

    def build_rdrxn(
        self, use_reagents: bool, use_atom_mapping=True
    ) -> cif.rdChemReactions.ChemicalReaction:
        """To build a rdkit ChemicalReaction object from the ChemicalEquation instance"""
        return cif.build_rdrxn(
            catalog=self.catalog,
            role_map=self.role_map,
            stoichiometry_coefficients=self.stoichiometry_coefficients,
            use_reagents=use_reagents,
            use_smiles=False,
            use_atom_mapping=use_atom_mapping,
        )

    def get_reactants(self) -> List[Molecule]:
        """To get the list of reactants"""
        return [
            mol for h, mol in self.catalog.items() if h in self.role_map["reactants"]
        ]

    def get_products(self) -> List[Molecule]:
        """To get the list of products"""
        return [
            mol for h, mol in self.catalog.items() if h in self.role_map["products"]
        ]

    def get_reagents(self) -> List[Molecule]:
        """To get the list of reagents"""
        return [
            mol for h, mol in self.catalog.items() if h in self.role_map["reagents"]
        ]


class Role(Enum):
    """Class for 2-levels based classification of molecules' roles in chemical reactions"""

    @classmethod
    def list(cls):
        """To return the list of possible values of the Role class"""
        return list(map(lambda c: c.value, cls))

    @classmethod
    def from_string(cls, role):
        """To build a Role instance from a string"""
        return cls(role)

    @classmethod
    def get_class_name(cls):
        return cls.__name__.lower()

    def get_full_name(self):
        return f"{self.get_class_name()}.{self.name}".lower()

    def __str__(self):
        return f"{self.get_class_name()}.{self.name}"


class Product(Role):
    """Class with possible classifications of Products in a reaction"""

    DESIRED = "desired_product"
    BYPRODUCT = "by_product"
    SIDEPRODUCT = "side_product"
    UNKNOWN = "unknown"


class Reagent(Role):
    """Class with possible classifications of Reagents in a reaction"""

    SOLVENT = "solvent"
    CATALYST = "catalyst"
    AGENT = "agent"
    ADDITIVE = "additive"
    REACTANT_QUENCH = "reactant_quench"
    REAGENT_QUENCH = "reagent_quench"
    UNKNOWN = "unknown"


class Reactant(Role):
    """Class with possible classifications of Reactants in a reaction"""

    BP1 = "bp1"
    BP2 = "bp2"
    BP3 = "bp3"
    UNKNOWN = "unknown"
