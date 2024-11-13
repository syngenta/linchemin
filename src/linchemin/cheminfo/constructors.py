from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Union

import linchemin.cheminfo.functions as cif
import linchemin.utilities as utilities
from linchemin import settings
from linchemin.cheminfo.chemical_hashes import (
    MolIdentifierFactory,
    UnavailableMolIdentifier,
    UnavailableReactionIdentifier,
    calculate_disconnection_hash_map,
    calculate_molecular_hash_map,
    calculate_pattern_hash_map,
    calculate_reaction_like_hash_map,
    get_all_molecular_identifiers,
    validate_molecular_identifier,
    validate_reaction_identifier,
)
from linchemin.cheminfo.models import (
    ChemicalEquation,
    Disconnection,
    Molecule,
    Pattern,
    Product,
    Ratam,
    Reactant,
    Reagent,
    Role,
    Template,
)

"""
Module containing the constructor classes of relevant cheminformatics models defined in 'models' module
"""


class BadMapping(Exception):
    """To be raised if an atom number is used more than once, indicating that the atom mapping is invalid"""

    pass


logger = utilities.console_logger(__name__)


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
        self,
        molecule_string: str,
        inp_fmt: str,
    ) -> Molecule:
        """To build a Molecule instance from a string"""
        rdmol_input = cif.rdmol_from_string(
            input_string=molecule_string, inp_fmt=inp_fmt
        )
        return self.build_from_rdmol(rdmol_input)

    def build_from_rdmol(self, rdmol: cif.Mol) -> Molecule:
        """To build a Molecule instance from a rdkit Mol instance"""
        rdmol_mapped = rdmol
        rdmol_unmapped = cif.remove_rdmol_atom_mapping(rdmol=rdmol_mapped)

        rdmol_unmapped_canonical = cif.new_molecule_canonicalization(rdmol_unmapped)
        rdmol_mapped_canonical = cif.new_molecule_canonicalization(rdmol_mapped)

        hash_map = calculate_molecular_hash_map(
            get_all_molecular_identifiers(),
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


# Ratam Constructor
AtomTransformation = namedtuple(
    "AtomTransformation",
    ["product_uid", "reactant_uid", "prod_atom_id", "react_atom_id", "map_num"],
)


class RatamConstructor:
    """Class implementing the constructor of the Ratam class"""

    def create_ratam(self, molecules_catalog: dict, desired_product: Molecule) -> Ratam:
        """To initialize an instance of the Ratam class"""
        ratam = Ratam()

        ratam.full_map_info = self.get_full_map_info(molecules_catalog, desired_product)

        ratam.atom_transformations = self.get_atom_transformations(ratam.full_map_info)
        dp_atom_transformations = {
            at
            for at in ratam.atom_transformations
            if at.product_uid == desired_product.uid
        }
        ratam.desired_product_unmapped_atoms_info = (
            self.get_unmapped_atoms_in_desired_product(
                ratam.full_map_info["products"][desired_product.uid],
                dp_atom_transformations,
                desired_product.uid,
            )
        )

        ratam.reactants_unmapped_atoms_info = self.get_unmapped_atoms_in_reactants(
            ratam.full_map_info["reactants"], ratam.atom_transformations
        )

        return ratam

    def get_full_map_info(
        self, molecules_catalog: dict, desired_product: Molecule
    ) -> Union[dict, None]:
        """To create a dictionary mapping each molecule with its list of atom mapping based on its role
        {'reactants': {uid: [map]}, 'reagents': {uid: [map]}, 'products': {uid: [map]}}
        """
        full_map_info = {"reactants": {}, "reagents": {}, "products": {}}
        full_map_info = self.products_map_info(molecules_catalog, full_map_info)
        dp_map_nums = {
            a.GetAtomMapNum() for a in desired_product.rdmol_mapped.GetAtoms()
        }
        full_map_info = self.precursors_map_info(
            molecules_catalog, full_map_info, dp_map_nums
        )

        full_map_info = cif.clean_full_map_info(full_map_info)
        self.mapping_sanity_check(full_map_info)
        return full_map_info

    def get_unmapped_atoms_in_reactants(
        self, reactants_map_info: dict, atom_transformations: Set[AtomTransformation]
    ) -> dict:
        """To get the information about unmapped atoms in the reactants"""
        info = {"unmapped_atoms": {}}
        reactants_tot_atoms = 0.0
        reactants_tot_unmapped = 0.0
        for r_uid, map_list in reactants_map_info.items():
            r_transformed_atoms = {
                at.map_num for at in atom_transformations if at.reactant_uid == r_uid
            }
            for mapping in map_list:
                n_atoms, n_unmapped_atoms, info = self.analyze_map_dict(
                    mapping, r_transformed_atoms, info, r_uid
                )
                reactants_tot_atoms += n_atoms
                reactants_tot_unmapped += n_unmapped_atoms

        info["fraction"] = round(reactants_tot_unmapped / reactants_tot_atoms, 2)
        return info

    def get_unmapped_atoms_in_desired_product(
        self,
        desired_product_map_info: List[dict],
        dp_atom_transformations: Set[AtomTransformation],
        dp_uid: int,
    ) -> dict:
        """To get the information about unmapped atoms in the desired product"""
        info = {"unmapped_atoms": {}}
        tot_atoms = 0.0
        tot_unmapped_atoms = 0.0
        transformed_atoms_map = {at.map_num for at in dp_atom_transformations}
        for mapping in desired_product_map_info:
            n_atoms, n_unmapped_atoms, info = self.analyze_map_dict(
                mapping, transformed_atoms_map, info, dp_uid
            )
            tot_atoms += n_atoms
            tot_unmapped_atoms += n_unmapped_atoms

        info["fraction"] = round(tot_unmapped_atoms / tot_atoms, 2)
        return info

    def analyze_map_dict(
        self, mapping, transformed_atoms_map, info, uid
    ) -> Tuple[int, int, dict]:
        """To check if there are unmapped atoms based on a mapping dictionary"""
        if unmapped_atoms := self.find_unmapped_atoms(mapping, transformed_atoms_map):
            if uid not in info["unmapped_atoms"].keys():
                info["unmapped_atoms"][uid] = [unmapped_atoms]
            else:
                info["unmapped_atoms"][uid].append(unmapped_atoms)
            return (
                len(mapping.keys()),
                len(unmapped_atoms),
                info,
            )
        else:
            return len(mapping.keys()), 0, info

    @staticmethod
    def find_unmapped_atoms(mapping: dict, transformed_atoms_id: set) -> set:
        return {
            a_id
            for a_id, map_num in mapping.items()
            if map_num not in transformed_atoms_id
        }

    @staticmethod
    def products_map_info(molecules_catalog: dict, full_map_info: dict) -> dict:
        """To collect the mapping info for product Molecules"""
        for mol in set(molecules_catalog["products"]):
            full_map_info["products"][mol.uid] = []
            for m in [n for n in molecules_catalog["products"] if n == mol]:
                mapping = {
                    a.GetIdx(): a.GetAtomMapNum() for a in m.rdmol_mapped.GetAtoms()
                }
                full_map_info["products"][mol.uid].append(mapping)
        return full_map_info

    def precursors_map_info(
        self,
        molecules_catalog: dict,
        full_map_info: dict,
        desired_product_map_nums: set,
    ) -> dict:
        """To collect the mapping info for product Molecules"""
        for mol in molecules_catalog["reactants"] + molecules_catalog["reagents"]:
            if mol.uid not in full_map_info["reactants"]:
                full_map_info["reactants"][mol.uid] = []
            if mol.uid not in full_map_info["reagents"]:
                full_map_info["reagents"][mol.uid] = []
            map_nums = {a.GetAtomMapNum() for a in mol.rdmol_mapped.GetAtoms()}
            full_map_info = self.update_full_info(
                full_map_info, mol, desired_product_map_nums, map_nums
            )

        return full_map_info

    @staticmethod
    def update_full_info(
        full_map_info: dict,
        precursor: Molecule,
        desired_product_map_nums: set,
        precursor_map_nums: set,
    ):
        """To update the full_map_info dictionary with a new entry of either a reactant or a reagent"""
        shared_map = next(
            (
                n
                for n in precursor_map_nums
                if n in desired_product_map_nums and n not in [0, -1]
            ),
            None,
        )
        if shared_map is not None:
            # it's a reactant!
            full_map_info["reactants"][precursor.uid].append(
                {
                    a.GetIdx(): a.GetAtomMapNum()
                    for a in precursor.rdmol_mapped.GetAtoms()
                }
            )
        else:
            # it's a reagent!
            full_map_info["reagents"][precursor.uid].append(
                {
                    a.GetIdx(): a.GetAtomMapNum()
                    for a in precursor.rdmol_mapped.GetAtoms()
                }
            )
        return full_map_info

    @staticmethod
    def mapping_sanity_check(full_map_info: dict) -> None:
        """Mapping sanity check: if a map number appears more than 2 times in the reactants,
        the mapping is invalid and an error is raised"""
        map_nums = []
        for uid, map_list in full_map_info["reactants"].items():
            for d in map_list:
                new_nums = list(d.values())
                map_nums.extend(iter(new_nums))
        d = {n: map_nums.count(n) for n in set(map_nums)}
        if [c for n, c in d.items() if c > 1 and n not in [0, -1]]:
            logger.error("Invalid mapping! The same map number is used more than once")
            raise BadMapping

    @staticmethod
    def get_atom_transformations(full_map_info: dict) -> Set[AtomTransformation]:
        """To create the list of AtomTransformations from a catalog of mapped Molecule objects"""
        atom_transformations = set()
        products = full_map_info["products"]
        reactants = full_map_info["reactants"]

        def get_matching_map_numbers(reactant_map: dict, product_map: dict) -> list:
            return [
                map_num
                for map_num in reactant_map.values()
                if map_num in product_map.values() and map_num not in [0, -1]
            ]

        def process_product_reactant_maps(
            prod_uid: str, prod_map: dict, react_uid: str, react_map: dict
        ) -> None:
            matching_map_numbers = get_matching_map_numbers(react_map, prod_map)
            if matching_map_numbers:
                atom_transformations.update(
                    build_atom_transformations(
                        matching_map_numbers,
                        prod_map,
                        prod_uid,
                        react_map,
                        react_uid,
                    )
                )

        def process_reactant_maps(
            product_uid: str, product_map: dict, r_maps: dict
        ) -> None:
            for r_uid, r_maps_group in r_maps.items():
                for r_map in r_maps_group:
                    process_product_reactant_maps(
                        product_uid, product_map, r_uid, r_map
                    )

        for p_uid, p_maps_group in products.items():
            for p_map in p_maps_group:
                process_reactant_maps(p_uid, p_map, reactants)

        return atom_transformations


def build_atom_transformations(
    matching_map_num: List[int],
    prod_map: dict,
    product_uid: str,
    reactant_map: dict,
    reactant_uid: str,
) -> List[AtomTransformation]:
    """To build the list of AtomTransformation objects for each pair of product-reactant with matching map number"""
    ats = []
    for map_num in matching_map_num:
        p_aids = {aid for aid, map_val in prod_map.items() if map_val == map_num}
        r_aids = {aid for aid, map_val in reactant_map.items() if map_val == map_num}
        ats.extend(
            [
                AtomTransformation(product_uid, reactant_uid, p_aid, r_aid, map_num)
                for p_aid in p_aids
                for r_aid in r_aids
            ]
        )
    return ats


# Disconnection Constructor
class DisconnectionConstructor:
    def __init__(self, identity_property_name: str):
        self.identity_property_name = identity_property_name

    def build_from_chemical_equation(
        self, chemical_equation: ChemicalEquation, desired_product: Molecule
    ) -> Union[Disconnection, None]:
        """To initialize the Disconnection instance and populate its attributes based on the reactive center of the
        ChemicalEquation"""

        rxn_reactive_center = RXNReactiveCenter(chemical_equation, desired_product)
        if (
            len(rxn_reactive_center.rxn_bond_info_list) == 0
            and len(rxn_reactive_center.rxn_atom_info_list) == 0
            and len(rxn_reactive_center.rxn_atomh_info_list) == 0
        ):
            return None

        product_changes = rxn_reactive_center.get_product_changes()
        disconnection = Disconnection()
        disconnection.molecule = desired_product
        disconnection.rdmol = rxn_reactive_center.disconnection_rdmol
        disconnection.reacting_atoms = product_changes.reacting_atoms
        disconnection.hydrogenated_atoms = product_changes.hydrogenated_atoms
        disconnection.new_bonds = product_changes.new_bonds
        disconnection.modified_bonds = product_changes.modified_bonds
        disconnection.hash_map = calculate_disconnection_hash_map(disconnection)
        disconnection.identity_property = disconnection.hash_map.get(
            "disconnection_summary"
        )
        disconnection.uid = utilities.create_hash(disconnection.identity_property)
        disconnection.rdmol_fragmented = self.get_fragments(
            rdmol=desired_product.rdmol_mapped,
            new_bonds=disconnection.new_bonds,
            fragmentation_method=2,
        )
        return disconnection

    def get_fragments(
        self, rdmol: cif.Mol, new_bonds: List[tuple], fragmentation_method: int = 1
    ) -> Union[cif.Mol, None]:
        """To get the fragments of the desired product Mol. Inspired by
        https://github.com/rdkit/rdkit/issues/2081
        """

        def get_fragments_method_1(rdmol, bonds) -> cif.Mol:
            rdmol_fragmented = cif.rdkit.Chem.FragmentOnBonds(
                rdmol,
                bonds,
                addDummies=True,
                # dummyLabels=[(0, 0)]
            )
            return rdmol_fragmented

        def get_fragments_method_2(rdmol: cif.Mol, bonds: List[tuple]) -> cif.Mol:
            bonds = [rdmol.GetBondBetweenAtoms(*tup).GetIdx() for tup in bonds]
            mh = cif.Chem.RWMol(cif.Chem.AddHs(rdmol))
            cif.Chem.Kekulize(mh, clearAromaticFlags=True)

            for bond in [mh.GetBondWithIdx(idx) for idx in bonds]:
                a = bond.GetEndAtomIdx()
                b = bond.GetBeginAtomIdx()
                mh.RemoveBond(a, b)

                n_ats = mh.GetNumAtoms()
                mh.AddAtom(cif.Chem.Atom(0))
                mh.AddBond(a, n_ats, cif.Chem.BondType.SINGLE)
                mh.AddAtom(cif.Chem.Atom(0))
                mh.AddBond(b, n_ats + 1, cif.Chem.BondType.SINGLE)
                n_ats += 2

            cif.Chem.SanitizeMol(mh)
            cif.rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_NONE
            cif.Chem.RemoveHs(mh)

            return mh

        fragmentation_method_factory = {
            1: get_fragments_method_1,
            2: get_fragments_method_2,
        }
        try:
            frag_func = fragmentation_method_factory[fragmentation_method]
            from rdkit.Chem import AllChem

            AllChem.Compute2DCoords(rdmol)
            rdmol_fragmented = frag_func(rdmol=rdmol, bonds=new_bonds)
            return rdmol_fragmented
        except Exception as e:
            logger.warning(
                f"Something went wrong while creating the fragments: {e}."
                f"None is returned"
            )
            return None


BondInfo = namedtuple("BondInfo", ("product_atoms", "product_bond", "status"))


@dataclass(eq=True, order=False, frozen=False)
class RxnBondInfo:
    """Class for keeping track of bonds that between reactants and products of a reaction are new or changed."""

    product_atoms: tuple
    product_bond: int
    status: str


@dataclass(eq=True, order=False, frozen=False)
class RXNProductChanges:
    """Class for keeping track of the changes in a reaction product"""

    reacting_atoms: List[int]
    hydrogenated_atoms: List[tuple]
    new_bonds: List[tuple]
    modified_bonds: List[tuple]
    rdmol: cif.Mol


class RXNReactiveCenter:
    """
    Class to identify the atoms and bonds that change in a reaction
    inspired from
    https://greglandrum.github.io/rdkit-blog/tutorial/reactions/2021/11/26/highlighting-changed-bonds-in-reactions.html
    """

    def __init__(self, chemical_equation: ChemicalEquation, desired_product: Molecule):
        chemical_equation.rdrxn.Initialize()
        (
            self.rxn_atom_info_list,
            self.rxn_atomh_info_list,
            self.rxn_bond_info_list,
            self.disconnection_rdmol,
        ) = self.find_modifications_in_products(chemical_equation, desired_product)

    def get_product_changes(self):
        """To get the changes generated by the reaction: reacting atoms, hydrogenated atoms, created bonds and
        modified bonds. They are returned as attributes of a RXNProductChanges instance
        """
        reacting_atoms = sorted([at.prod_atom_id for at in self.rxn_atom_info_list])
        hydrogenated_atoms = sorted(
            [(d["p_atom"], d["delta_hydrogen"]) for d in self.rxn_atomh_info_list]
        )

        new_bonds = sorted(
            [bi.product_atoms for bi in self.rxn_bond_info_list if bi.status == "new"]
        )
        modified_bonds = sorted(
            [
                bi.product_atoms
                for bi in self.rxn_bond_info_list
                if bi.status == "changed"
            ]
        )
        return RXNProductChanges(
            reacting_atoms,
            hydrogenated_atoms,
            new_bonds,
            modified_bonds,
            self.disconnection_rdmol,
        )

    @staticmethod
    def get_mapped_neighbors(atom: cif.Atom) -> dict:
        """To get the mapped neighbors of an atom"""
        res: dict = {}
        amap = atom.GetAtomMapNum()
        if amap == 0:
            return res
        for bond in atom.GetBonds():
            neighbor = next(
                a
                for a in [bond.GetEndAtom(), bond.GetBeginAtom()]
                if a.GetAtomMapNum() != amap
            )
            n_map = neighbor.GetAtomMapNum()
            if amap > n_map:
                res[(n_map, amap)] = (atom.GetIdx(), neighbor.GetIdx())
            else:
                res[(amap, n_map)] = (neighbor.GetIdx(), atom.GetIdx())

        return res

    def find_modifications_in_products(
        self, ce: ChemicalEquation, desired_product: Molecule
    ) -> Tuple[List[AtomTransformation], List[dict], List[RxnBondInfo], cif.Mol]:
        """To identify the list of reacting atoms, hydrogenated atoms and new bonds and modified bonds.
        It returns a 3-tuple"""

        # retrieve the map numbers of the reacting atoms
        maps_reacting_atoms = self.get_reacting_atoms_map_numbers(ce)
        # retrieve the AtomTransformations of the desired product involving the reacting atoms
        ats_desired_product = [
            at
            for at in ce.mapping.atom_transformations
            if at.product_uid == desired_product.uid
            and at.map_num in maps_reacting_atoms
        ]
        disconnection_rdmol = copy.deepcopy(desired_product.rdmol_mapped)
        seen = set()
        bonds = []
        hydrogenated_atoms = []
        # for each AtomTransformation involving reacting atoms..
        for at in ats_desired_product:
            # ...the involved atom of the desired product and of the reactant are identified
            reactant = next(
                mol for h, mol in ce.catalog.items() if h == at.reactant_uid
            )
            r_atom, p_atom = self.get_changing_atoms(
                at, disconnection_rdmol, reactant.rdmol_mapped
            )
            # if the atom changes the number of bonded hydrogens it's identified and the variation in the number
            # of hydrogens is computed
            delta_h = p_atom.GetTotalNumHs() - r_atom.GetTotalNumHs()
            if delta_h != 0:
                hydrogenated_atoms.append(
                    {"p_atom": p_atom.GetIdx(), "delta_hydrogen": delta_h}
                )

            # based on their neighbors, new and modified bonds are identified
            bonds, seen = self.get_bond_info(
                r_atom, reactant, p_atom, disconnection_rdmol, seen, bonds
            )

        rxn_bond_info_list = [RxnBondInfo(**item._asdict()) for item in bonds]
        return (
            ats_desired_product,
            hydrogenated_atoms,
            rxn_bond_info_list,
            disconnection_rdmol,
        )

    @staticmethod
    def get_reacting_atoms_map_numbers(ce: ChemicalEquation) -> List[int]:
        """ " To identify the map numbers associated with the reacting atoms in a ChemicalEquation"""
        ce.rdrxn.Initialize()
        reacting_atoms = ce.rdrxn.GetReactingAtoms()
        maps_reacting_atoms = []
        for ridx, reacting in enumerate(reacting_atoms):
            r = ce.rdrxn.GetReactantTemplate(ridx)
            maps_reacting_atoms.extend(
                r.GetAtomWithIdx(raidx).GetAtomMapNum() for raidx in reacting
            )
        return maps_reacting_atoms

    @staticmethod
    def get_changing_atoms(
        at: AtomTransformation, product_rdmol: cif.Mol, reactant_rdmol: cif.Mol
    ) -> tuple:
        """To identify the atoms involved in an AtomTransformation"""
        r_atom = next(
            atom
            for atom in reactant_rdmol.GetAtoms()
            if atom.GetIdx() == at.react_atom_id
        )
        p_atom = next(
            atom
            for atom in product_rdmol.GetAtoms()
            if atom.GetIdx() == at.prod_atom_id
        )
        return r_atom, p_atom

    def get_bond_info(
        self,
        r_atom: cif.Atom,
        reactant: Molecule,
        p_atom: cif.Atom,
        product: cif.Mol,
        seen: set,
        bonds: List,
    ) -> tuple:
        """To extract the information regarding new or modified bonds in a ChemicalEquation"""
        # based on their neighbors, new and modified bonds are identified
        rnbrs = self.get_mapped_neighbors(r_atom)
        pnbrs = self.get_mapped_neighbors(p_atom)
        for tpl in pnbrs:
            pbond = product.GetBondBetweenAtoms(*pnbrs[tpl])
            if (pbond.GetIdx()) in seen:
                continue
            seen.add(pbond.GetIdx())

            if tpl not in rnbrs:
                # new bond in product
                bonds.extend(
                    [
                        BondInfo(
                            product_atoms=pnbrs[tpl],
                            product_bond=pbond.GetIdx(),
                            status="new",
                        )
                    ]
                )
            else:
                # present in both reactants and products, check to see if it changed
                rbond = reactant.rdmol_mapped.GetBondBetweenAtoms(*rnbrs[tpl])
                if rbond.GetBondType() != pbond.GetBondType():
                    bonds.extend(
                        [
                            BondInfo(
                                product_atoms=pnbrs[tpl],
                                product_bond=pbond.GetIdx(),
                                status="changed",
                            )
                        ]
                    )

        return bonds, seen


# Pattern Constructor
class PatternConstructor:
    """Class implementing the constructor of the Pattern class"""

    def __init__(
        self,
        identity_property_name: str = settings.CONSTRUCTORS.pattern_identity_property_name,
    ):
        self.identity_property_name = identity_property_name

    def create_pattern(self, rdmol: cif.Mol) -> Pattern:
        pattern = Pattern()
        rdmol_mapped = rdmol
        rdmol_unmapped = cif.remove_rdmol_atom_mapping(rdmol=rdmol_mapped)
        rdmol_unmapped_canonical = cif.canonicalize_rdmol_lite(
            rdmol=rdmol_unmapped, is_pattern=True
        )
        rdmol_mapped_canonical = cif.canonicalize_rdmol_lite(
            rdmol=rdmol_mapped, is_pattern=True
        )
        pattern.rdmol = rdmol_unmapped_canonical
        pattern.rdmol_mapped = rdmol_mapped_canonical
        pattern.smarts = cif.compute_mol_smarts(rdmol=pattern.rdmol)
        pattern.hash_map = calculate_pattern_hash_map(pattern.smarts)

        pattern.identity_property = pattern.hash_map.get(self.identity_property_name)
        pattern.uid = utilities.create_hash(
            pattern.identity_property
        )  # the hashed identity property
        return pattern

    def build_from_molecule_string(
        self,
        molecule_string: str,
        inp_fmt: str,
    ) -> Pattern:
        rdmol_input = cif.rdmol_from_string(
            input_string=molecule_string, inp_fmt=inp_fmt
        )
        return self.create_pattern(rdmol_input)

    def build_from_rdmol(self, rdmol: cif.Mol) -> Pattern:
        return self.create_pattern(rdmol)


# Template Constructor
class TemplateConstructor:
    def __init__(
        self,
        identity_property_name: str = settings.CONSTRUCTORS.pattern_identity_property_name,
    ):
        """To initialize a Template instance. The machinery is based on rdchiral."""
        self.identity_property_name = identity_property_name

    def read_reaction(
        self, reaction_string: str, inp_fmt: str
    ) -> Tuple[cif.rdChemReactions.ChemicalReaction, utilities.OutcomeMetadata]:
        """To attempt in sanitizing the rdkit reaction"""
        try:
            rdrxn = cif.rdrxn_from_string(input_string=reaction_string, inp_fmt=inp_fmt)
            # Sanitization to cause issues with some molecular structures https://github.com/rdkit/rdkit/issues/7108
            # sanitize_msg = cif.rdChemReactions.SanitizeRxn(rdrxn, catchErrors=True)
            #
            # outcome = utilities.OutcomeMetadata(
            #     name="read_reaction",
            #     is_successful=True,
            #     log={"sanitize_msg": sanitize_msg},
            # )
            outcome = utilities.OutcomeMetadata(
                name="read_reaction", is_successful=True, log={}
            )
        except Exception as e:
            outcome = utilities.OutcomeMetadata(
                name="read_reaction", is_successful=False, log={"exception": e}
            )
            rdrxn = None

        return rdrxn, outcome

    def unpack_rdrxn(
        self, rdrxn: cif.rdChemReactions.ChemicalReaction
    ) -> Tuple[dict, dict, dict]:
        """To build the basic attributes of the Template using the ChemicalEquation Builder"""
        constructor = PatternConstructor(
            identity_property_name=self.identity_property_name
        )
        reaction_mols = cif.rdrxn_to_molecule_catalog(rdrxn, constructor)
        builder = UnmappedChemicalEquationGenerator()
        attributes, _ = builder.get_basic_attributes(reaction_mols, None)
        return (
            attributes["catalog"],
            attributes["stoichiometry_coefficients"],
            attributes["role_map"],
        )

    def build_from_rdrxn(
        self, rdrxn: cif.rdChemReactions.ChemicalReaction
    ) -> Union[Template, None]:
        """To initialize the instance based on an rdkit reaction object"""
        reaction_string = cif.rdChemReactions.ReactionToSmarts(rdrxn)
        return self.build_from_reaction_string(
            reaction_string=reaction_string, inp_fmt="smarts"
        )

    def build_from_reaction_string(
        self, reaction_string: str, inp_fmt: str
    ) -> Union[Template, None]:
        """To initialize the instance based on a reaction string"""
        rdchiral_output = cif.rdchiral_extract_template(
            reaction_string=reaction_string, inp_fmt=inp_fmt, reaction_id=None
        )
        return self.build_from_rdchiral_output(rdchiral_output=rdchiral_output)

    def create_rdchiral_data(self, rdchiral_output: Dict):
        """To build the data necessary to usage of rdchiral"""
        reaction_rwd_smarts = rdchiral_output.get("reaction_smarts")
        reaction_fwd_smarts = ">".join(reaction_rwd_smarts.split(">")[::-1])
        rdrxn, read_outcome = self.read_reaction(
            reaction_string=reaction_fwd_smarts, inp_fmt="smarts"
        )
        rdchiral_data = rdchiral_output.copy()
        rdchiral_data.pop("reaction_id")
        rdchiral_data.pop("reaction_smarts")
        rdchiral_data["reaction_fwd_smarts"] = reaction_fwd_smarts
        rdchiral_data["reaction_rwd_smarts"] = reaction_rwd_smarts
        return rdrxn, rdchiral_data

    def build_from_rdchiral_output(
        self, rdchiral_output: Dict
    ) -> Union[Template, None]:
        """To build the Template attributes based on rdchiral machinery.
        If rdchiral process fails, None is returned"""
        if not rdchiral_output:
            return None
        if not rdchiral_output.get("reaction_smarts"):
            return None
        template = Template()
        try:
            rdrxn, rdchiral_data = self.create_rdchiral_data(rdchiral_output)
        except Exception:
            logger.warning(
                "Issues with rdchiral calculation of the template. None is returned."
            )
            return None
        template.rdrxn = rdrxn
        template.rdchiral_data = rdchiral_data
        pattern_catalog, stoichiometry_coefficients, role_map = self.unpack_rdrxn(
            rdrxn=rdrxn
        )
        template.pattern_catalog = pattern_catalog
        template.stoichiometry_coefficients = stoichiometry_coefficients
        template.role_map = role_map
        template.hash_map = calculate_reaction_like_hash_map(pattern_catalog, role_map)
        template.uid = template.hash_map.get(
            settings.CONSTRUCTORS.template_identity_property
        )  # TODO: review

        template.rdrxn = cif.build_rdrxn(
            catalog=pattern_catalog,
            role_map=role_map,
            stoichiometry_coefficients=stoichiometry_coefficients,
            use_reagents=True,
            use_smiles=False,
            use_atom_mapping=True,
        )
        template.smarts = cif.rdrxn_to_string(rdrxn=template.rdrxn, out_fmt="smarts")

        return template


# ChemicalEquation Constructor
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

    def __init__(
        self,
        molecular_identity_property_name: str = settings.CONSTRUCTORS.molecular_identity_property_name,
        chemical_equation_identity_name: str = settings.CONSTRUCTORS.chemical_equation_identity_name,
    ):
        validate_molecular_identifier(
            molecular_identifier=molecular_identity_property_name
        )
        self.molecular_identity_property_name = molecular_identity_property_name
        validate_reaction_identifier(
            reaction_identifier=chemical_equation_identity_name
        )
        self.chemical_equation_identity_name = chemical_equation_identity_name

    @staticmethod
    def read_reaction(
        reaction_string: str, inp_fmt: str
    ) -> Tuple[cif.rdChemReactions.ChemicalReaction, utilities.OutcomeMetadata]:
        """To start the building of a ChemicalEquation instance from a reaction string"""
        try:
            rdrxn = cif.rdrxn_from_string(input_string=reaction_string, inp_fmt=inp_fmt)
            # Sanitization to cause issues with some molecular structures https://github.com/rdkit/rdkit/issues/7108
            # sanitize_msg = cif.rdChemReactions.SanitizeRxn(rdrxn, catchErrors=True)
            # outcome = utilities.OutcomeMetadata(
            #     name="read_reaction",
            #     is_successful=True,
            #     log={"sanitize_msg": sanitize_msg},
            # )
            outcome = utilities.OutcomeMetadata(
                name="read_reaction", is_successful=True, log={}
            )

        except Exception as e:
            logger.warning("Exception is raised, rdrxn is None")
            print(e)
            outcome = utilities.OutcomeMetadata(
                name="read_reaction", is_successful=False, log={"exception": e}
            )
            rdrxn = None

        return rdrxn, outcome

    def unpack_rdrxn(
        self,
        rdrxn: cif.rdChemReactions.ChemicalReaction,
        desired_product: Union[cif.Mol, None],
    ) -> ChemicalEquation:
        """To compute ChemicalEquation attributes from the associated rdkit ChemicalReaction object"""
        constructor = MoleculeConstructor(
            molecular_identity_property_name=self.molecular_identity_property_name
        )
        return create_chemical_equation(
            rdrxn=rdrxn,
            chemical_equation_identity_name=self.chemical_equation_identity_name,
            constructor=constructor,
            desired_product=desired_product,
        )

    def build_from_rdrxn(
        self,
        rdrxn: cif.rdChemReactions.ChemicalReaction,
        desired_product: Union[cif.Mol, None] = settings.CONSTRUCTORS.desired_product,
    ) -> ChemicalEquation:
        """To build a ChemicalEquation instance from a rdkit ChemicalReaction object"""
        return self.unpack_rdrxn(rdrxn=rdrxn, desired_product=desired_product)

    def build_from_reaction_string(
        self,
        reaction_string: str,
        inp_fmt: str,
        desired_product: Union[str, None] = settings.CONSTRUCTORS.desired_product,
    ) -> ChemicalEquation:
        """To build a ChemicalEquation instance from a reaction string"""
        rdrxn, read_outcome = self.read_reaction(
            reaction_string=reaction_string, inp_fmt=inp_fmt
        )
        if desired_product:
            desired_product = cif.rdmol_from_string(desired_product, inp_fmt=inp_fmt)
        return self.build_from_rdrxn(rdrxn=rdrxn, desired_product=desired_product)

    def build_from_db(self, mol_list: List[dict]) -> ChemicalEquation:
        """To build a ChemicalEquation instance from information extracted from a database.
        The input list of dictionaries is expected to have the following format:
        [{'smiles': mol_smiles, 'role': role, 'stoichiometry': n}]"""
        mol_constructor = MoleculeConstructor(
            molecular_identity_property_name=self.molecular_identity_property_name
        )
        stoichiometry_coefficients = {}
        catalog = {}
        desired_product = None
        for mol in mol_list:
            original_stoich = mol["stoichiometry"]
            original_role = mol["role"]
            # dividing possible substances in single molecules
            molecules = mol["smiles"].split(".")
            for m in molecules:
                # building the molecule instance
                molecule = mol_constructor.build_from_molecule_string(m, "smiles")
                # adding molecule to catalog dictionary
                catalog[molecule.uid] = molecule
                # defining molecule role
                role = self.get_role(original_role)
                if role.get_full_name() == "product.desired":
                    desired_product = molecule
                # adding stoichiometry info to the stoichiometry dictionary
                stoichiometry_coefficients = self.update_stoichiometry(
                    stoichiometry_coefficients, role, molecule.uid, original_stoich
                )

        role_map = self.get_role_map(stoichiometry_coefficients)
        if desired_product is None:
            desired_product = self.get_desired_product(catalog, role_map)
        basic_attributes = {
            "mapping": None,
            "role_map": role_map,
            "stoichiometry_coefficients": stoichiometry_coefficients,
            "catalog": catalog,
        }
        return self.initialize_builder(basic_attributes, desired_product)

    def initialize_builder(self, basic_attributes, desired_product) -> ChemicalEquation:
        """To initialize the appropriate ChemicalEquation builder"""
        builder = Builder()
        builder.set_builder("unmapped")
        return builder.get_chemical_equation(
            self.chemical_equation_identity_name,
            desired_product,
            basic_attributes=basic_attributes,
        )

    @staticmethod
    def get_role(input_role: str) -> Union[Product, Reactant, Reagent]:
        """To get the Role instance to the input role string"""
        for cls in Role.__subclasses__():
            if input_role == cls.get_class_name():
                return cls.from_string("unknown")
            elif input_role in cls.list():
                return cls.from_string(input_role)

    @staticmethod
    def get_stoichiometry_coeff(stoichiometry_coeff: float) -> int:
        """To ensure that a stoichiometry coefficient is an integer"""
        if isinstance(stoichiometry_coeff, int):
            return stoichiometry_coeff
        elif int(stoichiometry_coeff // 1) > 0:
            return int(stoichiometry_coeff // 1)
        elif int(stoichiometry_coeff) == 0:
            return 1

    def update_stoichiometry(
        self,
        stoichiometry_coefficients: dict,
        role: Union[Reactant, Reagent, Product],
        mol_uid: int,
        original_stoich: int,
    ) -> dict:
        """To update the stoichiometry coefficients dictionary with a new entry"""
        roles = {Reactant: "reactants", Reagent: "reagents", Product: "products"}
        stoich_coeff = self.get_stoichiometry_coeff(original_stoich)
        general_role = roles[type(role)]
        sc_dict = stoichiometry_coefficients.get(general_role, {})
        if mol_uid in sc_dict:
            sc_dict[mol_uid] += stoich_coeff
        else:
            sc_dict.update({mol_uid: stoich_coeff})
        stoichiometry_coefficients[general_role] = sc_dict
        return stoichiometry_coefficients

    @staticmethod
    def get_role_map(stoichiometry_coefficients: dict) -> dict:
        """To build the role_map dictionary"""
        role_map = {
            role: sorted(list(info.keys()))
            for role, info in stoichiometry_coefficients.items()
        }
        if "reagents" not in role_map:
            role_map["reagents"] = []
        return role_map

    @staticmethod
    def get_desired_product(catalog: dict, role_map: dict) -> Molecule:
        """To assign a default desired product if it has not been found"""
        products = [prod for h, prod in catalog.items() if h in role_map["products"]]
        return cif.select_desired_product(products)


def create_chemical_equation(
    rdrxn: cif.rdChemReactions.ChemicalReaction,
    chemical_equation_identity_name: str,
    constructor: MoleculeConstructor,
    desired_product: Union[Molecule, None],
) -> ChemicalEquation:
    """
    To initialize the correct builder of the ChemicalEquation based on the presence of the atom mapping.

    Parameters:
    ------------
    rdrxn: rdChemReactions.ChemicalReaction
        The rdkit Chemical Reaction object
    chemical_equation_identity_name: str
        The string indicating which representation of the ChemicalEquation determines its identity
    constructor: MoleculeConstructor
        The constructor for the Molecule objects involved
    desired_product: Union[Molecule, None]
        The Molecule object representing the product to be considered the desired product of the reaction;
        if it is not provided, the molecule with the heaviest molecular weight is considered the desired product.

    Returns:
    --------
    a new ChemicalEquation instance
    """
    builder_type = "mapped" if cif.has_mapped_products(rdrxn) else "unmapped"
    reaction_mols = cif.rdrxn_to_molecule_catalog(rdrxn, constructor)
    if desired_product is not None:
        desired_product_mol = constructor.build_from_rdmol(desired_product)
        if desired_product_mol not in reaction_mols["products"]:
            logger.error("The selected product does not appear in the reaction. ")
            raise ValueError
    else:
        desired_product_mol = cif.select_desired_product(reaction_mols["products"])
    builder = Builder()
    builder.set_builder(builder_type)
    chemical_equation = builder.get_chemical_equation(
        chemical_equation_identity_name, desired_product_mol, reaction_mols
    )

    return chemical_equation


class ChemicalEquationGenerator(ABC):
    """Abstract class for ChemicalEquationAttributesGenerator"""

    @abstractmethod
    def get_basic_attributes(
        self, reaction_mols: dict, desired_product: Union[Molecule, None]
    ) -> Tuple[dict, Molecule]:
        pass

    @abstractmethod
    def generate_template(
        self, chemical_equation: ChemicalEquation
    ) -> Union[Template, None]:
        pass

    @abstractmethod
    def generate_disconnection(
        self, chemical_equation: ChemicalEquation, desired_product
    ) -> Union[Disconnection, None]:
        pass

    @abstractmethod
    def generate_rdrxn(
        self, ce: ChemicalEquation, use_reagents: bool
    ) -> cif.rdChemReactions.ChemicalReaction:
        pass

    @abstractmethod
    def generate_smiles(self, rdrxn: cif.rdChemReactions.ChemicalReaction) -> str:
        pass


class UnmappedChemicalEquationGenerator(ChemicalEquationGenerator):
    def get_basic_attributes(
        self, reaction_mols: dict, desired_product: Union[Molecule, None]
    ) -> Tuple[dict, Molecule]:
        """To build the initial attributes of a ChemicalEquation: mapping, role_map, catalog and
        stoichiometry_coefficients. These are returned in a dictionary."""
        basic_attributes = {
            "mapping": None,
            "role_map": {
                role: sorted(list({m.uid for m in set(mols)}))
                for role, mols in reaction_mols.items()
            },
        }
        all_molecules = (
            reaction_mols["reactants"]
            + reaction_mols["reagents"]
            + reaction_mols["products"]
        )
        basic_attributes["catalog"] = {m.uid: m for m in set(all_molecules)}
        basic_attributes[
            "stoichiometry_coefficients"
        ] = self.generate_stoichiometry_coefficients(reaction_mols)
        return basic_attributes, desired_product

    def generate_rdrxn(
        self, ce: ChemicalEquation, use_reagents: bool
    ) -> cif.rdChemReactions.ChemicalReaction:
        """To build the rdkit rdrxn object associated with the ChemicalEquation"""
        return cif.build_rdrxn(
            catalog=ce.catalog,
            role_map=ce.role_map,
            stoichiometry_coefficients=ce.stoichiometry_coefficients,
            use_reagents=use_reagents,
            use_smiles=False,
            use_atom_mapping=False,
            mapping=ce.mapping,
        )

    def generate_smiles(self, rdrxn: cif.rdChemReactions.ChemicalReaction) -> str:
        """To build the smiles associated with the ChemicalEquation"""
        return cif.rdrxn_to_string(
            rdrxn=rdrxn, out_fmt="smiles", use_atom_mapping=False
        )

    @staticmethod
    def generate_stoichiometry_coefficients(reaction_mols: dict) -> dict:
        """To build the dictionary of the stoichiometry coefficients of an unmapped ChemicalEquation"""
        stoichiometry_coefficients = {}
        for role in ["reactants", "reagents", "products"]:
            molecules = reaction_mols[role]
            molecule_coeffs = {m.uid: molecules.count(m) for m in set(molecules)}
            stoichiometry_coefficients[role] = molecule_coeffs
        return stoichiometry_coefficients

    def generate_template(self, ce: ChemicalEquation):
        """To generate the template. It is None if the ChemicalEquation is unmapped."""
        return None

    def generate_disconnection(self, ce: ChemicalEquation, desired_product: Molecule):
        """To generate the disconnection. It is None if the ChemicalEquation is unmapped."""
        return None


class MappedChemicalEquationGenerator(ChemicalEquationGenerator):
    def get_basic_attributes(
        self, reaction_mols: dict, desired_product: Molecule
    ) -> Tuple[dict, Molecule]:
        """To build the initial attributes of a ChemicalEquation: mapping, role_map, catalog and
        stoichiometry_coefficients. These are returned in a dictionary."""
        basic_attributes = {
            "mapping": self.generate_mapping(reaction_mols, desired_product)
        }
        basic_attributes["role_map"] = {
            role: sorted(list(map_info.keys()))
            for role, map_info in basic_attributes["mapping"].full_map_info.items()
        }

        all_molecules = (
            reaction_mols["reactants"]
            + reaction_mols["reagents"]
            + reaction_mols["products"]
        )
        basic_attributes["catalog"] = {m.uid: m for m in set(all_molecules)}

        basic_attributes[
            "stoichiometry_coefficients"
        ] = self.generate_stoichiometry_coefficients(basic_attributes["mapping"])
        return basic_attributes, desired_product

    @staticmethod
    def generate_mapping(new_reaction_mols: dict, desired_product: Molecule) -> Ratam:
        """To generate the Ratam instance for the ChemicalEquation"""
        ratam_constructor = RatamConstructor()
        return ratam_constructor.create_ratam(new_reaction_mols, desired_product)

    @staticmethod
    def generate_stoichiometry_coefficients(mapping: Ratam) -> dict:
        """To build the dictionary of the stoichiometry coefficients of a mapped ChemicalEquation"""
        stoichiometry_coefficients = {"reactants": {}, "reagents": {}, "products": {}}
        for role, mapping_info in mapping.full_map_info.items():
            for uid, map_list in mapping_info.items():
                stoichiometry_coefficients[role].update({uid: len(map_list)})
        return stoichiometry_coefficients

    def generate_rdrxn(
        self, ce: ChemicalEquation, use_reagents: bool
    ) -> cif.rdChemReactions.ChemicalReaction:
        """To build the rdkit rdrxn object associated with the ChemicalEquation"""
        return cif.build_rdrxn(
            catalog=ce.catalog,
            role_map=ce.role_map,
            stoichiometry_coefficients=ce.stoichiometry_coefficients,
            use_reagents=use_reagents,
            use_smiles=False,
            use_atom_mapping=True,
            mapping=ce.mapping,
        )

    def generate_smiles(self, rdrxn: cif.rdChemReactions.ChemicalReaction) -> str:
        """To build the smiles associated with the ChemicalEquation"""
        return cif.rdrxn_to_string(rdrxn=rdrxn, out_fmt="smiles", use_atom_mapping=True)

    def generate_template(self, ce: ChemicalEquation) -> Template:
        """To build the template of the ChemicalEquation"""
        tc = TemplateConstructor()
        return tc.build_from_reaction_string(
            reaction_string=ce.smiles, inp_fmt="smiles"
        )

    def generate_disconnection(
        self, ce: ChemicalEquation, desired_product: Molecule
    ) -> Disconnection:
        """To build the disconnection of the ChemicalEquation"""
        dc = DisconnectionConstructor(identity_property_name="smiles")
        return dc.build_from_chemical_equation(ce, desired_product)


class Builder:
    builders = {
        "mapped": MappedChemicalEquationGenerator(),
        "unmapped": UnmappedChemicalEquationGenerator(),
    }
    __builder = None

    def set_builder(self, builder_type: str):
        self.__builder = self.builders[builder_type]

    def get_chemical_equation(
        self,
        chemical_equation_identity_name: str,
        desired_product: Molecule,
        reaction_mols: Union[dict, None] = None,
        basic_attributes: Union[dict, None] = None,
    ) -> ChemicalEquation:
        ce = ChemicalEquation()
        if basic_attributes is None:
            basic_attributes, desired_product = self.__builder.get_basic_attributes(
                reaction_mols, desired_product
            )
        ce.catalog = basic_attributes["catalog"]
        ce.mapping = basic_attributes["mapping"]
        ce.role_map = basic_attributes["role_map"]
        ce.stoichiometry_coefficients = basic_attributes["stoichiometry_coefficients"]
        use_reagents = chemical_equation_identity_name not in ["r_p", "u_r_p"]
        ce.rdrxn = self.__builder.generate_rdrxn(ce, use_reagents)
        ce.smiles = self.__builder.generate_smiles(ce.rdrxn)
        ce.hash_map = calculate_reaction_like_hash_map(ce.catalog, ce.role_map)
        ce.uid = ce.hash_map.get(chemical_equation_identity_name)
        ce.template = self.__builder.generate_template(ce)
        ce.disconnection = self.__builder.generate_disconnection(ce, desired_product)

        return ce


if __name__ == "__main__":
    print("main")
