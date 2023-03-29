from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
import copy

import imagesize

import linchemin.cheminfo.functions as cif
import linchemin.utilities as utilities
from linchemin import settings
from linchemin.cheminfo.models import (ChemicalEquation, Disconnection,
                                       Molecule, Pattern, Ratam, Template)

"""
Module containing the constructor classes of relevant cheminformatics models defined in 'models' module
"""


class UnavailableMolIdentifier(Exception):
    """ To be raised if the selected name for the molecular property name is not among the supported ones"""
    pass


class BadMapping(Exception):
    """ To be raised if an atom number is used more than once, indicating that the atom mapping is invalid"""
    pass


logger = utilities.console_logger(__name__)


# Molecular hash calculations
class MolIdentifierGenerator(ABC):
    """ Abstract class for generator of hash map fragments"""

    @abstractmethod
    def compute_identifier(self, rdmol, hash_map):
        pass


class InchiKeyGenerator(MolIdentifierGenerator):
    """ To compute inch and inchKey """

    def compute_identifier(self, rdmol, hash_map):
        hash_map['inchi'] = cif.Chem.MolToInchi(rdmol)
        hash_map['inchi_key'] = cif.Chem.InchiToInchiKey(hash_map['inchi'])
        return hash_map


class InchiKeyKET15Generator(MolIdentifierGenerator):
    """ To compute InchiKET15T and InchiKeyKET15T """

    def compute_identifier(self, rdmol, hash_map):
        hash_map['inchi_KET_15T'] = cif.Chem.MolToInchi(rdmol, options='-KET -15T')
        hash_map['inchikey_KET_15T'] = cif.Chem.InchiToInchiKey(hash_map['inchi_KET_15T'])
        return hash_map


class NoisoSmilesGenerator(MolIdentifierGenerator):
    """ To compute Noiso smiles """

    def compute_identifier(self, rdmol, hash_map):
        hash_map['noiso_smiles'] = cif.Chem.MolToSmiles(rdmol, isomericSmiles=False)
        return hash_map


class CxSmilesGenerator(MolIdentifierGenerator):
    """ To compute CxSmiles """

    def compute_identifier(self, rdmol, hash_map):
        hash_map['cx_smiles'] = cif.Chem.MolToCXSmiles(rdmol)
        return hash_map


class MolIdentifierFactory:
    """ Factory to give access to the Molecular Identifier Generators """
    molecular_identifiers = {'inchi_key': InchiKeyGenerator,
                             'inchikey_KET_15T': InchiKeyKET15Generator,
                             'noiso_smiles': NoisoSmilesGenerator,
                             'cx_smiles': CxSmilesGenerator,
                             }

    def __init__(self, rdmol=None):
        self.rdmol = rdmol

    def select_identifier(self, identifier_name, hash_map):
        generator = self.molecular_identifiers[identifier_name]
        return generator().compute_identifier(self.rdmol, hash_map)


# Molecule Constructor
class MoleculeConstructor:
    """ Class implementing the constructor of the Molecule class

            Attributes:
                molecular_identity_property_name: a string indicating which kind of input string determines the identity
                                                  of the object (e.g. 'smiles')
                hash_list: a list containing the additional hash values to be computed

    """
    all_available_identifiers = list(cif.HashFunction.names.keys()) + list(
        MolIdentifierFactory().molecular_identifiers.keys()) + ['smiles']

    def __init__(self, molecular_identity_property_name: str = settings.CONSTRUCTORS.molecular_identity_property_name,
                 hash_list: list = settings.CONSTRUCTORS.molecular_hash_list):
        if molecular_identity_property_name not in self.all_available_identifiers:
            logger.error('The selected molecular identity property is not available.'
                         f'Available options are {self.all_available_identifiers}')
            raise UnavailableMolIdentifier

        self.molecular_identity_property_name = molecular_identity_property_name
        self.hash_list = set(hash_list + [self.molecular_identity_property_name])

    def build_from_molecule_string(self, molecule_string: str, inp_fmt: str, ) -> Molecule:
        """ To build a Molecule instance from a string """
        rdmol_input = cif.rdmol_from_string(input_string=molecule_string, inp_fmt=inp_fmt)
        return self.build_from_rdmol(rdmol_input)

    def build_from_rdmol(self, rdmol: cif.Mol) -> Molecule:
        """ To build a Molecule instance from a rdkit Mol instance """
        rdmol_mapped = rdmol
        rdmol_unmapped = cif.remove_rdmol_atom_mapping(rdmol=rdmol_mapped)
        rdmol_unmapped_canonical = cif.canonicalize_rdmol_lite(rdmol=rdmol_unmapped, is_pattern=False)
        rdmol_mapped_canonical = cif.canonicalize_mapped_rdmol(
            cif.canonicalize_rdmol_lite(rdmol=rdmol_mapped, is_pattern=False))
        hash_map = calculate_molecular_hash_values(rdmol=rdmol_unmapped_canonical, hash_list=self.hash_list)
        identity_property = hash_map.get(self.molecular_identity_property_name)
        uid = utilities.create_hash(identity_property)
        smiles = cif.compute_mol_smiles(rdmol=rdmol_unmapped_canonical)
        return Molecule(rdmol=rdmol_unmapped_canonical, rdmol_mapped=rdmol_mapped_canonical,
                        molecular_identity_property_name=self.molecular_identity_property_name, hash_map=hash_map,
                        smiles=smiles, uid=uid, identity_property=identity_property)


# Ratam Constructor
AtomTransformation = namedtuple('AtomTransformation',
                                ['product_uid',
                                 'reactant_uid',
                                 'prod_atom_id',
                                 'react_atom_id',
                                 'map_num'])


class RatamConstructor:
    """ Class implementing the constructor of the Ratam class"""

    def create_ratam(self, molecules_catalog: dict,
                     desired_products: Molecule) -> Ratam:
        """ To initialize an instance of the Ratam class """
        ratam = Ratam()
        ratam.full_map_info = self.get_full_map_info(molecules_catalog, desired_products)
        ratam.atom_transformations = self.get_atom_transformations(ratam.full_map_info)
        return ratam

    def get_full_map_info(self, molecules_catalog: dict,
                          desired_product: Molecule) -> Union[dict, None]:
        """ To create a dictionary mapping each molecule with its list of atom mapping based on its role
            {'reactants': {uid: [map]}, 'reagents': {uid: [map]}, 'products': {uid: [map]}}"""
        full_map_info = {'reactants': {},
                         'reagents': {},
                         'products': {}}
        full_map_info = self.products_map_info(molecules_catalog, full_map_info)
        dp_map_nums = {a.GetAtomMapNum() for a in desired_product.rdmol_mapped.GetAtoms()}
        full_map_info = self.reactants_reagents_map_info(molecules_catalog,
                                                         full_map_info,
                                                         dp_map_nums)

        full_map_info = cif.clean_full_map_info(full_map_info)
        self.mapping_sanity_check(full_map_info)
        return full_map_info

    @staticmethod
    def products_map_info(molecules_catalog: dict,
                          full_map_info: dict) -> dict:
        """ To collect the mapping info for product Molecules """
        for mol in set(molecules_catalog['products']):
            full_map_info['products'][mol.uid] = []
            for m in [n for n in molecules_catalog['products'] if n == mol]:
                mapping = {a.GetIdx(): a.GetAtomMapNum() for a in m.rdmol_mapped.GetAtoms()}
                full_map_info['products'][mol.uid].append(mapping)
        return full_map_info

    @staticmethod
    def reactants_reagents_map_info(molecules_catalog: dict,
                                    full_map_info: dict,
                                    desired_product_map_nums: set) -> dict:
        """ To collect identify reactants and reagents based on the mapping and collect their mapping information """
        for mol in molecules_catalog['reactants'] + molecules_catalog['reagents']:
            if mol.uid not in full_map_info['reactants']:
                full_map_info['reactants'][mol.uid] = []
            if mol.uid not in full_map_info['reagents']:
                full_map_info['reagents'][mol.uid] = []
            map_nums = {a.GetAtomMapNum() for a in mol.rdmol_mapped.GetAtoms()}
            if [n for n in map_nums if n in desired_product_map_nums and n not in [0, -1]]:
                # it's a reactant!
                full_map_info['reactants'][mol.uid].append({a.GetIdx(): a.GetAtomMapNum()
                                                            for a in mol.rdmol_mapped.GetAtoms()})
            else:
                # it's a reagent!
                full_map_info['reagents'][mol.uid].append({a.GetIdx(): a.GetAtomMapNum()
                                                           for a in mol.rdmol_mapped.GetAtoms()})
        return full_map_info

    @staticmethod
    def mapping_sanity_check(full_map_info: dict) -> None:
        """ Mapping sanity check: if a map number appears more than 2 times in the reactants,
            the mapping is invalid and an error is raised """
        map_nums = []
        for uid, map_list in full_map_info['reactants'].items():
            for d in map_list:
                new_nums = list(d.values())
                map_nums.extend(iter(new_nums))
        d = {n: map_nums.count(n) for n in set(map_nums)}
        if [c for n, c in d.items() if c > 1 and n not in [0, -1]]:
            logger.error('Invalid mapping! The same map number is used more than once')
            raise BadMapping

    @staticmethod
    def get_atom_transformations(full_map_info: dict) -> set[AtomTransformation]:
        """ To create the list of AtomTransformations from a catalog of mapped Molecule objects """
        atom_transformations = set()
        for product_uid, prod_maps in full_map_info['products'].items():
            for prod_map in prod_maps:
                for reactant_uid, reactant_maps in full_map_info['reactants'].items():
                    for reactant_map in reactant_maps:
                        if matching_map_num := [map_num for map_num in reactant_map.values() if
                                                map_num in prod_map.values() and map_num not in [0, -1]]:
                            atom_transformations.update(build_atom_transformations(matching_map_num, prod_map,
                                                                                   product_uid, reactant_map,
                                                                                   reactant_uid))
        return atom_transformations


def build_atom_transformations(matching_map_num, prod_map, product_uid, reactant_map, reactant_uid):
    """ To build the list of AtomTransformation objects for each pair of product-reactant with matching map number"""
    ats = []
    for map_num in matching_map_num:
        p_aids = [aid for aid, map in prod_map.items() if map == map_num]
        r_aids = [aid for aid, map in reactant_map.items() if map == map_num]
        ats.extend([AtomTransformation(product_uid, reactant_uid, p_aid, r_aid, map_num)
                    for p_aid in p_aids
                    for r_aid in r_aids])
    return ats


# Disconnection Constructor
class DisconnectionConstructor:
    def __init__(self, identity_property_name: str):
        self.identity_property_name = identity_property_name

    def build_from_chemical_equation(self, chemical_equation: ChemicalEquation,
                                     desired_product: Molecule) -> Union[Disconnection, None]:
        """ To initialize the Disconnection instance and populate its attributes based on the reactive center of the
            ChemicalEquation """

        rxn_reactive_center = RXNReactiveCenter(chemical_equation, desired_product)
        if len(rxn_reactive_center.rxn_bond_info_list) == 0 and \
                len(rxn_reactive_center.rxn_atom_info_list) == 0 and \
                len(rxn_reactive_center.rxn_atomh_info_list) == 0:
            return None

        product_changes = rxn_reactive_center.get_product_changes()
        disconnection = Disconnection()
        disconnection.molecule = desired_product
        disconnection.rdmol = rxn_reactive_center.disconnection_rdmol
        disconnection.reacting_atoms = product_changes.reacting_atoms
        disconnection.hydrogenated_atoms = product_changes.hydrogenated_atoms
        disconnection.new_bonds = product_changes.new_bonds
        disconnection.modified_bonds = product_changes.modified_bonds

        disconnection.hash_map = calculate_disconnection_hash_values(disconnection)
        disconnection.identity_property = disconnection.hash_map.get('disconnection_summary')
        disconnection.uid = utilities.create_hash(disconnection.identity_property)
        disconnection.rdmol_fragmented = self.get_fragments(rdmol=desired_product.rdmol_mapped,
                                                            new_bonds=disconnection.new_bonds,
                                                            fragmentation_method=2)
        return disconnection

    def get_fragments(self, rdmol: cif.Mol, new_bonds: List[int], fragmentation_method: int = 1):
        """ To get the fragments of the desired product Mol. Inspired by
            https://github.com/rdkit/rdkit/issues/2081
        """

        def get_fragments_method_1(rdmol, bonds) -> cif.Mol:
            rdmol_fragmented = cif.rdkit.Chem.FragmentOnBonds(
                rdmol, bonds, addDummies=True,
                # dummyLabels=[(0, 0)]
            )
            return rdmol_fragmented

        def get_fragments_method_2(rdmol: cif.Mol, bonds) -> cif.Mol:
            bonds = [rdmol.GetBondBetweenAtoms(*tup).GetIdx() for tup in bonds]
            mh = cif.Chem.RWMol(cif.Chem.AddHs(rdmol))
            cif.Chem.Kekulize(mh, clearAromaticFlags=True)

            for bond in [mh.GetBondWithIdx(idx) for idx in bonds]:
                a = bond.GetEndAtomIdx()
                b = bond.GetBeginAtomIdx()
                mh.RemoveBond(a, b)

                nAts = mh.GetNumAtoms()
                mh.AddAtom(cif.Chem.Atom(0))
                mh.AddBond(a, nAts, cif.Chem.BondType.SINGLE)
                mh.AddAtom(cif.Chem.Atom(0))
                mh.AddBond(b, nAts + 1, cif.Chem.BondType.SINGLE)
                nAts += 2

            cif.Chem.SanitizeMol(mh)
            cif.rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_NONE
            cif.Chem.RemoveHs(mh)

            return mh

        fragmentation_method_factory = {
            1: get_fragments_method_1,
            2: get_fragments_method_2,
        }

        frag_func = fragmentation_method_factory[fragmentation_method]
        from rdkit.Chem import AllChem
        AllChem.Compute2DCoords(rdmol)
        rdmol_fragmented = frag_func(rdmol=rdmol, bonds=new_bonds)
        return rdmol_fragmented


BondInfo = namedtuple('BondInfo', ('product_atoms', 'product_bond', 'status'))


@dataclass(eq=True, order=False, frozen=False)
class RxnBondInfo:
    """Class for keeping track of bonds that between reactants and products of a reaction are new or changed."""
    product_atoms: tuple
    product_bond: int
    status: str


@dataclass(eq=True, order=False, frozen=False)
class RXNProductChanges:
    """Class for keeping track of the changes in a reaction product"""
    reacting_atoms: list[int]
    hydrogenated_atoms: list[tuple]
    # new_bonds: list[int]
    new_bonds: list[tuple]
    # modified_bonds: list[int]
    modified_bonds: list[tuple]
    rdmol: cif.Mol


class RXNReactiveCenter:
    """
    Class to identify the atoms and bonds that change in a reaction
    inspired from
    https://greglandrum.github.io/rdkit-blog/tutorial/reactions/2021/11/26/highlighting-changed-bonds-in-reactions.html
    """

    def __init__(self, chemical_equation, desired_product):
        chemical_equation.rdrxn.Initialize()
        self.rxn_atom_info_list, self.rxn_atomh_info_list, self.rxn_bond_info_list, self.disconnection_rdmol = self.find_modifications_in_products(
            chemical_equation, desired_product)

    def get_product_changes(self):
        """ To get the changes generated by the reaction: reacting atoms, hydrogenated atoms, created bonds and
            modified bonds. They are returned as attributes of a RXNProductChanges instance"""
        reacting_atoms = sorted([at.prod_atom_id for at in self.rxn_atom_info_list])
        hydrogenated_atoms = sorted([(d['p_atom'], d['delta_hydrogen']) for d in self.rxn_atomh_info_list])

        new_bonds = sorted(
            [bi.product_atoms for bi in self.rxn_bond_info_list if bi.status == 'new'])
        modified_bonds = sorted(
            [bi.product_atoms for bi in self.rxn_bond_info_list if bi.status == 'changed'])
        return RXNProductChanges(reacting_atoms,
                                 hydrogenated_atoms,
                                 new_bonds,
                                 modified_bonds,
                                 self.disconnection_rdmol)

    @staticmethod
    def get_mapped_neighbors(atom: cif.Atom):
        """ To get the mapped neighbors of an atom"""
        res: dict = {}
        amap = atom.GetAtomMapNum()
        if amap == 0:
            return res
        for bond in atom.GetBonds():
            neighbor = next(a for a in [bond.GetEndAtom(), bond.GetBeginAtom()] if a.GetAtomMapNum() != amap)
            n_map = neighbor.GetAtomMapNum()
            if amap > n_map:
                res[(n_map, amap)] = (atom.GetIdx(), neighbor.GetIdx())
            else:
                res[(amap, n_map)] = (neighbor.GetIdx(), atom.GetIdx())

        return res

    def find_modifications_in_products(self, ce: ChemicalEquation,
                                       desired_product: Molecule) -> tuple[list[AtomTransformation],
                                                                           list[dict],
                                                                           list[RxnBondInfo],
                                                                           cif.Mol]:
        """ To identify the list of reacting atoms, hydrogenated atoms and new bonds and modified bonds.
            It returns a 3-tuple """

        # retrieve the map numbers of the reacting atoms
        maps_reacting_atoms = self.get_reacting_atoms_map_numbers(ce)

        # retrieve the AtomTransformations of the desired product involving the reacting atoms
        ats_desired_product = [at for at in ce.mapping.atom_transformations
                               if at.product_uid == desired_product.uid and at.map_num in maps_reacting_atoms]
        disconnection_rdmol = copy.deepcopy(desired_product.rdmol_mapped)
        seen = set()
        bonds = []
        hydrogenated_atoms = []
        # for each AtomTransformation of involving the reacting atoms..
        for at in ats_desired_product:
            # ...the involved atom of the desired product and of the reactant are identified
            reactant = next(mol for h, mol in ce.catalog.items() if h == at.reactant_uid)
            r_atom, p_atom = self.get_changing_atoms(at, disconnection_rdmol, reactant.rdmol_mapped)
            # if the atom changes the number of bonded hydrogens it's identified and the variation in the number
            # of hydrogens is computed
            delta_h = p_atom.GetTotalNumHs() - r_atom.GetTotalNumHs()
            if delta_h != 0:
                hydrogenated_atoms.append({'p_atom': p_atom.GetIdx(), 'delta_hydrogen': delta_h})

            # based on their neighbors, new and modified bonds are identified
            bonds, seen = self.get_bond_info(r_atom,
                                             reactant,
                                             p_atom,
                                             disconnection_rdmol,
                                             seen,
                                             bonds)

        rxn_bond_info_list = [RxnBondInfo(**item._asdict()) for item in bonds]
        return ats_desired_product, hydrogenated_atoms, rxn_bond_info_list, disconnection_rdmol

    @staticmethod
    def get_reacting_atoms_map_numbers(ce: ChemicalEquation) -> list[int]:
        """" To identify the map numbers associated with the reacting atoms in a ChemicalEquation """
        ce.rdrxn.Initialize()
        reactingAtoms = ce.rdrxn.GetReactingAtoms()
        maps_reacting_atoms = []
        for ridx, reacting in enumerate(reactingAtoms):
            r = ce.rdrxn.GetReactantTemplate(ridx)
            maps_reacting_atoms.extend(r.GetAtomWithIdx(raidx).GetAtomMapNum() for raidx in reacting)
        return maps_reacting_atoms

    @staticmethod
    def get_changing_atoms(at, product_rdmol, reactant_rdmol):
        """ To identify the atoms involved in an AtomTransformation """
        r_atom = next(atom for atom in reactant_rdmol.GetAtoms()
                      if atom.GetIdx() == at.react_atom_id)
        p_atom = next(atom for atom in product_rdmol.GetAtoms()
                      if atom.GetIdx() == at.prod_atom_id)
        return r_atom, p_atom

    def get_bond_info(self, r_atom: cif.Atom,
                      reactant: Molecule,
                      p_atom: cif.Atom,
                      product: cif.Mol,
                      seen: set,
                      bonds: list):
        """ To extract the information regarding new or modified bonds in a ChemicalEquation"""
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
                bonds.extend([BondInfo(product_atoms=pnbrs[tpl],
                                       product_bond=pbond.GetIdx(),
                                       status='new')])
            else:
                # present in both reactants and products, check to see if it changed
                rbond = reactant.rdmol_mapped.GetBondBetweenAtoms(*rnbrs[tpl])
                if rbond.GetBondType() != pbond.GetBondType():
                    bonds.extend([BondInfo(product_atoms=pnbrs[tpl],
                                           product_bond=pbond.GetIdx(),
                                           status='changed')])

        return bonds, seen


# Pattern Constructor
class PatternConstructor:
    """ Class implementing the constructor of the Pattern class """

    def __init__(self, identity_property_name: str = settings.CONSTRUCTORS.pattern_identity_property_name):
        self.identity_property_name = identity_property_name

    def create_pattern(self, rdmol: cif.Mol):
        pattern = Pattern()
        rdmol_mapped = rdmol
        rdmol_unmapped = cif.remove_rdmol_atom_mapping(rdmol=rdmol_mapped)
        rdmol_unmapped_canonical = cif.canonicalize_rdmol_lite(rdmol=rdmol_unmapped, is_pattern=True)
        rdmol_mapped_canonical = cif.canonicalize_rdmol_lite(rdmol=rdmol_mapped, is_pattern=True)
        pattern.rdmol = rdmol_unmapped_canonical
        pattern.rdmol_mapped = rdmol_mapped_canonical
        pattern.smarts = cif.compute_mol_smarts(rdmol=pattern.rdmol)
        pattern.hash_map = calculate_pattern_hash_values(pattern.smarts)

        pattern.identity_property = pattern.hash_map.get(self.identity_property_name)
        pattern.uid = utilities.create_hash(pattern.identity_property)  # the hashed identity property
        return pattern

    def build_from_molecule_string(self, molecule_string: str, inp_fmt: str, ) -> Pattern:
        rdmol_input = cif.rdmol_from_string(input_string=molecule_string, inp_fmt=inp_fmt)
        return self.create_pattern(rdmol_input)

    def build_from_rdmol(self, rdmol: cif.Mol) -> Pattern:
        return self.create_pattern(rdmol)


# Template Constructor
class TemplateConstructor:
    def __init__(self, identity_property_name: str = settings.CONSTRUCTORS.pattern_identity_property_name):
        """ To initialize a Template instance. The machinery is based on rdchiral."""
        self.identity_property_name = identity_property_name

    def read_reaction(self, reaction_string: str,
                      inp_fmt: str) -> Tuple[cif.rdChemReactions.ChemicalReaction,
                                             utilities.OutcomeMetadata]:
        """ To attempt in sanitizing the rdkit reaction """
        try:
            rdrxn = cif.rdrxn_from_string(input_string=reaction_string, inp_fmt=inp_fmt)
            sanitize_msg = cif.rdChemReactions.SanitizeRxn(rdrxn, catchErrors=True)

            outcome = utilities.OutcomeMetadata(name='read_reaction', is_successful=True,
                                                log={'sanitize_msg': sanitize_msg})
        except Exception as e:
            outcome = utilities.OutcomeMetadata(name='read_reaction', is_successful=False,
                                                log={'exception': e})
            rdrxn = None

        return rdrxn, outcome

    def unpack_rdrxn(self, rdrxn: cif.rdChemReactions.ChemicalReaction) -> tuple[dict, dict, dict]:
        """ To build the basic attributes of the Template using the ChemicalEquation Builder """
        constructor = PatternConstructor(identity_property_name=self.identity_property_name)
        reaction_mols = cif.rdrxn_to_molecule_catalog(rdrxn, constructor)
        builder = UnmappedChemicalEquationGenerator()
        attributes, _ = builder.get_basic_attributes(reaction_mols, None)
        return attributes['catalog'], attributes['stoichiometry_coefficients'], attributes['role_map']

    def build_from_rdrxn(self, rdrxn: cif.rdChemReactions.ChemicalReaction) -> Union[Template, None]:
        """ To initialize the instance based on an rdkit reaction object"""
        reaction_string = cif.rdChemReactions.ReactionToSmarts(rdrxn)
        return self.build_from_reaction_string(reaction_string=reaction_string, inp_fmt='smarts')

    def build_from_reaction_string(self, reaction_string: str,
                                   inp_fmt: str) -> Union[Template, None]:
        """ To initialize the instance based on a reaction string """
        rdchiral_output = cif.rdchiral_extract_template(reaction_string=reaction_string, inp_fmt=inp_fmt,
                                                        reaction_id=None)
        return self.build_from_rdchiral_output(rdchiral_output=rdchiral_output)

    def create_rdchiral_data(self, rdchiral_output: Dict):
        """ To build the data necessary to usage of rdchiral """
        reaction_rwd_smarts = rdchiral_output.get('reaction_smarts')
        reaction_fwd_smarts = '>'.join(reaction_rwd_smarts.split('>')[::-1])
        rdrxn, read_outcome = self.read_reaction(reaction_string=reaction_fwd_smarts, inp_fmt='smarts')
        rdchiral_data = rdchiral_output.copy()
        rdchiral_data.pop('reaction_id')
        rdchiral_data.pop('reaction_smarts')
        rdchiral_data['reaction_fwd_smarts'] = reaction_fwd_smarts
        rdchiral_data['reaction_rwd_smarts'] = reaction_rwd_smarts
        return rdrxn, rdchiral_data

    def build_from_rdchiral_output(self, rdchiral_output: Dict) -> Union[Template, None]:
        """ To build the Template attributes based on rdchiral machinery.
            If rdchiral process fails, None is returned """
        if not rdchiral_output:
            return None
        if not rdchiral_output.get('reaction_smarts'):
            return None
        template = Template()
        try:
            rdrxn, rdchiral_data = self.create_rdchiral_data(rdchiral_output)
        except Exception:
            logger.warning('Issues with rdchiral calculation of the template. None is returned.')
            return None
        template.rdrxn = rdrxn
        template.rdchiral_data = rdchiral_data
        pattern_catalog, stoichiometry_coefficients, role_map = self.unpack_rdrxn(rdrxn=rdrxn)
        template.pattern_catalog = pattern_catalog
        template.stoichiometry_coefficients = stoichiometry_coefficients
        template.role_map = role_map
        template.hash_map = create_reaction_like_hash_values(pattern_catalog, role_map)
        template.uid = template.hash_map.get(settings.CONSTRUCTORS.template_identity_property)  # TODO: review

        template.rdrxn = cif.build_rdrxn(catalog=pattern_catalog,
                                         role_map=role_map,
                                         stoichiometry_coefficients=stoichiometry_coefficients,
                                         use_reagents=True,
                                         use_smiles=False,
                                         use_atom_mapping=True)
        template.smarts = cif.rdrxn_to_string(rdrxn=template.rdrxn, out_fmt='smarts')

        return template


# ChemicalEquation Constructor
class ChemicalEquationConstructor:
    """ Class implementing the constructor of the ChemicalEquation class

        Attributes:
            molecular_identity_property_name: a string indicating the property determining the identity
                                              of the molecules in the chemical equation (e.g. 'smiles')

    """

    def __init__(self, molecular_identity_property_name: str = settings.CONSTRUCTORS.molecular_identity_property_name,
                 chemical_equation_identity_name: str = settings.CONSTRUCTORS.chemical_equation_identity_name):

        self.molecular_identity_property_name = molecular_identity_property_name
        self.chemical_equation_identity_name = chemical_equation_identity_name

    def read_reaction(self, reaction_string: str,
                      inp_fmt: str) -> Tuple[cif.rdChemReactions.ChemicalReaction,
                                             utilities.OutcomeMetadata]:
        """ To start the building of a ChemicalEquation instance from a reaction string """
        try:
            rdrxn = cif.rdrxn_from_string(input_string=reaction_string, inp_fmt=inp_fmt)
            sanitize_msg = cif.rdChemReactions.SanitizeRxn(rdrxn, catchErrors=True)

            outcome = utilities.OutcomeMetadata(name='read_reaction', is_successful=True,
                                                log={'sanitize_msg': sanitize_msg})
        except Exception as e:
            outcome = utilities.OutcomeMetadata(name='read_reaction', is_successful=False,
                                                log={'exception': e})
            rdrxn = None

        return rdrxn, outcome

    def unpack_rdrxn(self, rdrxn: cif.rdChemReactions.ChemicalReaction,
                     desired_product: Union[cif.Mol, None]) -> ChemicalEquation:
        """ To compute ChemicalEquation attributes from the associated rdkit ChemicalReaction object """
        constructor = MoleculeConstructor(molecular_identity_property_name=self.molecular_identity_property_name)
        return create_chemical_equation(rdrxn=rdrxn,
                                        chemical_equation_identity_name=self.chemical_equation_identity_name,
                                        constructor=constructor,
                                        desired_product=desired_product)

    def build_from_rdrxn(self, rdrxn: cif.rdChemReactions.ChemicalReaction,
                         desired_product: Union[
                             cif.Mol, None] = settings.CONSTRUCTORS.desired_product) -> ChemicalEquation:
        """ To build a ChemicalEquation instance from a rdkit ChemicalReaction object """
        return self.unpack_rdrxn(rdrxn=rdrxn,
                                 desired_product=desired_product)

    def build_from_reaction_string(self, reaction_string: str,
                                   inp_fmt: str,
                                   desired_product: Union[
                                       str, None] = settings.CONSTRUCTORS.desired_product) -> ChemicalEquation:
        """ To build a ChemicalEquation instance from a reaction string """
        rdrxn, read_outcome = self.read_reaction(reaction_string=reaction_string, inp_fmt=inp_fmt)
        if desired_product:
            desired_product = cif.rdmol_from_string(desired_product, inp_fmt=inp_fmt)
        return self.build_from_rdrxn(rdrxn=rdrxn, desired_product=desired_product)


def create_chemical_equation(rdrxn: cif.rdChemReactions.ChemicalReaction,
                             chemical_equation_identity_name: str,
                             constructor: MoleculeConstructor,
                             desired_product: Union[Molecule, None]) -> ChemicalEquation:
    """ Initializes the correct builder of the ChemicalEquation based on the presence of the atom mapping.

        :param:
            rdrxn: the initial RDKit ChemReaction object

            chemical_equation_identity_name: a string indicating which representation of the ChemicalEquation determines
                                             its identity

            constructor: the constructor for the Molecule objects involved

        :return:
            a new ChemicalEquation instance
    """
    builder_type = 'mapped' if cif.has_mapped_products(rdrxn) else 'unmapped'
    reaction_mols = cif.rdrxn_to_molecule_catalog(rdrxn, constructor)
    if desired_product is not None:
        desired_product_mol = constructor.build_from_rdmol(desired_product)
        if desired_product_mol not in reaction_mols['products']:
            logger.error('The selected product does not appear in the reaction. ')
            raise ValueError
    else:
        desired_product_mol = cif.select_desired_product(reaction_mols)
    builder = Builder()
    builder.set_builder(builder_type)
    return builder.get_chemical_equation(reaction_mols,
                                         chemical_equation_identity_name,
                                         desired_product_mol)


class ChemicalEquationGenerator(ABC):
    """ Abstract class for ChemicalEquationAttributesGenerator """

    def get_basic_attributes(self,
                             reaction_mols: dict,
                             desired_product: Union[Molecule, None]) -> tuple[dict, Molecule]:
        pass

    @abstractmethod
    def generate_template(self,
                          chemical_equation: ChemicalEquation) -> Union[Template, None]:
        pass

    @abstractmethod
    def generate_disconnection(self,
                               chemical_equation: ChemicalEquation,
                               desired_product) -> Union[Disconnection, None]:
        pass

    @abstractmethod
    def generate_rdrxn(self,
                       ce: ChemicalEquation,
                       use_reagents: bool) -> cif.rdChemReactions.ChemicalReaction:
        pass

    @abstractmethod
    def generate_smiles(self,
                        rdrxn: cif.rdChemReactions.ChemicalReaction) -> str:
        pass


class UnmappedChemicalEquationGenerator(ChemicalEquationGenerator):

    def get_basic_attributes(self,
                             reaction_mols: dict,
                             desired_product: Union[Molecule, None]) -> tuple[dict, Molecule]:
        """ To build the initial attributes of a ChemicalEquation: mapping, role_map, catalog and
        stoichiometry_coefficients. These are returned in a dictionary. """
        basic_attributes = {'mapping': None,
                            'role_map': {role: sorted(list({m.uid for m in set(mols)})) for role, mols in
                                         reaction_mols.items()}}
        all_molecules = reaction_mols['reactants'] + reaction_mols['reagents'] + reaction_mols['products']
        basic_attributes['catalog'] = {m.uid: m for m in set(all_molecules)}
        basic_attributes['stoichiometry_coefficients'] = self.generate_stoichiometry_coefficients(reaction_mols)
        return basic_attributes, desired_product

    def generate_rdrxn(self, ce: ChemicalEquation,
                       use_reagents: bool) -> cif.rdChemReactions.ChemicalReaction:
        """ To build the rdkit rdrxn object associated with the ChemicalEquation """
        return cif.build_rdrxn(catalog=ce.catalog,
                               role_map=ce.role_map,
                               stoichiometry_coefficients=ce.stoichiometry_coefficients,
                               use_reagents=use_reagents,
                               use_smiles=False,
                               use_atom_mapping=False,
                               mapping=ce.mapping)

    def generate_smiles(self, rdrxn: cif.rdChemReactions.ChemicalReaction) -> str:
        """ To build the smiles associated with the ChemicalEquation """
        return cif.rdrxn_to_string(rdrxn=rdrxn, out_fmt='smiles', use_atom_mapping=False)

    @staticmethod
    def generate_stoichiometry_coefficients(reaction_mols: dict) -> dict:
        """ To build the dictionary of the stoichiometry coefficients of an unmapped ChemicalEquation """
        stoichiometry_coefficients = {}
        for role in ['reactants', 'reagents', 'products']:
            molecules = reaction_mols[role]
            molecule_coeffs = {m.uid: molecules.count(m) for m in set(molecules)}
            stoichiometry_coefficients[role] = molecule_coeffs
        return stoichiometry_coefficients

    def generate_template(self, ce):
        """ To generate the template. It is None if the ChemicalEquation is unmapped. """
        return None

    def generate_disconnection(self, ce, desired_product):
        """ To generate the disconnection. It is None if the ChemicalEquation is unmapped. """
        return None


class MappedChemicalEquationGenerator(ChemicalEquationGenerator):

    def get_basic_attributes(self,
                             reaction_mols: dict,
                             desired_product: Molecule) -> tuple[dict, Molecule]:
        """ To build the initial attributes of a ChemicalEquation: mapping, role_map, catalog and
            stoichiometry_coefficients. These are returned in a dictionary. """
        basic_attributes = {'mapping': self.generate_mapping(reaction_mols, desired_product)}
        basic_attributes['role_map'] = {role: sorted(list(map_info.keys()))
                                        for role, map_info in basic_attributes['mapping'].full_map_info.items()}

        all_molecules = reaction_mols['reactants'] + reaction_mols['reagents'] + reaction_mols['products']
        basic_attributes['catalog'] = {m.uid: m for m in set(all_molecules)}

        basic_attributes['stoichiometry_coefficients'] = self.generate_stoichiometry_coefficients(
            basic_attributes['mapping'])
        return basic_attributes, desired_product

    @staticmethod
    def generate_mapping(new_reaction_mols: dict, desired_product) -> Ratam:
        """ To generate the Ratam instance for the ChemicalEquation """
        ratam_constructor = RatamConstructor()
        return ratam_constructor.create_ratam(new_reaction_mols, desired_product)

    @staticmethod
    def generate_stoichiometry_coefficients(mapping: Ratam) -> dict:
        """ To build the dictionary of the stoichiometry coefficients of a mapped ChemicalEquation """
        stoichiometry_coefficients = {'reactants': {}, 'reagents': {}, 'products': {}}
        for role, mapping_info in mapping.full_map_info.items():
            for uid, map_list in mapping_info.items():
                stoichiometry_coefficients[role].update({uid: len(map_list)})
        return stoichiometry_coefficients

    def generate_rdrxn(self,
                       ce: ChemicalEquation,
                       use_reagents: bool) -> cif.rdChemReactions.ChemicalReaction:
        """ To build the rdkit rdrxn object associated with the ChemicalEquation """
        return cif.build_rdrxn(catalog=ce.catalog,
                               role_map=ce.role_map,
                               stoichiometry_coefficients=ce.stoichiometry_coefficients,
                               use_reagents=use_reagents,
                               use_smiles=False,
                               use_atom_mapping=True,
                               mapping=ce.mapping)

    def generate_smiles(self,
                        rdrxn: cif.rdChemReactions.ChemicalReaction) -> str:
        """ To build the smiles associated with the ChemicalEquation """
        return cif.rdrxn_to_string(rdrxn=rdrxn, out_fmt='smiles', use_atom_mapping=True)

    def generate_template(self,
                          ce: ChemicalEquation) -> Template:
        """ To build the template of the ChemicalEquation """
        tc = TemplateConstructor()
        return tc.build_from_reaction_string(reaction_string=ce.smiles, inp_fmt='smiles')

    def generate_disconnection(self,
                               ce: ChemicalEquation,
                               desired_product: Molecule) -> Disconnection:
        """ To build the disconnection of the ChemicalEquation """
        dc = DisconnectionConstructor(identity_property_name='smiles')
        return dc.build_from_chemical_equation(ce, desired_product)


class Builder:
    builders = {'mapped': MappedChemicalEquationGenerator(),
                'unmapped': UnmappedChemicalEquationGenerator()}
    __builder = None

    def set_builder(self, builder_type: str):
        self.__builder = self.builders[builder_type]

    def get_chemical_equation(self, reaction_mols: dict,
                              chemical_equation_identity_name: str,
                              desired_product: Molecule) -> ChemicalEquation:
        ce = ChemicalEquation()
        basic_attributes, desired_product = self.__builder.get_basic_attributes(reaction_mols, desired_product)
        ce.catalog = basic_attributes['catalog']
        ce.mapping = basic_attributes['mapping']
        ce.role_map = basic_attributes['role_map']
        ce.stoichiometry_coefficients = basic_attributes['stoichiometry_coefficients']
        use_reagents = chemical_equation_identity_name not in ['r_p', 'u_r_p']
        ce.rdrxn = self.__builder.generate_rdrxn(ce, use_reagents)
        ce.smiles = self.__builder.generate_smiles(ce.rdrxn)
        ce.hash_map = create_reaction_like_hash_values(ce.catalog, ce.role_map)
        ce.uid = ce.hash_map.get(chemical_equation_identity_name)
        ce.template = self.__builder.generate_template(ce)
        ce.disconnection = self.__builder.generate_disconnection(ce, desired_product)

        return ce


def calculate_molecular_hash_values(rdmol: cif.Mol, hash_list: Union[set, None] = None) -> dict:
    """ To compute the hash_map dictionary containing molecular properties/representations names and the
        corresponding hash values """
    molhashf = cif.HashFunction.names
    if hash_list is None:
        hash_list = MoleculeConstructor.all_available_identifiers
    hash_map = {}

    if rdkit_hashes := [h for h in hash_list if h in molhashf]:
        hash_map |= {k: cif.MolHash(rdmol, v) for k, v in molhashf.items() if k in rdkit_hashes}
    if 'smiles' in hash_list:
        hash_map |= {'smiles': cif.MolHash(rdmol, v) for k, v in molhashf.items() if k == 'CanonicalSmiles'}

    if other_hashes := [h for h in hash_list if h not in rdkit_hashes]:
        factory = MolIdentifierFactory(rdmol)
        for h in other_hashes:
            if h not in MoleculeConstructor.all_available_identifiers:
                logger.warning(f'{h} is not supported as molecular identifier')
            elif h != 'smiles':
                hash_map |= factory.select_identifier(h, hash_map)

    """

    hash_map['ExtendedMurcko_AG'] = smiles_to_anonymus_graph(hash_map['ExtendedMurcko'])
    hash_map['ExtendedMurcko_EG'] = smiles_to_element_graph(hash_map['ExtendedMurcko'])
    hash_map['MurckoScaffold_AG'] = smiles_to_anonymus_graph(hash_map['MurckoScaffold'])
    hash_map['MurckoScaffold_EG'] = smiles_to_element_graph(hash_map['MurckoScaffold'])
    """
    return hash_map


# ChemicalEquation hash calculations
def create_reaction_like_hash_values(catalog, role_map):
    """ To calculate the hash keys for reaction-like objects """
    mol_list_map = {role: [catalog.get(uid) for uid in uid_list]
                    for role, uid_list in role_map.items()}

    # get the identity_property for each molecule
    idp_list_map = {role: [m.identity_property for m in molecule_list]
                    for role, molecule_list in mol_list_map.items()}

    # for each role concatenate the properties
    idp_str_map = {role: '.'.join(sorted(v)) for role, v in idp_list_map.items()}

    # add some more strings
    idp_str_map['r_p'] = '>>'.join([idp_str_map.get('reactants'), idp_str_map.get('products')])
    idp_str_map['r_r_p'] = '>'.join(
        [idp_str_map.get('reactants'), idp_str_map.get('reagents'), idp_str_map.get('products')])
    idp_str_map['u_r_p'] = '>>'.join(sorted([idp_str_map.get('reactants'), idp_str_map.get('products')]))
    idp_str_map['u_r_r_p'] = '>>'.join(
        sorted([idp_str_map.get('reactants'), idp_str_map.get('reagents'), idp_str_map.get('products')]))
    return {role: utilities.create_hash(v) for role, v in idp_str_map.items()}


def calculate_disconnection_hash_values(disconnection):
    idp = disconnection.molecule.identity_property

    changes_map = {
        'reacting_atoms': disconnection.reacting_atoms,
        'hydrogenated_atoms': disconnection.hydrogenated_atoms,
        'new_bonds': disconnection.new_bonds,
        'mod_bonds': disconnection.modified_bonds,

    }

    """
    | separates properties and is followed by the name and a:  
    """
    changes_str = '|'.join([f'{k}:{",".join(map(str, v))}' for k, v in changes_map.items()])

    disconnection_summary = '|'.join([idp, changes_str])

    return {'disconnection_summary': disconnection_summary}


def calculate_pattern_hash_values(smarts):
    return {'smarts': smarts}


if __name__ == '__main__':
    print('main')
