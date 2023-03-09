from __future__ import annotations

from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

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
AtomTransformation = namedtuple('AtomTransformation', ['product_uid', 'reactant_uid', 'prod_atom_id', 'react_atom_id',
                                                       'map_num'])


class RatamConstructor:
    """ Class implementing the constructor of the Ratam class"""

    def create_ratam(self, molecules_catalog: dict):
        ratam = Ratam()
        ratam.full_map_info = self.get_full_map_info(molecules_catalog)
        ratam.atom_transformations = self.get_atom_transformations(molecules_catalog, ratam.full_map_info)
        return ratam

    def get_full_map_info(self, molecules_catalog: dict) -> dict:
        full_map_info: dict = {}
        all_molecules = molecules_catalog['reactants_reagents'] + molecules_catalog['products']
        for mol in all_molecules:
            full_map_info[mol.uid] = []
            m_appearances = [m for m in all_molecules if m == mol]
            for m in m_appearances:
                mapping = {}
                for a in m.rdmol_mapped.GetAtoms():
                    if isinstance(a.GetAtomMapNum(), int):
                        mapping[a.GetIdx()] = a.GetAtomMapNum()
                    else:
                        mapping[a.GetIdx()] = -1
                full_map_info[mol.uid].append(mapping)
        self.mapping_sanity_check(full_map_info)
        return full_map_info

    def mapping_sanity_check(self, full_map_info):
        """ Mapping sanity check: if a map number appears more than 2 times, it means that it is used more than once
            and thus the mapping is invalid and an error is raised
        """
        map_nums = []
        for map_list in full_map_info.values():
            for d in map_list:
                new_nums = list(d.values())
                map_nums.extend(iter(new_nums))
        for n in map_nums:
            if map_nums.count(n) > 2 and n not in [0, -1]:
                logger.error('Invalid mapping! The same map number is used more than once')
                raise BadMapping

    def get_atom_transformations(self, reaction_mols: dict, full_map_info: dict):
        """ To create the list of AtomTransformations from a catalog of mapped Molecule objects """
        atom_transformations = set()
        for product in reaction_mols['products']:
            prod_maps = full_map_info[product.uid]
            for prod_map in prod_maps:
                for reactant in reaction_mols['reactants_reagents']:
                    reactant_maps = full_map_info[reactant.uid]
                    for reactant_map in reactant_maps:
                        if matching_map_num := [map_num for map_num in reactant_map.values() if
                                                map_num in prod_map.values() and map_num not in [0, -1]]:
                            atom_transformations.update(build_atom_transformations(matching_map_num, prod_map,
                                                                                   product.uid, reactant_map,
                                                                                   reactant.uid))
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

    def build_from_rdrxn(self, rdrxn: cif.rdChemReactions.ChemicalReaction, desired_product_idx: int = 0) -> \
            Union[Disconnection, None]:

        rxn_reactive_center = RXNReactiveCenter(rdrxn=rdrxn)

        if len(rxn_reactive_center.rxn_bond_info_list) == 0 and \
                len(rxn_reactive_center.rxn_atom_info_list) == 0 and \
                len(rxn_reactive_center.rxn_atomh_info_list) == 0:
            return None

        product_rdmol = cif.Chem.Mol(rdrxn.GetProductTemplate(desired_product_idx))
        product_changes = rxn_reactive_center.get_product_changes(product_idx=desired_product_idx)

        molecule_constructor = MoleculeConstructor(molecular_identity_property_name=self.identity_property_name)
        product_molecule = molecule_constructor.build_from_rdmol(rdmol=product_rdmol)
        disconnection = Disconnection()
        disconnection.molecule = product_molecule
        disconnection.rdmol = product_molecule.rdmol
        reacting_atoms, hydrogenated_atoms, new_bonds, modified_bonds = self.re_map(product_changes,
                                                                                    rdmol_old=product_rdmol,
                                                                                    rdmol_new=product_molecule.rdmol)
        disconnection.reacting_atoms = reacting_atoms
        disconnection.hydrogenated_atoms = hydrogenated_atoms
        disconnection.new_bonds = new_bonds
        disconnection.modified_bonds = modified_bonds

        disconnection.hash_map = calculate_disconnection_hash_values(disconnection)
        disconnection.identity_property = disconnection.hash_map.get('disconnection_summary')
        disconnection.uid = utilities.create_hash(disconnection.identity_property)

        disconnection.rdmol_fragmented = self.get_fragments(rdmol=product_rdmol, new_bonds=new_bonds,
                                                            fragmentation_method=2)

        return disconnection

    @staticmethod
    def re_map(product_changes, rdmol_old, rdmol_new):
        """
        rdmol_old and  rdmol_new differ by atom and bond ordering
        this function maps some atoms and bonds from the rdmol_old to rdmol_new

        """
        reacting_atoms_ = product_changes.reacting_atoms
        hydrogenated_atoms_ = product_changes.hydrogenated_atoms
        new_bonds_ = product_changes.new_bonds
        modified_bonds_ = product_changes.modified_bonds

        match = rdmol_new.GetSubstructMatch(rdmol_old)
        reacting_atoms = sorted([match[x] for x in reacting_atoms_])
        hydrogenated_atoms = sorted([match[x] for x in hydrogenated_atoms_])
        new_bonds = sorted(
            [rdmol_new.GetBondBetweenAtoms(match[bond.GetBeginAtomIdx()], match[bond.GetEndAtomIdx()]).GetIdx()
             for bond in rdmol_old.GetBonds() if bond.GetIdx() in new_bonds_])

        modified_bonds = sorted([
            rdmol_new.GetBondBetweenAtoms(match[bond.GetBeginAtomIdx()], match[bond.GetEndAtomIdx()]).GetIdx()
            for bond in rdmol_old.GetBonds() if bond.GetIdx() in modified_bonds_])

        return reacting_atoms, hydrogenated_atoms, new_bonds, modified_bonds

    def build_from_reaction_string(self, reaction_string: str, inp_fmt: str) -> Union[Disconnection, None]:
        rdrxn = cif.rdrxn_from_string(input_string=reaction_string, inp_fmt=inp_fmt)
        rdrxn.Initialize()
        rdrxn.Validate()
        cif.Chem.rdChemReactions.PreprocessReaction(rdrxn)
        return self.build_from_rdrxn(rdrxn=rdrxn)

    def get_fragments(self, rdmol: cif.Mol, new_bonds: List[int], fragmentation_method: int = 1):
        """
        inspired by
            https://github.com/rdkit/rdkit/issues/2081

        """

        def get_fragments_method_1(rdmol, bonds) -> cif.Mol:
            rdmol_fragmented = cif.rdkit.Chem.FragmentOnBonds(
                rdmol, bonds, addDummies=True,
                # dummyLabels=[(0, 0)]
            )
            return rdmol_fragmented

        def get_fragments_method_2(rdmol: cif.Mol, bonds) -> cif.Mol:
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


AtomInfo = namedtuple('AtomInfo', ('mapnum', 'reactant', 'reactant_atom', 'product', 'product_atom'))
BondInfo = namedtuple('BondInfo', ('product', 'product_atoms', 'product_bond', 'status'))


@dataclass(eq=True, order=False, frozen=False)
class RxnAtomInfo:
    """Class for keeping track of atoms involved in a reaction."""
    mapnum: int
    reactant: int
    reactant_atom: int
    product: int
    product_atom: int


@dataclass(eq=True, order=False, frozen=False)
class RxnBondInfo:
    """Class for keeping track of bonds that between reactants and products of a reaction are new or changed."""
    product: int
    product_atoms: tuple
    product_bond: int
    status: str


@dataclass(eq=True, order=False, frozen=False)
class RXNProductChanges:
    """Class for keeping track of the changes in a reaction product"""
    reacting_atoms: List[int]
    hydrogenated_atoms: List[int]
    new_bonds: List[int]
    modified_bonds: List[int]


class RXNReactiveCenter:
    """
    Class to identify the atoms and bonds that change in a reaction
    inspired from
    https://greglandrum.github.io/rdkit-blog/tutorial/reactions/2021/11/26/highlighting-changed-bonds-in-reactions.html
    """

    def __init__(self, rdrxn: cif.rdChemReactions.ChemicalReaction):
        rdrxn.Initialize()
        self.rxn_atom_info_list, self.rxn_atomh_info_list, self.rxn_bond_info_list, = self.find_modifications_in_products(
            rdrxn)

    def get_product_changes(self, product_idx: int = 0):
        reacting_atoms = sorted([ai.product_atom for ai in self.rxn_atom_info_list if ai.product == product_idx])
        hydrogenated_atoms = sorted([ai.product_atom for ai in self.rxn_atomh_info_list if ai.product == product_idx])

        new_bonds = sorted(
            [bi.product_bond for bi in self.rxn_bond_info_list if bi.product == product_idx and bi.status == 'new'])
        modified_bonds = sorted(
            [bi.product_bond for bi in self.rxn_bond_info_list if
             bi.product == product_idx and bi.status == 'changed'])

        return RXNProductChanges(reacting_atoms, hydrogenated_atoms, new_bonds, modified_bonds)

    @staticmethod
    def map_reacting_atoms_to_products(rdrxn: cif.rdChemReactions.ChemicalReaction, reactingAtoms):
        """ figures out which atoms in the products each mapped atom in the reactants maps to """
        res = []
        for ridx, reacting in enumerate(reactingAtoms):
            reactant = rdrxn.GetReactantTemplate(ridx)
            for raidx in reacting:
                mapnum = reactant.GetAtomWithIdx(raidx).GetAtomMapNum()
                foundit = False
                for pidx, product in enumerate(rdrxn.GetProducts()):
                    for paidx, patom in enumerate(product.GetAtoms()):
                        if patom.GetAtomMapNum() == mapnum:
                            res.append(AtomInfo(mapnum, ridx, raidx, pidx, paidx))
                            foundit = True
                            break
                        if foundit:
                            break
        return res

    @staticmethod
    def get_mapped_neighbors(atom: cif.Atom):
        """ test all mapped neighbors of a mapped atom"""
        res: dict = {}
        amap = atom.GetAtomMapNum()
        if not amap:
            return res
        for nbr in atom.GetNeighbors():
            if nmap := nbr.GetAtomMapNum():
                if amap > nmap:
                    res[(nmap, amap)] = (atom.GetIdx(), nbr.GetIdx())
                else:
                    res[(amap, nmap)] = (nbr.GetIdx(), atom.GetIdx())
        return res

    def find_modifications_in_products(self, rxn) -> tuple[List[RxnAtomInfo], List[RxnAtomInfo], List[RxnBondInfo]]:
        """ returns a 3-tuple with the modified atoms and bonds from the reaction """
        reactingAtoms = rxn.GetReactingAtoms()
        amap = self.map_reacting_atoms_to_products(rxn, reactingAtoms)
        res = []
        seen = set()
        amaph = []

        # this is all driven from the list of reacting atoms:
        for itm in amap:
            _, ridx, raidx, pidx, paidx = itm
            reactant = rxn.GetReactantTemplate(ridx)
            ratom = reactant.GetAtomWithIdx(raidx)
            product = rxn.GetProductTemplate(pidx)
            patom = product.GetAtomWithIdx(paidx)

            # check if the number of hydrogen (total or explicit or implicit) attached to an atom has increased when moving from reactant to product
            q = patom.GetTotalNumHs() > ratom.GetTotalNumHs() or patom.GetNumExplicitHs() > ratom.GetNumExplicitHs() or patom.GetNumImplicitHs() > ratom.GetNumImplicitHs()
            # print('\n', q)
            # print(f"idx:symbol explicit/implicit/total {ratom.GetIdx()}:{ratom.GetSymbol()} {ratom.GetNumExplicitHs()}/{ratom.GetNumImplicitHs()}/{ratom.GetTotalNumHs()}")
            # print(f"idx:symbol explicit/implicit/total {patom.GetIdx()}:{patom.GetSymbol()} {patom.GetNumExplicitHs()}/{patom.GetNumImplicitHs()}/{patom.GetTotalNumHs()}")
            if q:
                amaph.append(itm)

            rnbrs = self.get_mapped_neighbors(ratom)
            pnbrs = self.get_mapped_neighbors(patom)
            for tpl in pnbrs:
                pbond = product.GetBondBetweenAtoms(*pnbrs[tpl])
                if (pidx, pbond.GetIdx()) in seen:
                    continue
                seen.add((pidx, pbond.GetIdx()))
                if tpl not in rnbrs:
                    # new bond in product
                    res.append(BondInfo(pidx, pnbrs[tpl], pbond.GetIdx(), 'new'))
                else:
                    # present in both reactants and products, check to see if it changed
                    rbond = reactant.GetBondBetweenAtoms(*rnbrs[tpl])
                    if rbond.GetBondType() != pbond.GetBondType():
                        res.append(BondInfo(pidx, pnbrs[tpl], pbond.GetIdx(), 'changed'))

        rxn_atom_info_list = [RxnAtomInfo(**item._asdict()) for item in amap]
        rxn_bond_info_list = [RxnBondInfo(**item._asdict()) for item in res]
        rxn_atomh_info_list = [RxnAtomInfo(**item._asdict()) for item in amaph]
        return rxn_atom_info_list, rxn_atomh_info_list, rxn_bond_info_list,


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

        self.identity_property_name = identity_property_name

    def read_reaction(self, reaction_string: str, inp_fmt: str) -> Tuple[cif.rdChemReactions.ChemicalReaction,
                                                                         utilities.OutcomeMetadata]:
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

    def unpack_rdrxn(self, rdrxn: cif.rdChemReactions.ChemicalReaction):
        constructor = PatternConstructor(identity_property_name=self.identity_property_name)
        reaction_mols = cif.rdrxn_to_molecule_catalog(rdrxn, constructor)
        builder = UnmappedChemicalEquationGenerator()
        attributes = builder.get_basic_attributes(reaction_mols)
        return attributes['catalog'], attributes['stoichiometry_coefficients'], attributes['role_map']

    def build_from_rdrxn(self, rdrxn: cif.rdChemReactions.ChemicalReaction) -> Union[Template, None]:
        reaction_string = cif.rdChemReactions.ReactionToSmarts(rdrxn)
        return self.build_from_reaction_string(reaction_string=reaction_string, inp_fmt='smarts')

    def build_from_reaction_string(self, reaction_string: str, inp_fmt: str) -> Union[Template, None]:
        rdchiral_output = cif.rdchiral_extract_template(reaction_string=reaction_string, inp_fmt=inp_fmt,
                                                        reaction_id=None)
        return self.build_from_rdchiral_output(rdchiral_output=rdchiral_output)

    def create_rdchiral_data(self, rdchiral_output: Dict):
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
        if not rdchiral_output:
            return None
        if not rdchiral_output.get('reaction_smarts'):
            return None
        template = Template()
        rdrxn, rdchiral_data = self.create_rdchiral_data(rdchiral_output)
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

    def read_reaction(self, reaction_string: str, inp_fmt: str) -> Tuple[cif.rdChemReactions.ChemicalReaction,
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

    def unpack_rdrxn(self, rdrxn: cif.rdChemReactions.ChemicalReaction):
        """ To compute ChemicalEquation attributes from the associated rdkit ChemicalReaction object """
        constructor = MoleculeConstructor(molecular_identity_property_name=self.molecular_identity_property_name)
        chemical_equation = create_chemical_equation(
            rdrxn=rdrxn, chemical_equation_identity_name=self.chemical_equation_identity_name,
            constructor=constructor)
        return chemical_equation

    def build_from_rdrxn(self, rdrxn: cif.rdChemReactions.ChemicalReaction) -> ChemicalEquation:
        """ To build a ChemicalEquation instance from a rdkit ChemicalReaction object """
        return self.unpack_rdrxn(rdrxn=rdrxn)

    def build_from_reaction_string(self, reaction_string: str, inp_fmt: str) -> ChemicalEquation:
        """ To build a ChemicalEquation instance from a reaction string """
        rdrxn, read_outcome = self.read_reaction(reaction_string=reaction_string, inp_fmt=inp_fmt)

        return self.build_from_rdrxn(rdrxn=rdrxn)


def create_chemical_equation(rdrxn: cif.rdChemReactions.ChemicalReaction,
                             chemical_equation_identity_name: str,
                             constructor: MoleculeConstructor) -> ChemicalEquation:
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
    builder = Builder()
    builder.set_builder(builder_type)
    return builder.get_chemical_equation(reaction_mols, chemical_equation_identity_name)


class ChemicalEquationGenerator(ABC):
    """ Abstract class for ChemicalEquationAttributesGenerator """

    def get_basic_attributes(self, reaction_mols: dict):
        pass

    @abstractmethod
    def generate_template(self, chemical_equation: ChemicalEquation):
        pass

    @abstractmethod
    def generate_disconnection(self, chemical_equation: ChemicalEquation):
        pass

    @abstractmethod
    def generate_rdrxn(self, catalog: dict,
                       role_map: dict,
                       stoichiometry_coefficients: dict,
                       use_reagents: bool,
                       use_atom_mapping: bool = False,
                       use_smiles: bool = False):
        pass

    @abstractmethod
    def generate_smiles(self, rdrxn: cif.rdChemReactions.ChemicalReaction) -> str:
        pass


class UnmappedChemicalEquationGenerator(ChemicalEquationGenerator):

    def get_basic_attributes(self, reaction_mols: dict) -> dict:
        """ To build the initial attributes of a ChemicalEquation: mapping, role_map, catalog and
        stoichiometry_coefficients. These are returned in a dictionary. """
        basic_attributes = {'mapping': None,
                            'role_map': {role: sorted(list({m.uid for m in set(mols)})) for role, mols in
                                         reaction_mols.items()}}
        all_molecules = reaction_mols['reactants'] + reaction_mols['reagents'] + reaction_mols['products']
        basic_attributes['catalog'] = {m.uid: m for m in set(all_molecules)}
        basic_attributes['stoichiometry_coefficients'] = self.generate_stoichiometry_coefficients(reaction_mols)
        return basic_attributes

    def generate_rdrxn(self, catalog, role_map, stoichiometry_coefficients, use_reagents,
                       use_atom_mapping=False, use_smiles=False) -> cif.rdChemReactions.ChemicalReaction:
        """ To build the rdkit rdrxn object associated with the ChemicalEquation """
        return cif.build_rdrxn(catalog=catalog,
                               role_map=role_map,
                               stoichiometry_coefficients=stoichiometry_coefficients,
                               use_reagents=use_reagents,
                               use_smiles=use_smiles,
                               use_atom_mapping=use_atom_mapping)

    def generate_smiles(self, rdrxn: cif.rdChemReactions.ChemicalReaction) -> str:
        """ To build the smiles associated with the ChemicalEquation """
        return cif.rdrxn_to_string(rdrxn=rdrxn, out_fmt='smiles', use_atom_mapping=False)

    def generate_stoichiometry_coefficients(self, reaction_mols):
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

    def generate_disconnection(self, ce):
        """ To generate the disconnection. It is None if the ChemicalEquation is unmapped. """
        return None


class MappedChemicalEquationGenerator(ChemicalEquationGenerator):

    def get_basic_attributes(self, reaction_mols: dict) -> dict:
        """ To build the initial attributes of a ChemicalEquation: mapping, role_map, catalog and
               stoichiometry_coefficients. These are returned in a dictionary. """
        new_reaction_mols = {'reactants_reagents': reaction_mols['reactants'] + reaction_mols['reagents'],
                             'products': reaction_mols['products']}
        basic_attributes = {'mapping': self.generate_mapping(new_reaction_mols)}
        desired_product = cif.select_desired_product(reaction_mols)
        basic_attributes['role_map'] = cif.role_reassignment(new_reaction_mols, basic_attributes['mapping'],
                                                             desired_product)

        all_molecules = reaction_mols['reactants'] + reaction_mols['reagents'] + reaction_mols['products']
        basic_attributes['catalog'] = {m.uid: m for m in set(all_molecules)}

        basic_attributes['stoichiometry_coefficients'] = self.generate_stoichiometry_coefficients(
            basic_attributes['role_map'], reaction_mols, basic_attributes['mapping'], desired_product)
        return basic_attributes

    def generate_mapping(self, new_reaction_mols: dict) -> Ratam:
        ratam_constructor = RatamConstructor()
        return ratam_constructor.create_ratam(new_reaction_mols)

    def generate_stoichiometry_coefficients(self, role_map: dict,
                                            reaction_mols: dict,
                                            mapping: Ratam,
                                            desired_product: Molecule) -> dict:
        """ To build the dictionary of the stoichiometry coefficients of a mapped ChemicalEquation """
        # the products' stoichiometry coefficients are computed based on the number of repetitions in the products list
        stoichiometry_coefficients = {'products': self.products_stoichiometry(reaction_mols)}
        desired_prod_map_numbers = [n for m in mapping.full_map_info.get(desired_product.uid)
                                    for n in m.values() if n not in [0, -1]]
        all_molecules = reaction_mols['reactants'] + reaction_mols['reagents']
        # the reactants' stoichiometry coefficients are computed based in the number of mapping in which atoms
        # end up in the desired product
        stoichiometry_coefficients = self.reactants_stoichiometry(all_molecules,
                                                                  role_map,
                                                                  stoichiometry_coefficients,
                                                                  mapping,
                                                                  desired_prod_map_numbers)
        # what is left are the reagents and their coefficients can be computed based on the number of repetitions
        reagents_coeffs = {m.uid: all_molecules.count(m) for m in set(all_molecules)}
        stoichiometry_coefficients['reagents'] = reagents_coeffs
        # print(stoichiometry_coefficients)
        return stoichiometry_coefficients

    def products_stoichiometry(self, reaction_mols: dict) -> dict:
        """ To build the stoichiometry coefficients of the products of the ChemicalEquation"""
        products = reaction_mols['products']
        return {m.uid: products.count(m) for m in set(products)}

    def reactants_stoichiometry(self, all_molecules: list,
                                role_map: dict,
                                stoichiometry_coefficients: dict,
                                mapping: Ratam,
                                desired_prod_map_numbers: list) -> dict:
        """ To build the stoichiometry coefficients of the reactants of the ChemicalEquation"""
        reactants_uid = role_map.get('reactants')
        stoichiometry_coefficients['reactants'] = {}
        for uid in reactants_uid:
            coeff = 0
            mapping_list = mapping.full_map_info.get(uid)
            for map in mapping_list:
                if [map_nr for h, map_nr in map.items() if map_nr in desired_prod_map_numbers]:
                    coeff += 1
                    mol = [m for m in all_molecules if m.uid == uid][0]
                    all_molecules.remove(mol)
            stoichiometry_coefficients['reactants'][uid] = coeff
        return stoichiometry_coefficients

    def generate_rdrxn(self, catalog: dict,
                       role_map: dict,
                       stoichiometry_coefficients: dict,
                       use_reagents: bool,
                       use_atom_mapping: bool = True,
                       use_smiles: bool = False) -> cif.rdChemReactions.ChemicalReaction:
        """ To build the rdkit rdrxn object associated with the ChemicalEquation """
        return cif.build_rdrxn(catalog=catalog,
                               role_map=role_map,
                               stoichiometry_coefficients=stoichiometry_coefficients,
                               use_reagents=use_reagents,
                               use_smiles=use_smiles,
                               use_atom_mapping=use_atom_mapping)

    def generate_smiles(self, rdrxn: cif.rdChemReactions.ChemicalReaction) -> str:
        """ To build the smiles associated with the ChemicalEquation """
        return cif.rdrxn_to_string(rdrxn=rdrxn, out_fmt='smiles', use_atom_mapping=True)

    def generate_template(self, ce: ChemicalEquation) -> Template:
        """ To build the template of the ChemicalEquation """
        tc = TemplateConstructor()
        return tc.build_from_reaction_string(reaction_string=ce.smiles, inp_fmt='smiles')

    def generate_disconnection(self, ce: ChemicalEquation) -> Disconnection:
        """ To build the disconnection of the ChemicalEquation """
        dc = DisconnectionConstructor(identity_property_name='smiles')
        rdrxn = ce.build_rdrxn(use_reagents=False, use_atom_mapping=True)
        return dc.build_from_rdrxn(rdrxn=rdrxn)


class Builder:
    builders = {'mapped': MappedChemicalEquationGenerator(),
                'unmapped': UnmappedChemicalEquationGenerator()}
    __builder = None

    def set_builder(self, builder_type: str):
        self.__builder = self.builders[builder_type]

    def get_chemical_equation(self, reaction_mols, chemical_equation_identity_name):
        ce = ChemicalEquation()
        basic_attributes = self.__builder.get_basic_attributes(reaction_mols)
        ce.catalog = basic_attributes['catalog']
        ce.mapping = basic_attributes['mapping']
        ce.role_map = basic_attributes['role_map']
        ce.stoichiometry_coefficients = basic_attributes['stoichiometry_coefficients']
        use_reagents = chemical_equation_identity_name not in ['r_p', 'u_r_p']
        ce.rdrxn = self.__builder.generate_rdrxn(ce.catalog, ce.role_map, ce.stoichiometry_coefficients, use_reagents)
        ce.smiles = self.__builder.generate_smiles(ce.rdrxn)
        ce.hash_map = create_reaction_like_hash_values(ce.catalog, ce.role_map)
        ce.uid = ce.hash_map.get(chemical_equation_identity_name)
        ce.template = self.__builder.generate_template(ce)
        ce.disconnection = self.__builder.generate_disconnection(ce)

        return ce


def calculate_molecular_hash_values(rdmol: cif.Mol, hash_list: Union[set, None] = None) -> dict:
    """ To compute the hash_map dictionary containing molecular properties/representations names and the
        corresponding hash values """
    molhashf = cif.HashFunction.names
    if hash_list is None:
        hash_list = MoleculeConstructor.all_available_identifiers
    hash_map = {}

    if rdkit_hashes := [h for h in hash_list if h in molhashf]:
        hash_map.update({k: cif.MolHash(rdmol, v) for k, v in molhashf.items() if k in rdkit_hashes})
    if 'smiles' in hash_list:
        hash_map.update({'smiles': cif.MolHash(rdmol, v) for k, v in molhashf.items() if k == 'CanonicalSmiles'})

    if other_hashes := [h for h in hash_list if h not in rdkit_hashes]:
        factory = MolIdentifierFactory(rdmol)
        for h in other_hashes:
            if h not in MoleculeConstructor.all_available_identifiers:
                logger.warning(f'{h} is not supported as molecular identifier')
            elif h != 'smiles':
                hash_map.update(factory.select_identifier(h, hash_map))

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
    | separates properties and is followded by the name and a:
    """
    changes_str = '|'.join([f'{k}:{",".join(map(str, v))}' for k, v in changes_map.items()])

    disconnection_summary = '|'.join([idp, changes_str])

    return {'disconnection_summary': disconnection_summary}


def calculate_pattern_hash_values(smarts):
    return {'smarts': smarts}


if __name__ == '__main__':
    print('main')
