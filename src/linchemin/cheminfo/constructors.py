from typing import List, Dict, Tuple, Union
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass

from linchemin.cheminfo.models import (Molecule, ChemicalEquation, Ratam, Disconnection, Template, Pattern)
import linchemin.cheminfo.functions as cif
import linchemin.utilities as utilities

""" 
Module containing the constructor classes of relevant cheminformatics models defined in 'models' module
"""


class BadMapping(ValueError):
    """ To be raised if an atom number is used more than once, indicating that the atom mapping is invalid"""
    pass


logger = utilities.console_logger(__name__)


# Molecule Constructor
class MoleculeConstructor:
    """ Class implementing the constructor of the Molecule class

            Attributes:
                identity_property_name: a string indicating which kind of input string determines the identity
                                        of the object (e.g. 'smiles')

    """

    def __init__(self, identity_property_name: str):
        self.identity_property_name = identity_property_name

    def build_from_molecule_string(self, molecule_string: str, inp_fmt: str, ) -> Molecule:
        """ To build a Molecule instance from a string """
        rdmol_input = cif.rdmol_from_string(input_string=molecule_string, inp_fmt=inp_fmt)
        return self.build_from_rdmol(rdmol_input)

    def build_from_rdmol(self, rdmol: cif.Mol) -> Molecule:
        """ To build a Molecule instance from a rdkit Mol instance """
        rdmol_mapped = rdmol
        rdmol_unmapped = cif.remove_rdmol_atom_mapping(rdmol=rdmol_mapped)
        rdmol_unmapped_canonical = cif.canonicalize_rdmol_lite(rdmol=rdmol_unmapped, is_pattern=False)
        rdmol_mapped_canonical = cif.canonicalize_mapped_rdmol(cif.canonicalize_rdmol_lite(rdmol=rdmol_mapped, is_pattern=False))
        hash_map = calculate_molecular_hash_values(rdmol=rdmol_unmapped_canonical, hash_list=['CanonicalSmiles',
                                                                                              'inchi_key',
                                                                                              'inchi_KET_15T'])
        identity_property = hash_map.get(self.identity_property_name)
        uid = utilities.create_hash(identity_property)
        smiles = cif.compute_mol_smiles(rdmol=rdmol_unmapped_canonical)
        return Molecule(rdmol=rdmol_unmapped_canonical, rdmol_mapped=rdmol_mapped_canonical,
                        identity_property_name=self.identity_property_name, hash_map=hash_map, smiles=smiles,
                        uid=uid, identity_property=identity_property)


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

    def get_full_map_info(self, molecules_catalog: dict):
        full_map_info = {}
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
                                                map_num in prod_map.values()
                                                and map_num not in [0, -1]]:
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

        if len(rxn_reactive_center.rxn_bond_info_list) == 0:
            return None

        product_rdmol = cif.Chem.Mol(rdrxn.GetProductTemplate(desired_product_idx))
        product_changes = rxn_reactive_center.get_product_changes(product_idx=desired_product_idx)

        molecule_constructor = MoleculeConstructor(identity_property_name=self.identity_property_name)
        product_molecule = molecule_constructor.build_from_rdmol(rdmol=product_rdmol)
        disconnection = Disconnection()
        disconnection.molecule = product_molecule
        disconnection.rdmol = product_molecule.rdmol
        reacting_atoms, new_bonds, modified_bonds = self.re_map(product_changes,
                                                                rdmol_old=product_rdmol,
                                                                rdmol_new=product_molecule.rdmol)
        disconnection.reacting_atoms = reacting_atoms
        disconnection.new_bonds = new_bonds
        disconnection.modified_bonds = modified_bonds

        disconnection.hash_map = calculate_disconnection_hash_values(disconnection)
        disconnection.identity_property = disconnection.hash_map.get('disconnection_summary')
        disconnection.uid = utilities.create_hash(disconnection.identity_property)

        """
        rxn_atom_info_list = rxn_reactive_center.rxn_atom_info_list
        rxn_bond_info_list = rxn_reactive_center.rxn_bond_info_list
        print('rxn_atom_info_list:', rxn_atom_info_list)
        print('rxn_bond_info_list:', rxn_bond_info_list)
        print('reacting_atoms:', product_changes.reacting_atoms)
        print('new_bonds:', product_changes.new_bonds)
        print('modified_bonds:', product_changes.modified_bonds)
        """
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
        new_bonds_ = product_changes.new_bonds
        modified_bonds_ = product_changes.modified_bonds

        match = rdmol_new.GetSubstructMatch(rdmol_old)
        reacting_atoms = sorted([match[x] for x in reacting_atoms_])

        new_bonds = sorted(
            [rdmol_new.GetBondBetweenAtoms(match[bond.GetBeginAtomIdx()], match[bond.GetEndAtomIdx()]).GetIdx()
             for bond in rdmol_old.GetBonds() if bond.GetIdx() in new_bonds_])

        modified_bonds = sorted([
            rdmol_new.GetBondBetweenAtoms(match[bond.GetBeginAtomIdx()], match[bond.GetEndAtomIdx()]).GetIdx()
            for bond in rdmol_old.GetBonds() if bond.GetIdx() in modified_bonds_])

        return reacting_atoms, new_bonds, modified_bonds

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
    reacting_atoms: list[int]
    new_bonds: list[int]
    modified_bonds: list[int]


class RXNReactiveCenter:
    """
    Class to identify the atoms and bonds that change in a reaction
    inspired from
    https://greglandrum.github.io/rdkit-blog/tutorial/reactions/2021/11/26/highlighting-changed-bonds-in-reactions.html
    """

    def __init__(self, rdrxn: cif.rdChemReactions.ChemicalReaction):
        rdrxn.Initialize()
        self.rxn_atom_info_list, self.rxn_bond_info_list = self.find_modifications_in_products(rdrxn)

    def get_product_changes(self, product_idx: int = 0):
        reacting_atoms = sorted([ai.product_atom for ai in self.rxn_atom_info_list if ai.product == product_idx])
        new_bonds = sorted(
            [bi.product_bond for bi in self.rxn_bond_info_list if bi.product == product_idx and bi.status == 'new'])
        modified_bonds = sorted(
            [bi.product_bond for bi in self.rxn_bond_info_list if
             bi.product == product_idx and bi.status == 'changed'])
        return RXNProductChanges(reacting_atoms, new_bonds, modified_bonds)

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
        res = {}
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

    def find_modifications_in_products(self, rxn) -> tuple[list[RxnAtomInfo], list[RxnBondInfo]]:
        """ returns a 2-tuple with the modified atoms and bonds from the reaction """
        reactingAtoms = rxn.GetReactingAtoms()
        amap = self.map_reacting_atoms_to_products(rxn, reactingAtoms)
        res = []
        seen = set()
        # this is all driven from the list of reacting atoms:
        for _, ridx, raidx, pidx, paidx in amap:
            reactant = rxn.GetReactantTemplate(ridx)
            ratom = reactant.GetAtomWithIdx(raidx)
            product = rxn.GetProductTemplate(pidx)
            patom = product.GetAtomWithIdx(paidx)

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
        return rxn_atom_info_list, rxn_bond_info_list


# Pattern Constructor
class PatternConstructor:
    """ Class implementing the constructor of the Pattern class """

    def __init__(self, identity_property_name: str = 'smarts'):
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
    def __init__(self, identity_property_name: str = 'smarts'):

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
        template.uid = template.hash_map.get('r_p')  # TODO: review

        template.rdrxn = cif.build_rdrxn(catalog=pattern_catalog,
                                         role_map=role_map,
                                         stoichiometry_coefficients=stoichiometry_coefficients,
                                         use_smiles=False,
                                         use_atom_mapping=True)
        template.smarts = cif.rdrxn_to_string(rdrxn=template.rdrxn, out_fmt='smarts')

        return template


# ChemicalEquation Constructor
class ChemicalEquationConstructor:
    """ Class implementing the constructor of the ChemicalEquation class

        Attributes:
            identity_property_name: a string indicating which kind of input string determines the identity
                                    of the object (e.g. 'smiles')

    """

    def __init__(self, identity_property_name: str):

        self.identity_property_name = identity_property_name

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
        constructor = MoleculeConstructor(identity_property_name=self.identity_property_name)
        chemical_equation = create_chemical_equation(
            rdrxn=rdrxn, identity_property_name=self.identity_property_name,
            constructor=constructor)
        return chemical_equation

    def build_from_rdrxn(self, rdrxn: cif.rdChemReactions.ChemicalReaction) -> ChemicalEquation:
        """ To build a ChemicalEquation instance from a rdkit ChemicalReaction object """
        chemical_equation = self.unpack_rdrxn(rdrxn=rdrxn)
        return chemical_equation

    def build_from_reaction_string(self, reaction_string: str, inp_fmt: str) -> ChemicalEquation:
        """ To build a ChemicalEquation instance from a reaction string """
        rdrxn, read_outcome = self.read_reaction(reaction_string=reaction_string, inp_fmt=inp_fmt)

        return self.build_from_rdrxn(rdrxn=rdrxn)


def create_chemical_equation(rdrxn: cif.rdChemReactions.ChemicalReaction, identity_property_name, constructor):
    """ Initializes the correct builder of the ChemicalEquation based on the presence of the atom mapping.

        Parameters:
            rdrxn: the initial RDKit ChemReaction object

            identity_property_name:

            constructor: the constructor for the Molecule objects involved

        Returns:
            a new ChemicalEquation instance
    """
    builder_type = 'mapped' if cif.has_mapped_products(rdrxn) else 'unmapped'
    reaction_mols = cif.rdrxn_to_molecule_catalog(rdrxn, constructor)
    builder = Builder()
    builder.set_builder(builder_type)
    return builder.get_chemical_equation(reaction_mols)


class ChemicalEquationGenerator(ABC):
    """ Abstract class for ChemicalEquationAttributesGenerator """

    def get_basic_attributes(self, reaction_mols: dict):
        pass

    @abstractmethod
    def generate_template(self, *args):
        pass

    @abstractmethod
    def generate_disconnection(self, *args):
        pass

    @abstractmethod
    def generate_rdrxn(self, catalog, role_map, stoichiometry_coefficients, use_atom_mapping=False, use_smiles=False):
        pass

    @abstractmethod
    def generate_smiles(self, rdrxn: cif.rdChemReactions.ChemicalReaction):
        pass

    def generate_stoichiometry_coefficients(self, role_map, all_molecules):
        stoichiometry_coefficients = {}
        for role, mol_uid_list in role_map.items():
            mols = [m for m in all_molecules if m.uid in mol_uid_list]
            stoichiometry_coefficients_tmp = {m.uid: mols.count(m) for m in set(mols) if m.uid in mol_uid_list}
            stoichiometry_coefficients[role] = stoichiometry_coefficients_tmp

        return stoichiometry_coefficients


class UnmappedChemicalEquationGenerator(ChemicalEquationGenerator):

    def get_basic_attributes(self, reaction_mols: dict):
        basic_attributes = {'mapping': None,
                            'role_map': {role: sorted([m.uid for m in set(mols)]) for role, mols in
                                         reaction_mols.items()}}
        all_molecules = reaction_mols['reactants'] + reaction_mols['reagents'] + reaction_mols['products']
        basic_attributes['catalog'] = {m.uid: m for m in set(all_molecules)}
        basic_attributes['stoichiometry_coefficients'] = self.generate_stoichiometry_coefficients(
            basic_attributes['role_map'], all_molecules)
        return basic_attributes

    def generate_rdrxn(self, catalog, role_map, stoichiometry_coefficients, use_atom_mapping=False, use_smiles=False):
        return cif.build_rdrxn(catalog=catalog,
                               role_map=role_map,
                               stoichiometry_coefficients=stoichiometry_coefficients,
                               use_smiles=use_smiles,
                               use_atom_mapping=use_atom_mapping)

    def generate_smiles(self, rdrxn: cif.rdChemReactions.ChemicalReaction):
        return cif.rdrxn_to_string(rdrxn=rdrxn, out_fmt='smiles', use_atom_mapping=False)

    def generate_template(self, ce):
        return None

    def generate_disconnection(self, ce):
        return None


class MappedChemicalEquationGenerator(ChemicalEquationGenerator):

    def get_basic_attributes(self, reaction_mols: dict):
        new_reaction_mols = {'reactants_reagents': reaction_mols['reactants'] + reaction_mols['reagents'],
                             'products': reaction_mols['products']}
        basic_attributes = {'mapping': self.generate_mapping(new_reaction_mols)}
        desired_product = cif.select_desired_product(reaction_mols)
        basic_attributes['role_map'] = cif.role_reassignment(new_reaction_mols, basic_attributes['mapping'],
                                                             desired_product)

        all_molecules = reaction_mols['reactants'] + reaction_mols['reagents'] + reaction_mols['products']
        basic_attributes['catalog'] = {m.uid: m for m in set(all_molecules)}

        basic_attributes['stoichiometry_coefficients'] = self.generate_stoichiometry_coefficients(
            basic_attributes['role_map'], all_molecules)
        return basic_attributes

    def generate_mapping(self, new_reaction_mols: dict):
        ratam_constructor = RatamConstructor()
        return ratam_constructor.create_ratam(new_reaction_mols)

    def generate_rdrxn(self, catalog, role_map, stoichiometry_coefficients, use_atom_mapping=True, use_smiles=False):
        return cif.build_rdrxn(catalog=catalog,
                               role_map=role_map,
                               stoichiometry_coefficients=stoichiometry_coefficients,
                               use_smiles=use_smiles,
                               use_atom_mapping=use_atom_mapping)

    def generate_smiles(self, rdrxn: cif.rdChemReactions.ChemicalReaction):
        return cif.rdrxn_to_string(rdrxn=rdrxn, out_fmt='smiles', use_atom_mapping=True)

    def generate_template(self, ce):
        tc = TemplateConstructor()
        return tc.build_from_reaction_string(reaction_string=ce.smiles, inp_fmt='smiles')

    def generate_disconnection(self, ce):
        dc = DisconnectionConstructor(identity_property_name='smiles')
        rdrxn = ce.build_rdrxn(use_atom_mapping=True)
        return dc.build_from_rdrxn(rdrxn=rdrxn)


class Builder:
    builders = {'mapped': MappedChemicalEquationGenerator(),
                'unmapped': UnmappedChemicalEquationGenerator()}
    __builder = None

    def set_builder(self, builder_type: str):
        self.__builder = self.builders[builder_type]

    def get_chemical_equation(self, reaction_mols):
        ce = ChemicalEquation()
        basic_attributes = self.__builder.get_basic_attributes(reaction_mols)
        ce.catalog = basic_attributes['catalog']
        ce.mapping = basic_attributes['mapping']
        ce.role_map = basic_attributes['role_map']
        ce.stoichiometry_coefficients = basic_attributes['stoichiometry_coefficients']
        ce.rdrxn = self.__builder.generate_rdrxn(ce.catalog, ce.role_map, ce.stoichiometry_coefficients)
        ce.smiles = self.__builder.generate_smiles(ce.rdrxn)
        ce.hash_map = create_reaction_like_hash_values(ce.catalog, ce.role_map)
        ce.uid = ce.hash_map.get('r_r_p')  # TODO: review
        ce.template = self.__builder.generate_template(ce)
        ce.disconnection = self.__builder.generate_disconnection(ce)

        return ce


def calculate_molecular_hash_values(rdmol: cif.Mol, hash_list: List[str] = None) -> dict:
    molhashf = cif.HashFunction.names
    if hash_list:
        hash_list += ['CanonicalSmiles']
    else:
        hash_list = list(molhashf.keys()) + ['inchi_key', 'inchikey_KET_15T', 'noiso_smiles', 'cx_smiles']

    hash_map = {k: cif.MolHash(rdmol, v) for k, v in molhashf.items() if k in hash_list}

    hash_map['smiles'] = hash_map.get('CanonicalSmiles')
    if 'inchi_key' in hash_list:
        hash_map['inchi'] = cif.Chem.MolToInchi(rdmol)
        hash_map['inchi_key'] = cif.Chem.InchiToInchiKey(hash_map['inchi'])
    if 'inchikey_KET_15T' in hash_list:
        hash_map['inchi_KET_15T'] = cif.Chem.MolToInchi(rdmol, options='-KET -15T')
        hash_map['inchikey_KET_15T'] = cif.Chem.InchiToInchiKey(hash_map['inchi_KET_15T'])
    if 'noiso_smiles' in hash_list:
        hash_map['noiso_smiles'] = cif.Chem.MolToSmiles(rdmol, isomericSmiles=False)
    if 'cx_smiles' in hash_list:
        hash_map['cx_smiles'] = cif.Chem.MolToCXSmiles(rdmol)
    """

    hash_map['ExtendedMurcko_AG'] = smiles_to_anonymus_graph(hash_map['ExtendedMurcko'])
    hash_map['ExtendedMurcko_EG'] = smiles_to_element_graph(hash_map['ExtendedMurcko'])
    hash_map['MurckoScaffold_AG'] = smiles_to_anonymus_graph(hash_map['MurckoScaffold'])
    hash_map['MurckoScaffold_EG'] = smiles_to_element_graph(hash_map['MurckoScaffold'])
    """

    return hash_map


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
    changes = '__'.join(['_'.join(map(str, disconnection.reacting_atoms)), '_'.join(map(str, disconnection.new_bonds)),
                         '_'.join(map(str, disconnection.modified_bonds)), ])
    disconnection_summary = '|'.join([idp, changes])
    return {'disconnection_summary': disconnection_summary}


def calculate_pattern_hash_values(smarts):
    return {'smarts': smarts}


if __name__ == '__main__':
    print('main')
