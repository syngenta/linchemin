import copy
from collections import namedtuple
from typing import Callable, List, Dict
from functools import partial
from abc import ABC, abstractmethod
import re
import rdkit
from rdkit import Chem
from rdkit.Chem import rdchem, rdChemReactions, DataStructs, rdFingerprintGenerator, Draw
from rdkit.Chem.rdchem import Mol, Atom
from rdkit.Chem.Draw import DrawingOptions, rdMolDraw2D
from rdkit.Chem.rdMolHash import HashFunction, MolHash
from rdkit import RDLogger
import linchemin.utilities as utilities
from rdchiral import template_extractor

# RDLogger.DisableLog('rdApp.*')
logger = utilities.console_logger(__name__)


# RDMOLECULE
def rdmol_from_string(input_string: str, inp_fmt: str):
    function_map = {'smiles': Chem.MolFromSmiles,
                    'smarts': Chem.MolFromSmarts}
    func = function_map.get(inp_fmt)
    return func(input_string)


def remove_rdmol_atom_mapping(rdmol: Mol) -> Mol:
    rdmol_unmapped = copy.deepcopy(rdmol)
    [a.SetAtomMapNum(0) for a in rdmol_unmapped.GetAtoms()]
    return rdmol_unmapped


def detect_rdmol_problems(rdmol):
    task_name = 'detect_problems'
    if rdmol:
        problems = Chem.DetectChemistryProblems(rdmol)
        problem_log = [{'type': problem.GetType(), 'message': problem.Message()} for problem in problems]
        is_successful = True
    else:
        problem_log = []
        is_successful = False
    log = {'is_successful': is_successful, 'task_name': task_name, 'message': problem_log}
    return rdmol, log


def sanitize_rdmol(rdmol):
    task_name = 'sanitize'
    if rdmol:
        try:
            sanit_fail = Chem.SanitizeMol(rdmol, catchErrors=False,
                                          sanitizeOps=rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_ALL)
            message = sanit_fail
            is_successful = True
        except Exception as e:
            exception = e
            rdmol = None
            is_successful = False
            message = exception
    else:
        message = ''
        is_successful = False
    log = {'is_successful': is_successful, 'task_name': task_name, 'message': message}
    return rdmol, log


def get_canonical_order(rdmol: rdkit.Chem.rdchem.Mol) -> tuple:
    """
        https://gist.github.com/ptosco/36574d7f025a932bc1b8db221903a8d2

        :param rdmol:
        :return:
        """
    canon_idx_old_idx = [(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(rdmol))]
    old_idcs_sorted_by_canon_idcs = tuple(zip(*sorted(canon_idx_old_idx)))
    canonical_order = old_idcs_sorted_by_canon_idcs[1]
    return canonical_order


def canonicalize_rdmol(rdmol: rdkit.Chem.rdchem.Mol) -> rdkit.Chem.rdchem.Mol:
    """
        Atoms in a molecule are uniquely aligned in an arbitrary way.
        Each cheminformatics software has such a standardization process, called canonicalization, in it.
        According to a RDKit discussion, RDKit aligns atoms in a molecule during graph traversal.
        The atom order is not the same one in the canonical SMILES’s atom order.
        We must pay attention to this, if the atom order impacts the calculation results.
        The RDKit algorithm is described here:
            https://pubs.acs.org/doi/10.1021/acs.jcim.5b00543
        The key function is found here:
            https://www.rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html#rdkit.Chem.rdmolfiles.CanonicalRankAtoms
        A usable recipe is reported here
            https://www.rdkit.org/docs/Cookbook.html#reorder-atoms
            https://gist.github.com/ptosco/36574d7f025a932bc1b8db221903a8d2
        Some furhter discussion here below
            https://sourceforge.net/p/rdkit/mailman/message/37085522/
            https://github.com/rdkit/rdkit/issues/2637
            https://github.com/rdkit/rdkit/issues/2006
            https://sourceforge.net/p/rdkit/mailman/message/34923617/

        :param rdmol:
        :return: rdmol_canonicalized
        """
    task_name = 'canonicalize_rdmol'
    if rdmol:
        try:
            new_order = get_canonical_order(rdmol=rdmol)
            rdmol = Chem.RenumberAtoms(rdmol, list(new_order))
            is_successful = True
            message = ''
        except Exception as e:
            exception = e
            rdmol = None
            is_successful = False
            message = exception
    else:
        message = ''
        is_successful = False
    log = {'is_successful': is_successful, 'task_name': task_name, 'message': message}

    return rdmol, log


def canonicalize_rdmol_lite(rdmol: rdkit.Chem.rdchem.Mol, is_pattern: bool = False) -> rdkit.Chem.rdchem.Mol:
    if is_pattern:
        return Chem.MolFromSmarts(Chem.MolToSmarts(rdmol))
    else:
        return Chem.MolFromSmiles(Chem.MolToSmiles(rdmol))


def rdmol_to_bstr(rdmol: rdkit.Chem.rdchem.Mol):
    return rdmol.ToBinary()


def bstr_to_rdmol(rdmol_bstr: str) -> rdkit.Chem.rdchem.Mol:
    return rdkit.Chem.rdchem.Mol(rdmol_bstr)


MolPropertyFunction = Callable[[Mol], str]


def compute_mol_smiles(rdmol: Mol, isomeric_smiles: bool = True) -> str:
    return Chem.MolToSmiles(rdmol, isomericSmiles=isomeric_smiles)


def compute_mol_smarts(rdmol: Mol, isomeric_smiles: bool = True) -> str:
    return Chem.MolToSmarts(rdmol, isomericSmiles=isomeric_smiles)


def compute_mol_inchi_key(rdmol: Mol) -> str:
    inchi = Chem.MolToInchi(rdmol, options='-KET -15T')
    return Chem.InchiToInchiKey(inchi)


def get_mol_property_function(property_name: str) -> MolPropertyFunction:
    mol_property_function_factory = {'smiles': compute_mol_smiles,
                                     'inchi_key': compute_mol_inchi_key}
    return mol_property_function_factory.get(property_name)


def draw_mol(smiles: str, filename: str):
    """
        Produces an file with the picture of a molecule using RDKit

        Parameters:
            smiles: string indicating the smiles of the molecule
            filename: string indicated the name of the output file

        Returns:
            None. The image file is produced and stored in the directory
        """
    rdmol = rdmol_from_string(smiles, inp_fmt='smiles')

    DrawingOptions.atomLabelFontSize = 55
    DrawingOptions.dotsPerAngstrom = 100
    DrawingOptions.bondLineWidth = 3.0
    Draw.MolToFile(rdmol, filename)


####################################################################################################
# utilities functions for creating fingerprints

def generate_rdkit_fp(params):
    return rdFingerprintGenerator.GetRDKitFPGenerator(
        minPath=params.get('minPath', 1),
        maxPath=params.get('maxPath', 7),
        countSimulation=params.get('countSimulation', False),
        numBitsPerFeature=params.get('numBitsPerFeature', 2),
        fpSize=params.get('fpSize', 2048))


def generate_morgan_fp(params):
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=params.get('radius', 3),
        useCountSimulation=params.get('countSimulation', False),
        includeChirality=params.get('includeChirality', False),
        useBondTypes=params.get('useBondTypes', True),
        countBounds=params.get('countBounds', None),
        fpSize=params.get('fpSize', 2048))


def generate_topological_fp(params):
    return rdFingerprintGenerator.GetTopologicalTorsionGenerator(
        includeChirality=params.get('includeChirality', False),
        torsionAtomCount=params.get('torsionAtomCount', 4),
        countSimulation=params.get('countSimulation', True),
        countBounds=params.get('countBounds', None),
        fpSize=params.get('fpSize', 2048))


# RDRXN
def rdrxn_from_string(input_string: str, inp_fmt: str) -> rdChemReactions.ChemicalReaction:
    format_function_map = {'smiles': partial(Chem.rdChemReactions.ReactionFromSmarts, useSmiles=True),
                           'smarts': partial(Chem.rdChemReactions.ReactionFromSmarts, useSmiles=False),
                           'rxn': Chem.rdChemReactions.ReactionFromRxnBlock}

    func = format_function_map.get(inp_fmt)
    return func(input_string)


def rdrxn_to_string(rdrxn: rdChemReactions.ChemicalReaction, out_fmt: str, use_atom_mapping: bool = False) -> str:
    if not use_atom_mapping:
        Chem.rdChemReactions.RemoveMappingNumbersFromReactions(rdrxn)
    function_map = {'smiles': partial(Chem.rdChemReactions.ReactionToSmiles, canonical=True),
                    'smarts': Chem.rdChemReactions.ReactionToSmarts,
                    # 'rxn': partial(Chem.rdChemReactions.ReactionToRxnBlock, forceV3000=True),
                    'rxn': Chem.rdChemReactions.ReactionToRxnBlock,
                    # 'rxn': Chem.rdChemReactions.ReactionToV3KRxnBlock,
                    }
    func = function_map.get(out_fmt)
    return func(rdrxn)


def rdrxn_to_rxn_mol_catalog(rdrxn: rdChemReactions.ChemicalReaction) -> Dict[str, List[Mol]]:
    return {'reactants': list(rdrxn.GetReactants()),
            'reagents': list(rdrxn.GetAgents()),
            'products': list(rdrxn.GetProducts())}


def rdrxn_from_rxn_mol_catalog(rxn_mol_catalog: Dict[str, List[Mol]]) -> rdChemReactions.ChemicalReaction:
    reaction_smiles_empty = '>>'
    rdrxn = rdChemReactions.ReactionFromSmarts(reaction_smiles_empty, useSmiles=True)
    for rdmol in rxn_mol_catalog.get('reactants'):
        rdrxn.AddReactantTemplate(rdmol)
    for rdmol in rxn_mol_catalog.get('reagents'):
        rdrxn.AddAgentTemplate(rdmol)
    for rdmol in rxn_mol_catalog.get('products'):
        rdrxn.AddProductTemplate(rdmol)
    rdChemReactions.SanitizeRxn(rdrxn)
    return rdrxn


def rdmol_catalog_to_molecule_catalog(rdmol_catalog: Dict[str, List[Mol]], constructor):
    mol_catalog = {}
    for role, rdmols in rdmol_catalog.items():
        mol_catalog[role] = [constructor.build_from_rdmol(rdmol=rdmol) for r, rdmol_list in rdmol_catalog.items()
                             if r == role for rdmol in rdmol_list]
    return mol_catalog


def is_mapped(reaction_rdmols: dict) -> bool:
    """ To check if a rdrxn is mapped by looking at the mapping numbers of the products """
    atoms = [mol.GetAtoms() for mol in reaction_rdmols.get('products')]
    mapped_atoms = [a.GetAtomMapNum() for a_list in atoms for a in a_list]
    if not mapped_atoms or all(mapped_atoms) == 0:
        return False
    else:
        return True


def select_desired_product(mol_catalog: dict):
    """ To select the 'desired product' among the products of a reaction """
    return mol_catalog['products'][0]


def new_role_reassignment(reaction_mols: dict, cem, desired_product):
    """ To reassign the roles of reactants and reagents based on the mapping on the desired product """
    if desired_product not in reaction_mols['products']:
        logger.error('The selected product is not among the reaction products.')
        return None
    desired_product_transformations = [at for at in cem.atom_transformations if at.product_uid == desired_product.uid]
    true_reactants_uid = {at.reactant_uid for at in desired_product_transformations}
    true_reagents = {r.uid for r in reaction_mols['reactants_reagents'] if r.uid not in true_reactants_uid}
    true_reactants = {r.uid for r in reaction_mols['reactants_reagents'] if r.uid in true_reactants_uid}
    products = [m.uid for m in reaction_mols['products']]
    reagents = check_reagents(cem.full_map_info, true_reagents)
    return {'reactants': sorted(list(true_reactants)),
            'reagents': sorted(list(reagents)),
            'products': sorted(products)}


def check_reagents(full_map_info, reagents):
    """ To check if some molecules appear both mapped and unmapped and put the unmapped ones among the reagents"""
    for uid, map_list in full_map_info.items():
        for d in map_list:
            map_nums = list(d.values())
            if not map_nums or all(map_nums) == 0 or all(map_nums) == -1:
                reagents.add(uid)
    return reagents


def rdchiral_extract_template(reaction_string: str, inp_fmt: str, reaction_id: int = None):
    if inp_fmt != 'smiles':
        raise NotImplementedError
    mapped_smiles_split = reaction_string.split('>')
    rdchiral_input = {'_id': reaction_id,
                      'reactants': mapped_smiles_split[0],
                      'agents': mapped_smiles_split[1],
                      'products': mapped_smiles_split[2]}
    return template_extractor.extract_from_reaction(reaction=rdchiral_input)


# def unpack_rdrxn(rdrxn: rdChemReactions.ChemicalReaction, identity_property_name, constructor):
#     """ Initializes the correct builder of the ChemicalEquation attributes based on the presence of the atom mapping """
#     reaction_rdmols = rdrxn_to_rxn_mol_catalog(rdrxn=rdrxn)
#     builder_type = 'mapped' if is_mapped(reaction_rdmols) else 'unmapped'
#     reaction_mols = rdmol_catalog_to_molecule_catalog(reaction_rdmols, constructor)
#     builder = AttributeBuilder()
#     builder.set_builder(builder_type)
#     return builder.get_attributes(reaction_mols)
#
#
# class ChemicalEquationAttributes:
#     def __init__(self):
#         self.catalog = None
#         self.role_map = None
#         self.stoichiometry_coefficients = None
#         self.mapping = None
#         self.disconnection = None
#         self.template = None
#
#     def set_catalog(self, catalog):
#         self.catalog = catalog
#
#     def set_role_map(self, role_map):
#         self.role_map = role_map
#
#     def set_stoichiometry_coefficients(self, stoichiometry_coefficients):
#         self.stoichiometry_coefficients = stoichiometry_coefficients
#
#     def set_mapping(self, mapping):
#         self.mapping = mapping
#
#     def set_disconnection(self, disconnection):
#         self.disconnection = disconnection
#
#     def set_template(self, template):
#         self.template = template
#
#
# class ChemicalEquationAttributesGenerator(ABC):
#     """ Abstract class for ChemicalEquationAttributesGenerator """
#
#     def get_basic_attributes(self, reaction_mols: dict):
#         pass
#
#     @abstractmethod
#     def generate_template(self, *args):
#         pass
#
#     @abstractmethod
#     def generate_disconnection(self, *args):
#         pass
#
#
# class UnmappedChemicalEquationAttributesGenerator(ChemicalEquationAttributesGenerator):
#
#     def get_basic_attributes(self, reaction_mols: dict):
#         mapping = None
#         role_map = {role: sorted([m.uid for m in set(mols)]) for role, mols in reaction_mols.items()}
#         all_molecules = reaction_mols['reactants'] + reaction_mols['reagents'] + reaction_mols['products']
#         catalog = {m.uid: m for m in set(all_molecules)}
#         stoichiometry_coefficients = {}
#
#         for role, mol_uid_list in role_map.items():
#             mols = [m for m in all_molecules if m.uid in mol_uid_list]
#             stoichiometry_coefficients_tmp = {m.uid: mols.count(m) for m in set(mols) if m.uid in mol_uid_list}
#             stoichiometry_coefficients[role] = stoichiometry_coefficients_tmp
#
#         return catalog, role_map, stoichiometry_coefficients, mapping
#
#     def generate_template(self):
#         return None
#
#     def generate_disconnection(self):
#         return None
#
#
# class MappedChemicalEquationAttributesGenerator(ChemicalEquationAttributesGenerator):
#
#     def get_basic_attributes(self, reaction_mols: dict):
#         new_reaction_mols = {'reactants_reagents': reaction_mols['reactants'] + reaction_mols['reagents'],
#                              'products': reaction_mols['products']}
#         mapping = self.generate_mapping(new_reaction_mols)
#
#         desired_product = select_desired_product(reaction_mols)
#         role_map = new_role_reassignment(new_reaction_mols, mapping, desired_product)
#
#         all_molecules = reaction_mols['reactants'] + reaction_mols['reagents'] + reaction_mols['products']
#         catalog = {m.uid: m for m in set(all_molecules)}
#         stoichiometry_coefficients = {}
#
#         for role, mol_uid_list in role_map.items():
#             mols = [m for m in all_molecules if m.uid in mol_uid_list]
#             stoichiometry_coefficients_tmp = {m.uid: mols.count(m) for m in set(mols) if m.uid in mol_uid_list}
#             stoichiometry_coefficients[role] = stoichiometry_coefficients_tmp
#
#         return catalog, role_map, stoichiometry_coefficients, mapping
#
#     def generate_mapping(self, new_reaction_mols: dict):
#         return ChemicalEquationMapping(new_reaction_mols)
#
#     def generate_template(self):
#         return None
#
#     def generate_disconnection(self):
#         return None
#
#
# class AttributeBuilder:
#     builders = {'mapped': MappedChemicalEquationAttributesGenerator(),
#                 'unmapped': UnmappedChemicalEquationAttributesGenerator()}
#     __builder = None
#
#     def set_builder(self, builder_type: str):
#         self.__builder = self.builders[builder_type]
#
#     def get_attributes(self, reaction_mols):
#         attributes = ChemicalEquationAttributes()
#
#         catalog, role_map, stoichiometry_coefficients, mapping = self.__builder.get_basic_attributes(reaction_mols)
#         attributes.set_catalog(catalog)
#         attributes.set_role_map(role_map)
#         attributes.set_stoichiometry_coefficients(stoichiometry_coefficients)
#         attributes.set_mapping(mapping)
#
#         template = self.__builder.generate_template()
#         attributes.set_template(template)
#
#         disconnection = self.__builder.generate_disconnection()
#         attributes.set_disconnection(disconnection)
#
#         return attributes


"""
*******Old chemical equation builder****************
def unpack_unmapped_rdrxn(reaction_rdmols, constructor):
    catalog = {}
    role_map = {}
    stoichiometry_coefficients = {}
    for role, rdmol_list in reaction_rdmols.items():
        list_tmp = [constructor.build_from_rdmol(rdmol=rdmol) for rdmol in rdmol_list]
        set_tmp = set(list_tmp)
        stoichiometry_coefficients_tmp = {m.uid: list_tmp.count(m) for m in set_tmp}
        stoichiometry_coefficients[role] = stoichiometry_coefficients_tmp
        _tmp = {m.uid: m for m in set_tmp}
        # the sorting provides the arbitrary canonicalization on the ordering  of molecules for each role
        role_map[role] = sorted(list(stoichiometry_coefficients_tmp.keys()))
        catalog = {**catalog, **_tmp}
    return catalog, role_map, stoichiometry_coefficients, None


def unpack_mapped_rdrxn(reaction_rdmols, constructor):
    reactants_reagents = [constructor.build_from_rdmol(rdmol=rdmol) for role, rdmol_list in reaction_rdmols.items()
                          if role in ['reactants', 'reagents'] for rdmol in rdmol_list]
    products = [constructor.build_from_rdmol(rdmol=rdmol) for role, rdmol_list in reaction_rdmols.items()
                if role == 'products' for rdmol in rdmol_list]
    reaction_mols = {'reactants_reagents': reactants_reagents, 'products': products}
    chemical_equation_mapping = ChemicalEquationMapping(reaction_mols)
    new_role_map = new_role_reassignment(reaction_mols, chemical_equation_mapping, products[0])
    stoichiometry_coefficients = {}
    catalog = {}
    for role, mol_uid_list in new_role_map.items():
        mols = [m for m in reactants_reagents + products if m.uid in mol_uid_list]
        stoichiometry_coefficients_tmp = {m.uid: mols.count(m) for m in set(mols) if m.uid in mol_uid_list}
        stoichiometry_coefficients[role] = stoichiometry_coefficients_tmp
        _tmp = {m.uid: m for m in set(mols)}
        catalog = {**catalog, **_tmp}
    return catalog, new_role_map, stoichiometry_coefficients, chemical_equation_mapping
"""


def build_rdrxn(catalog: Dict,
                role_map: Dict,
                stoichiometry_coefficients: Dict,
                use_smiles: bool,
                use_atom_mapping: bool) -> rdChemReactions.ChemicalReaction:
    reaction_smiles_empty = '>>'
    rdrxn_new = rdChemReactions.ReactionFromSmarts(reaction_smiles_empty, useSmiles=use_smiles)
    for mol_id in role_map.get('reactants'):
        if rdmol_mapped := catalog.get(mol_id).rdmol_mapped:
            for _ in range(stoichiometry_coefficients.get('reactants').get(mol_id)):
                rdrxn_new.AddReactantTemplate(rdmol_mapped)
    for mol_id in role_map.get('reagents'):
        if rdmol_mapped := catalog.get(mol_id).rdmol_mapped:
            for _ in range(stoichiometry_coefficients.get('reagents').get(mol_id)):
                rdrxn_new.AddAgentTemplate(rdmol_mapped)
    for mol_id in role_map.get('products'):
        if rdmol_mapped := catalog.get(mol_id).rdmol_mapped:
            for _ in range(stoichiometry_coefficients.get('products').get(mol_id)):
                rdrxn_new.AddProductTemplate(rdmol_mapped)
    if not use_atom_mapping:
        Chem.rdChemReactions.RemoveMappingNumbersFromReactions(rdrxn_new)
    return rdrxn_new


def canonicalize_rdrxn(rdrxn: rdChemReactions.ChemicalReaction) -> rdChemReactions.ChemicalReaction:
    # canonicalize a reaction passing through a smiles
    # canonicalize the order of atoms in each molecule
    # canonicalize the order of molecules in the reaction
    rdrxn = rdChemReactions.ReactionFromSmarts(rdChemReactions.ReactionToSmiles(rdrxn, canonical=True), useSmiles=True)
    return rdrxn


def activate_rdrxn(rdrxn: rdChemReactions.ChemicalReaction) -> (rdChemReactions.ChemicalReaction, dict):
    # TODO: the whole error handling logic can be better
    # http://www.rdkit.org/Python_Docs/rdkit.Chem.SimpleEnum.Enumerator-module.html#PreprocessReaction
    # http://www.rdkit.org/Python_Docs/rdkit.Chem.rdChemReactions-module.html

    try:

        rdChemReactions.SanitizeRxn(rdrxn)
        rdrxn.Initialize()
        nWarn, nError, nReacts, nProds, reactantLabels = rdChemReactions.PreprocessReaction(rdrxn)
        rdrxn_log = {'n_warn': nWarn, 'n_error': nError, 'n_reactants': nReacts, 'n_products': nProds,
                     'reactant_labels': reactantLabels}
        is_successful = True
        if nError:
            rdrxn = None
            is_successful = False
        exception = ''

    except Exception as e:
        exception = e
        rdrxn = None
        rdrxn_log = {}
        is_successful = False

    log = {'is_successful': is_successful, 'exception': exception, 'rdrxn_log': rdrxn_log}
    return rdrxn, log


def rdrxn_role_reassignment(rdrxn: rdChemReactions.ChemicalReaction,
                            desired_product_idx: int = 0) -> rdChemReactions.ChemicalReaction:
    """function to reassign the reactant/reagent role in a ChemicalReaction by identifying
    1) reactant molecules that should be reagent
    2) reagent molecules that should be reactants
    and reconstructing a ChemicalReaction with correct roles
    this role reassignment is requires reaction atom to atom to atom mapping to identify reactants/reagents

    a reactant contributes with at least one heavy atom to the desired product of a reaction
    a reagent does not contribute with any heavy atom to the desired product of a reaction
    either reactant or reagents can contribute to by-products

        Parameters:
        rdrxn: RDKit ChemicalReaction
        desired_product_idx: integer index of the reference product used to assess the reactant contribution

    Returns:
        rdrxn: RDKit ChemicalReaction

    """
    if not rdChemReactions.HasReactionAtomMapping(rdrxn):
        return rdrxn
    molecule_catalog = {'reactants': [{'rdmol': rdmol, 'role_molecule_idx': i,
                                       'matam': {atom.GetIdx(): atom.GetAtomMapNum() for atom in rdmol.GetAtoms()}} for
                                      i, rdmol in enumerate(rdrxn.GetReactants())], 'reagents': [
        {'rdmol': rdmol, 'role_molecule_idx': i,
         'matam': {atom.GetIdx(): atom.GetAtomMapNum() for atom in rdmol.GetAtoms()}} for i, rdmol in
        enumerate(rdrxn.GetAgents())], 'products': [{'rdmol': rdmol, 'role_molecule_idx': i,
                                                     'matam': {atom.GetIdx(): atom.GetAtomMapNum() for atom in
                                                               rdmol.GetAtoms()}} for i, rdmol in
                                                    enumerate(rdrxn.GetProducts())]}

    rxn_mol_catalog = {'reactants': [], 'reagents': [],
                       'products': [item.get('rdmol') for item in molecule_catalog.get('products')]}

    keys_output = set(molecule_catalog.get('products')[desired_product_idx].get('matam').values())

    for input_item in molecule_catalog.get('reactants') + molecule_catalog.get('reagents'):
        keys_input = set(input_item.get('matam').values())
        x = keys_input.intersection(keys_output) - {0}
        rdmol = input_item.get('rdmol')
        if len(x) == 0:
            rxn_mol_catalog.get('reagents').append(rdmol)
        else:
            rxn_mol_catalog.get('reactants').append(rdmol)
    return rdrxn_from_rxn_mol_catalog(rxn_mol_catalog=rxn_mol_catalog)


def draw_reaction(smiles: str, filename: str):
    """
    Produces a file with the picture of a reaction using RDKit

    Parameters:
        smiles: string indicating the smiles of the reaction
        filename: string indicated the name of the output file

    Returns:
        None. The image file is produced and stored in the directory
    """
    rxn = rdrxn_from_string(smiles, inp_fmt='smiles')

    DrawingOptions.atomLabelFontSize = 55
    DrawingOptions.dotsPerAngstrom = 100
    DrawingOptions.bondLineWidth = 3.0
    rimage = Draw.ReactionToImage(rxn)
    rimage.save(filename)


###### descriptors
def calculate_atom_oxidation_number_by_EN(atom: Atom) -> int:
    pauling_en_map = {1: 2.2, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 11: 0.93, 12: 1.31,
                      13: 1.61,
                      14: 1.9, 15: 2.19, 16: 2.58, 17: 3.16, 19: 0.82, 20: 1, 21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66,
                      25: 1.55, 26: 1.83, 27: 1.88, 28: 1.91, 29: 1.9, 30: 1.65, 31: 1.81, 32: 2.01, 33: 2.18, 34: 2.55,
                      35: 2.96, 36: 3, 37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33, 41: 1.6, 42: 2.16, 43: 1.9, 44: 2.2,
                      45: 2.28, 46: 2.2, 47: 1.93, 48: 1.69, 49: 1.78, 50: 1.96, 51: 2.05, 52: 2.1, 53: 2.66, 54: 2.6,
                      55: 0.79, 56: 0.89, 57: 1.1, 58: 1.12, 59: 1.13, 60: 1.14, 62: 1.17, 64: 1.2, 66: 1.22, 67: 1.23,
                      68: 1.24, 69: 1.25, 71: 1.27, 72: 1.3, 73: 1.5, 74: 2.36, 75: 1.9, 76: 2.2, 77: 2.2, 78: 2.28,
                      79: 2.54, 80: 2, 81: 1.62, 82: 2.33, 83: 2.02, 84: 2, 85: 2.2, 88: 0.9, 89: 1.1, 90: 1.3, 91: 1.5,
                      92: 1.38, 93: 1.36, 94: 1.28, 95: 1.3, 96: 1.3, 97: 1.3, 98: 1.3, 99: 1.3, 100: 1.3, 101: 1.3,
                      102: 1.39}
    en_map = pauling_en_map
    a_atomic_number = atom.GetAtomicNum()
    bonds_neighbor = atom.GetBonds()

    def sf(number):
        if number > 0:
            return -1
        elif number < 0:
            return 1
        else:
            return 0

    # iterate overall bonds of the atom
    # for each bond
    # get the atomic number of the "other" atom
    # calculate the electronegativity difference  between the atom and its neighbor
    # normalize the electronic difference to  -1, 0, +1
    # account for high order bonds by multiplying the normalized value by the bond order
    # sum over the contributions of the individual atoms and add the formal charge of the atom
    # the calculation is split by Hs and non Hs so that we can avoind adding H atoms to the molecule,
    # and we avoid problems with the implicit/explicit H definition in rdkit

    neighbors_nonHs_contribution = [sf(en_map.get(a_atomic_number) - en_map.get(bond.GetOtherAtom(atom).GetAtomicNum()))
                                    * bond.GetBondTypeAsDouble()
                                    for bond in bonds_neighbor if bond.GetOtherAtom(atom).GetAtomicNum() != 1]

    neighbors_Hs_contribution = [sf(en_map.get(a_atomic_number) - en_map.get(1)) for x in range(atom.GetTotalNumHs())]

    oxidation_number = int(atom.GetFormalCharge() + sum(neighbors_nonHs_contribution) + sum(neighbors_Hs_contribution))

    return oxidation_number


def compute_oxidation_numbers(rdmol: Mol) -> Mol:
    rdmol_ = copy.deepcopy(rdmol)
    Chem.rdmolops.Kekulize(rdmol_)
    atoms = rdmol.GetAtoms()
    atoms_ = rdmol_.GetAtoms()
    for atom, atom_ in zip(atoms, atoms_):
        oxidation_number = calculate_atom_oxidation_number_by_EN(atom_)
        atom.SetIntProp('_OxidationNumber', oxidation_number)
    return rdmol


def calculate_molecular_hash_values(rdmol: Mol, hash_list: List[str] = None) -> dict:
    molhashf = HashFunction.names
    if hash_list:
        hash_list += ['CanonicalSmiles']
    else:
        hash_list = list(molhashf.keys()) + ['inchi_key', 'inchikey_KET_15T', 'noiso_smiles', 'cx_smiles']

    hash_map = {k: MolHash(rdmol, v) for k, v in molhashf.items() if k in hash_list}

    hash_map['smiles'] = hash_map.get('CanonicalSmiles')
    if 'inchi_key' in hash_list:
        hash_map['inchi'] = Chem.MolToInchi(rdmol)
        hash_map['inchi_key'] = Chem.InchiToInchiKey(hash_map['inchi'])
    if 'inchikey_KET_15T' in hash_list:
        hash_map['inchi_KET_15T'] = Chem.MolToInchi(rdmol, options='-KET -15T')
        hash_map['inchikey_KET_15T'] = Chem.InchiToInchiKey(hash_map['inchi_KET_15T'])
    if 'noiso_smiles' in hash_list:
        hash_map['noiso_smiles'] = Chem.MolToSmiles(rdmol, isomericSmiles=False)
    if 'cx_smiles' in hash_list:
        hash_map['cx_smiles'] = Chem.MolToCXSmiles(rdmol)
    """
    
    hash_map['ExtendedMurcko_AG'] = smiles_to_anonymus_graph(hash_map['ExtendedMurcko'])
    hash_map['ExtendedMurcko_EG'] = smiles_to_element_graph(hash_map['ExtendedMurcko'])
    hash_map['MurckoScaffold_AG'] = smiles_to_anonymus_graph(hash_map['MurckoScaffold'])
    hash_map['MurckoScaffold_EG'] = smiles_to_element_graph(hash_map['MurckoScaffold'])
    """

    return hash_map


def smiles_to_anonymus_graph(r):
    r1 = re.sub(r'[a-zA-Z]', r'*', r)  # replace all atoms with * wildcard
    for bt in ['-', ':', '=', '#', '/', '\\']:  # remove bond information
        r1 = r1.replace(bt, '')
    return r1


def smiles_to_element_graph(r):
    r1 = r.upper()  # remove aromatic info from atoms
    for bt in ['-', ':', '=', '#', '/', '\\']:  # remove bond information
        r1 = r1.replace(bt, '')
    return r1


if __name__ == '__main__':
    print('main mode')
