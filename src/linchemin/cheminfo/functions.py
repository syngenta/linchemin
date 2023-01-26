import copy
from typing import Callable, List, Dict
from functools import partial
import re
import rdkit
from rdkit import Chem
from rdkit.Chem import rdchem, rdChemReactions, DataStructs, rdFingerprintGenerator, Draw
from rdkit.Chem.rdchem import Mol, Atom
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

        :param
            rdmol: an RDKit Mol object

        :return:
            a tuple containing the canonical order of the atoms
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

        :param:
            rdmol: a rdkit.Chem.rdchem.Mol object

        :return:
            rdmol_canonicalized: a rdkit.Chem.rdchem.Mol object with the atoms in canonical order
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


def canonicalize_mapped_rdmol(mapped_rdmol: rdkit.Chem.rdchem.Mol) -> rdkit.Chem.rdchem.Mol:
    """ To ensure that the atom ids in a mapped rdkit Mol object are independent from the mapping.
        The presence of the atom mapping has an impact on the atom ids, so that for the same molecule
        they can be different. This is not acceptable for us, as we use those ids to track an atom along a route
        and to do so we need them to be always identical for a given mapped molecule.

        Function taken from:
        https://sourceforge.net/p/rdkit/mailman/message/35862258/
        https://gist.github.com/ptosco/36574d7f025a932bc1b8db221903a8d2
    """
    # the map numbers are extracted and stored in the dictionary mapping them onto the 'original' atom indices
    d = {}
    for atom in mapped_rdmol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            d[atom.GetIdx()] = atom.GetAtomMapNum()
            # the map numbers are removed, so that they do not impact the atom ordering
            atom.SetAtomMapNum(0)

    # the new indices are mapped onto the old ones
    canon_idx_old_idx = [(j, i) for i, j in
                         enumerate(Chem.CanonicalRankAtoms(mapped_rdmol))]  # [(0, 0), (2, 1), (1, 2)]
    # the old indices are sorted according to the new ones
    old_idcs_sorted_by_canon_idcs = tuple(zip(*sorted(canon_idx_old_idx)))  # ((0, 1, 2), (0, 2, 1))
    # the canonical order to be used is extracted
    canonical_order = old_idcs_sorted_by_canon_idcs[1]  # (0, 2, 1)
    # the map numbers are mapped onto the new indices with a lookup through the old indices
    mapping = {}
    for new_id, old_id in canon_idx_old_idx:
        if (mapping_n := [m for old, m in d.items() if old == old_id]):
            mapping[new_id] = int(mapping_n[0])
        else:
            mapping[new_id] = 0
    # a new rdkit Mol object is created with the atoms in canonical order
    new_mapped_rdmol = Chem.RenumberAtoms(mapped_rdmol, canonical_order)
    # the map numbers are reassigned to the correct atoms
    for atom in new_mapped_rdmol.GetAtoms():
        i = atom.GetIdx()
        for aid, d in mapping.items():
            if aid == i:
                atom.SetAtomMapNum(d)

    return new_mapped_rdmol


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


def is_mapped_molecule(rdmol):
    """ To check whether a rdmol is mapped """
    mapped_atoms = [a.GetAtomMapNum() for a in rdmol.GetAtoms()]
    if not mapped_atoms or set(mapped_atoms) == {0} or set(mapped_atoms) == {-1}:
        return False
    else:
        return True


####################################################################################################
# RDRXN
def rdrxn_from_string(input_string: str, inp_fmt: str) -> rdChemReactions.ChemicalReaction:
    format_function_map = {'smiles': partial(Chem.rdChemReactions.ReactionFromSmarts, useSmiles=True),
                           'smarts': partial(Chem.rdChemReactions.ReactionFromSmarts, useSmiles=False),
                           'rxn': Chem.rdChemReactions.ReactionFromRxnBlock}

    func = format_function_map.get(inp_fmt)
    return func(input_string)


def rdrxn_to_string(rdrxn: rdChemReactions.ChemicalReaction, out_fmt: str, use_atom_mapping: bool = False) -> str:
    if not use_atom_mapping:
        rdrxn_ = copy.deepcopy(rdrxn)
        Chem.rdChemReactions.RemoveMappingNumbersFromReactions(rdrxn_)
    else:
        rdrxn_ = rdrxn
    function_map = {'smiles': partial(Chem.rdChemReactions.ReactionToSmiles, canonical=True),
                    'smarts': Chem.rdChemReactions.ReactionToSmarts,
                    # 'rxn': partial(Chem.rdChemReactions.ReactionToRxnBlock, forceV3000=True),
                    'rxn': Chem.rdChemReactions.ReactionToRxnBlock,
                    # 'rxn': Chem.rdChemReactions.ReactionToV3KRxnBlock,
                    }
    func = function_map.get(out_fmt)
    return func(rdrxn_)


def rdrxn_to_rxn_mol_catalog(rdrxn: rdChemReactions.ChemicalReaction) -> Dict[str, List[Mol]]:
    """ To build from a rdkit rdrxn a dictionary in the form {'role': [Mol]}"""
    return {'reactants': list(rdrxn.GetReactants()),
            'reagents': list(rdrxn.GetAgents()),
            'products': list(rdrxn.GetProducts())}


def rdrxn_from_rxn_mol_catalog(rxn_mol_catalog: Dict[str, List[Mol]]) -> rdChemReactions.ChemicalReaction:
    """ To build a dictionary in the form {'role': [Mol]} a rdkit rdrxn """
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


def rdrxn_to_molecule_catalog(rdrxn: rdChemReactions.ChemicalReaction, constructor):
    """ To build from a rdkit rdrxn a dictionary in the form {'role': [Molecule]}"""
    reaction_rdmols = rdrxn_to_rxn_mol_catalog(rdrxn=rdrxn)
    mol_catalog = {}
    for role, rdmols in reaction_rdmols.items():
        mol_catalog[role] = [constructor.build_from_rdmol(rdmol=rdmol) for r, rdmol_list in reaction_rdmols.items()
                             if r == role for rdmol in rdmol_list]
    return mol_catalog


def has_mapped_products(rdrxn: rdChemReactions.ChemicalReaction) -> bool:
    """ To check if a rdrxn has any mapped product """
    if any([is_mapped_molecule(mol) for mol in list(rdrxn.GetProducts())]):
        return True
    else:
        return False


def select_desired_product(mol_catalog: dict):
    """ To select the 'desired product' among the products of a reaction.

        :param:
            mol_catalog: a dictionary with Molecule instances {'reactants_reagents': [Molecule], 'products': [Molecule]}

        :return:
            the Molecule instance of the desired product
    """
    return mol_catalog['products'][0]


def rdrxn_role_reassignment(rdrxn: rdChemReactions.ChemicalReaction,
                            desired_product_idx: int = 0) -> rdChemReactions.ChemicalReaction:
    """function to reassign the reactant/reagent role in a ChemicalReaction by identifying
    1) reactant molecules that should be reagent
    2) reagent molecules that should be reactants
    and reconstructing a ChemicalReaction with correct roles
    this role reassignment is requires reaction atom to atom mapping to identify reactants/reagents

    a reactant contributes with at least one heavy atom to the desired product of a reaction
    a reagent does not contribute with any heavy atom to the desired product of a reaction
    either reactant or reagents can contribute to by-products

        :param:
            rdrxn: the rdChemReactions.ChemicalReaction object for which the role reassignment should be performed

            desired_product_idx: the integer index of the reference product used to assess the reactant contribution, default: 0

        :return: the rdChemReactions.ChemicalReaction object with reassigned roles
    """
    if not rdChemReactions.HasReactionAtomMapping(rdrxn):
        return rdrxn
    molecule_catalog = {'reactants': [{'rdmol': rdmol,
                                       'role_molecule_idx': i,
                                       'matam': {atom.GetIdx(): atom.GetAtomMapNum() for atom in rdmol.GetAtoms()}} for
                                      i, rdmol in enumerate(rdrxn.GetReactants())],
                        'reagents': [{'rdmol': rdmol,
                                      'role_molecule_idx': i,
                                      'matam': {atom.GetIdx(): atom.GetAtomMapNum() for atom in rdmol.GetAtoms()}} for
                                     i, rdmol in enumerate(rdrxn.GetAgents())],
                        'products': [{'rdmol': rdmol,
                                      'role_molecule_idx': i,
                                      'matam': {atom.GetIdx(): atom.GetAtomMapNum() for atom in rdmol.GetAtoms()}} for
                                     i, rdmol in enumerate(rdrxn.GetProducts())]}

    rxn_mol_catalog = {'reactants': [],
                       'reagents': [],
                       'products': [item.get('rdmol') for item in molecule_catalog.get('products')]}

    keys_output = set(molecule_catalog.get('products')[desired_product_idx].get('matam').values())

    for input_item in molecule_catalog.get('reactants') + molecule_catalog.get('reagents'):
        keys_input = set(input_item.get('matam').values())
        x = keys_input.intersection(keys_output) - {0} - {-1}
        rdmol = input_item.get('rdmol')
        if len(x) == 0:
            rxn_mol_catalog.get('reagents').append(rdmol)
        else:
            rxn_mol_catalog.get('reactants').append(rdmol)
    return rdrxn_from_rxn_mol_catalog(rxn_mol_catalog=rxn_mol_catalog)


def role_reassignment(reaction_mols: dict, ratam, desired_product):
    """ To reassign the roles of reactants and reagents based on the mapping on the desired product.

        :param:
            reaction_mols: a dictionary in the form {'reactants_reagents': [Molecule], 'products': [Molecule]}

            ratam: a Ratam object containing the mapping information of the reaction of interest

            desired_product: the Molecule instance of the desired product

        :return: a dictionary in the form {'reactants': [Molecule], 'reagents': [Molecule], 'products': [Molecule]}
                 with the new roles based on atom mapping


    """
    if desired_product not in reaction_mols['products']:
        logger.error('The selected product is not among the reaction products.')
        return None
    desired_product_transformations = [at for at in ratam.atom_transformations if at.product_uid == desired_product.uid]
    true_reactants_uid = {at.reactant_uid for at in desired_product_transformations}
    true_reagents = {r.uid for r in reaction_mols['reactants_reagents'] if r.uid not in true_reactants_uid}
    true_reactants = {r.uid for r in reaction_mols['reactants_reagents'] if r.uid in true_reactants_uid}
    products = [m.uid for m in reaction_mols['products']]
    reagents = check_reagents(ratam.full_map_info, true_reagents)
    return {'reactants': sorted(list(true_reactants)),
            'reagents': sorted(list(reagents)),
            'products': sorted(products)}


def check_reagents(full_map_info, reagents):
    """ To add unmapped molecule as reagents of a chemical equation.

        :param:
            full_map_info: a dictionary in the form {uid: [map]} [attribute of the Ratam class]

            reagents: a set containing the uid of the already discovered reagents

        :return: reagents: an updated set, containing the uid of unmapped reagents.
    """
    for uid, map_list in full_map_info.items():
        for d in map_list:
            map_nums = list(d.values())
            if not map_nums or set(map_nums) == {0} or set(map_nums) == {-1}:
                reagents.add(uid)
    return reagents


def mapping_diagnosis(chemical_equation, desired_product):
    """ To check possible issues in the atom mapping: (i) if there are unmapped atoms in the desired product (issues
        in computing route metrics); (ii) if there are unmapped atoms in the reactants (possible hint for leaving groups)

        :param:
            chemical_equation: the ChemicalEquation instance of interest

            desired_product: the Molecule instance of the desired product

        :return: unmapped_fragments: a list of smiles referring to the unmapped atoms of each reactant
    """

    if (unmapped_atoms_in_desired_product := [a for a in desired_product.rdmol_mapped.GetAtoms()
                                          if a.GetAtomMapNum() in [0, -1]]):
        logger.warning('Some atoms in the desired product remain unmapped: possible important reactants are missing')

    for uid in chemical_equation.role_map['reactants']:
        mols = [m.rdmol_mapped for u, m in chemical_equation.catalog.items() if u == uid]
        unamapped_fragments = []
        for m in mols:
            if (unmapped_atoms := [a for a in m.GetAtoms() if a.GetAtomMapNum() in [0, -1]]):
                atoms_indices = [a.GetIdx() for a in unmapped_atoms]
                fragment = Chem.rdmolfiles.MolFragmentToSmiles(m, atomsToUse=atoms_indices,
                                                               atomSymbols=[a.GetSymbol() for a in m.GetAtoms()])

                unamapped_fragments.append(fragment)
    return unamapped_fragments


def rdchiral_extract_template(reaction_string: str, inp_fmt: str, reaction_id: int = None):
    if inp_fmt != 'smiles':
        raise NotImplementedError
    mapped_smiles_split = reaction_string.split('>')
    rdchiral_input = {'_id': reaction_id,
                      'reactants': mapped_smiles_split[0],
                      'agents': mapped_smiles_split[1],
                      'products': mapped_smiles_split[2]}
    return template_extractor.extract_from_reaction(reaction=rdchiral_input)


def build_rdrxn(catalog: Dict,
                role_map: Dict,
                stoichiometry_coefficients: Dict,
                use_reagents: bool,
                use_smiles: bool,
                use_atom_mapping: bool) -> rdChemReactions.ChemicalReaction:
    reaction_smiles_empty = '>>'
    rdrxn_new = rdChemReactions.ReactionFromSmarts(reaction_smiles_empty, useSmiles=use_smiles)
    for mol_id in role_map.get('reactants'):
        if rdmol_mapped := catalog.get(mol_id).rdmol_mapped:
            for _ in range(stoichiometry_coefficients.get('reactants').get(mol_id)):
                rdrxn_new.AddReactantTemplate(rdmol_mapped)

    if use_reagents:
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


####################################################################################################
# Descriptors
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
