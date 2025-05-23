import copy
import re
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import rdkit
from rdchiral import template_extractor
from rdkit import Chem
from rdkit.Chem import (
    DataStructs,
    Descriptors,
    rdchem,
    rdChemReactions,
    rdFingerprintGenerator,
)
from rdkit.Chem.rdchem import Atom, Mol
from rdkit.Chem.rdMolHash import HashFunction, MolHash

import linchemin.utilities as utilities

# RDLogger.DisableLog('rdApp.*')

logger = utilities.console_logger(__name__)


class CanonicalizationError(Exception):
    """To be raised if a canonicalization fails"""

    pass


# RDMOLECULE
def rdmol_from_string(input_string: str, inp_fmt: str) -> Mol:
    """To generate an RDKit Mol object from a molecular string"""
    function_map = {
        "smiles": Chem.MolFromSmiles,
        "smarts": Chem.MolFromSmarts,
        "mol_block": Chem.MolFromMolBlock,
    }
    func = function_map.get(inp_fmt)
    return func(input_string)


def compute_mol_smiles(rdmol: Mol, isomeric_smiles: bool = True) -> str:
    """To compute a molecular smiles from a rdkit Mol object"""
    return Chem.MolToSmiles(rdmol, isomericSmiles=isomeric_smiles)


def compute_mol_smarts(rdmol: Mol, isomeric_smiles: bool = True) -> str:
    """To compute a molecular smarts from a rdkit Mol object"""
    return Chem.MolToSmarts(rdmol, isomericSmiles=isomeric_smiles)


def compute_mol_inchi_key(rdmol: Mol) -> str:
    """To compute a molecular inchi from a rdkit Mol object"""
    inchi = Chem.MolToInchi(rdmol, options="-KET -15T")
    return Chem.InchiToInchiKey(inchi)


def compute_mol_blockV3k(rdmol: Mol) -> str:
    """To compute the V3000 MolBlock corresponding to a rdkit Mol object"""
    return Chem.MolToV3KMolBlock(rdmol)


def compute_mol_block(rdmol: Mol) -> str:
    """To compute the MolBlock corresponding to a rdkit Mol object"""
    return Chem.MolToMolBlock(rdmol)


def remove_rdmol_atom_mapping(rdmol: Mol) -> Mol:
    """To remove atom mapping from an RDKit Mol object"""
    rdmol_unmapped = copy.deepcopy(rdmol)
    [a.SetAtomMapNum(0) for a in rdmol_unmapped.GetAtoms()]
    return rdmol_unmapped


def detect_rdmol_problems(rdmol):
    task_name = "detect_problems"
    if rdmol:
        problems = Chem.DetectChemistryProblems(rdmol)
        problem_log = [
            {"type": problem.GetType(), "message": problem.Message()}
            for problem in problems
        ]
        is_successful = True
    else:
        problem_log = []
        is_successful = False
    log = {
        "is_successful": is_successful,
        "task_name": task_name,
        "message": problem_log,
    }
    return rdmol, log


def sanitize_rdmol(rdmol):
    task_name = "sanitize"
    if rdmol:
        try:
            sanit_fail = Chem.SanitizeMol(
                rdmol,
                catchErrors=False,
                sanitizeOps=rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_ALL,
            )
            message = sanit_fail
            is_successful = True
        except Exception as e:
            exception = e
            rdmol = None
            is_successful = False
            message = exception
    else:
        message = ""
        is_successful = False
    log = {"is_successful": is_successful, "task_name": task_name, "message": message}
    return rdmol, log


def get_canonical_order(rdmol: Mol) -> tuple:
    """
    https://gist.github.com/ptosco/36574d7f025a932bc1b8db221903a8d2

    Parameters:
    -----------
    rdmol: Mol
        The input rdkit Mol object

    Returns:
    canonical order: tuple
        The canonical order of the atom sin the input molecule
    """
    canon_idx_old_idx = [(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(rdmol))]
    old_idcs_sorted_by_canon_idcs = tuple(zip(*sorted(canon_idx_old_idx)))
    return old_idcs_sorted_by_canon_idcs[1]


def map_atoms_to_canon_mol(
    mol_mapped: Mol, atom_indices: Iterable[int]
) -> Tuple[int, ...]:
    """
    To get a map between the atoms' indices in a mapped rdkit Mol and those in the canonical smiles.

    Parameters
    ------------
    mol_mapped:Mol
        The mapped rdkit Mol object
    atom_indices: Iterable[int]
        The iterable with the atoms' indices for which the corresponding canonical indices are needed

    Returns
    --------
    Tuple[int, ...]
        The tuple containing the canonical indices for the input atoms

    """
    mol = copy.deepcopy(mol_mapped)
    # remove atom mapping
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    # save as canonical smiles to compute _smilesAtomOutputOrder
    _ = Chem.MolToSmiles(mol)
    # _smilesAtomOutputOrder store mapping between atom-mapped and canonical molecules
    # e.g. mapper = [0, 6, 3, 7, 2, 8, 1, 5, 4]
    # atom with idx=7 in atom-mapped molecule will have idx=3 in canonical one
    # atom with idx=2 in atom-mapped molecule will have idx=4 in canonical one
    # https://github.com/rdkit/rdkit/discussions/5091
    mapper = list(map(int, mol.GetProp("_smilesAtomOutputOrder")[1:-2].split(",")))
    return tuple(mapper.index(atom_idx) for atom_idx in atom_indices)


def canonicalize_rdmol(rdmol: Mol) -> Mol:
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

    Parameters:
    ------------
    rdmol: Mol
        The input rdkit Mol object

    Returns:
    ---------
    rdmol_canonicalized: Mol
        The molecule object with the atoms in canonical order
    """
    task_name = "canonicalize_rdmol"
    message: Union[str, Exception]
    if rdmol:
        try:
            new_order = get_canonical_order(rdmol=rdmol)
            rdmol = Chem.RenumberAtoms(rdmol, list(new_order))
            is_successful = True
            message = ""
        except Exception as e:
            exception = e
            rdmol = None
            is_successful = False
            message = exception
    else:
        message = ""
        is_successful = False
    log = {"is_successful": is_successful, "task_name": task_name, "message": message}

    return rdmol, log


def canonicalize_rdmol_lite(
    rdmol: Mol, is_pattern: bool = False, use_extended_info: bool = True
) -> Mol:
    """
    To canonicalize an RDKit molecule with options for pattern (SMARTS) and extended information (CXSMILES).

    Args:
        rdmol (Mol): The RDKit molecule to canonicalize.
        is_pattern (bool, optional): Whether the molecule should be treated as a pattern (SMARTS). Defaults to False.
        use_extended_info (bool, optional): Whether to use extended canonicalization information (CXSMILES). Defaults to True.

    Returns:
        Mol: The canonicalized molecule

    Raises:
        CanonicalizationError: if the canonicalization fails
    """
    try:
        if is_pattern:
            mol_to_str = Chem.MolToCXSmarts if use_extended_info else Chem.MolToSmarts
            str_to_mol = Chem.MolFromSmarts if use_extended_info else Chem.MolFromSmarts
        else:
            mol_to_str = (
                partial(
                    Chem.MolToCXSmiles,
                    canonical=True,
                )
                if use_extended_info
                else Chem.MolToSmiles
            )
            str_to_mol = (
                Chem.MolFromSmiles
            )  # Smiles and CXSmiles use the same MolFrom function

        mol_str = mol_to_str(rdmol)
        rdmol = str_to_mol(mol_str)

        return rdmol
    except Exception as e:
        logger.error("Molecule canonicalization failed")
        # Log the error, raise an exception, or return None depending on the desired behavior
        raise CanonicalizationError


def canonicalize_mapped_rdmol(mapped_rdmol: Mol) -> Mol:
    """
    To ensure that the atom ids in a mapped rdkit Mol object are independent of the mapping.
    The presence of the atom mapping has an impact on the atom ids, so that for the same molecule
    they can be different. This is not acceptable for us, as we use those ids to track an atom along a route
    and to do so we need them to be always identical for a given mapped molecule.

    Function taken from:
    https://sourceforge.net/p/rdkit/mailman/message/35862258/
    https://gist.github.com/ptosco/36574d7f025a932bc1b8db221903a8d2

    Parameters:
    ------------
    mapped_rdmol : Mol
        The mapped rdkit Mol object for which the atoms need to ve canonicalized

    Returns:
    --------
    canonicalized_mol: Mol
        The mapped rdkit Mol with atoms in canonical order
    """
    # the map numbers are extracted and stored in the dictionary mapping them onto the 'original' atom indices
    map_numbers = extract_map_numers(mapped_rdmol)

    # the canonical atom indices are mapped onto the old ones
    canon_idx_old_idx = [
        (j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mapped_rdmol))
    ]  # [(0, 0), (2, 1), (1, 2)]
    # the old indices are sorted according to the new ones
    old_idcs_sorted_by_canon_idcs = tuple(
        zip(*sorted(canon_idx_old_idx))
    )  # ((0, 1, 2), (0, 2, 1))

    # the canonical order to be used is extracted
    canonical_order = old_idcs_sorted_by_canon_idcs[1]  # (0, 2, 1)

    # the map numbers are mapped onto the new indices with a lookup through the old indices
    mapping = map_number_lookup(map_numbers, canon_idx_old_idx)

    # a new rdkit Mol object is created with the atoms in canonical order,
    # assigning the map numbers to the correct atoms
    new_mapped_rdmol = Chem.RenumberAtoms(mapped_rdmol, canonical_order)

    assign_map_numbers(new_mapped_rdmol, mapping)

    return new_mapped_rdmol


def extract_map_numers(mapped_rdmol: Mol) -> dict:
    """To extract the map numbers from the atoms of a Mol object"""
    d = {}
    for atom in mapped_rdmol.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            d[atom.GetIdx()] = atom.GetAtomMapNum()
            # the map numbers are removed, so that they do not impact the atom ordering
            atom.SetAtomMapNum(0)
    return d


def extract_atom_properties(rdmol: Mol) -> Dict[int, Dict[str, Any]]:
    """
    To extract the properties of all atoms in an RDKit molecule and clear them from the molecule.

    Args:
        rdmol (Mol): The RDKit molecule from which to extract atom properties.

    Returns:
        Dict[int, Dict[str, Any]]: A dictionary mapping atom indices to their respective properties.
    """
    properties = {}
    for atom in rdmol.GetAtoms():
        atom_idx = atom.GetIdx()
        props = atom.GetPropsAsDict(True, False)
        properties[atom_idx] = props
        # Clearing properties using ClearProp on each property name
        for prop_name in list(
            props.keys()
        ):  # Create a list to avoid modifying the dict while iterating
            atom.ClearProp(prop_name)
    return properties


def atomic_properties_lookup(
    properties_old_ids: Dict[int, Any], canon_idx_old_idx: List[Tuple[int, int]]
) -> Dict[int, Any]:
    """
    To associate the atomic properties with new atom IDs based on a mapping from new IDs to old IDs.

    Args:
        properties_old_ids (Dict[int, Any]): A dictionary containing atomic properties keyed by old atom IDs.
        canon_idx_old_idx (List[Tuple[int, int]]): A list of tuples mapping new atom IDs to old atom IDs.

    Returns:
        Dict[int, Any]: A dictionary containing atomic properties keyed by new atom IDs.

    Raises:
        KeyError: If an old atom ID from the mapping does not exist in the properties_old_ids dictionary.
    """
    properties_new_ids = {}
    for new_id, old_id in canon_idx_old_idx:
        if old_id not in properties_old_ids:
            raise KeyError(
                f"Old atom ID {old_id} is not present in the properties dictionary."
            )
        properties_new_ids[new_id] = properties_old_ids[old_id]
    return properties_new_ids


def set_atomic_properties(rdmol: Mol, atomic_properties: dict) -> None:
    """
    To set properties on atoms in an RDKit molecule based on the provided atomic properties dictionary.

    Args:
        rdmol (Chem.Mol): The RDKit molecule to which properties will be added.
        atomic_properties (Dict[int, Dict[str, Any]]): A dictionary mapping atom indices to property dictionaries.
    """
    for atom in rdmol.GetAtoms():
        atom_idx = atom.GetIdx()
        # Directly access the dictionary for the current atom index if it exists
        props_dict = atomic_properties.get(atom_idx, {})
        for prop_name, prop_value in props_dict.items():
            # Use the appropriate RDKit method to set the property based on its type
            if isinstance(prop_value, int):
                atom.SetIntProp(prop_name, prop_value)
            elif isinstance(prop_value, float):
                atom.SetDoubleProp(prop_name, prop_value)
            elif isinstance(prop_value, str):
                atom.SetProp(prop_name, prop_value)
            else:
                logger.warning(
                    f"Unsupported property found: {prop_name}: {prop_value} ({type(prop_value)})"
                )
                pass


def new_molecule_canonicalization(rdmol: Mol) -> Mol:
    """
    To canonicalize an RDKit molecule by renumbering its atoms according to their canonical
    ranking and reassigning their properties accordingly.

    Args:
        rdmol (Mol): The RDKit molecule to be canonicalized.

    Returns:
        Mol: A new RDKit molecule object with atoms renumbered and original properties reassigned.
    """
    # Update the property cache of the molecule
    rdmol.UpdatePropertyCache()

    # Extract atom properties and clear them from the original molecule
    properties_old_ids = extract_atom_properties(rdmol)

    # Determine the canonical ranking of atoms, including chirality
    canonical_ranks = Chem.CanonicalRankAtoms(rdmol, includeChirality=True)

    # Create a mapping of new canonical indices to old indices
    canonical_to_old_idx_map = [
        (new_idx, old_idx) for old_idx, new_idx in enumerate(canonical_ranks)
    ]

    # Sort the old indices according to the new canonical indices
    canonical_order = [old_idx for _, old_idx in sorted(canonical_to_old_idx_map)]

    # Lookup the old properties and map them to the new canonical indices
    properties_new_ids = atomic_properties_lookup(
        properties_old_ids, canonical_to_old_idx_map
    )

    # Renumber the atoms of the original molecule according to the canonical order
    new_rdmol = Chem.RenumberAtoms(rdmol, canonical_order)

    # Reassign the properties to the atoms in the new molecule
    set_atomic_properties(new_rdmol, properties_new_ids)

    # Sanitize molecule: following the procedure described here https://github.com/rdkit/rdkit/issues/2361
    Chem.SanitizeMol(new_rdmol)
    cleaning = True
    force = True
    flag_possible = True
    Chem.AssignStereochemistry(new_rdmol, cleaning, force, flag_possible)
    return new_rdmol


def map_number_lookup(map_numbers: dict, canon_idx_old_idx: tuple) -> dict:
    """To associate the map numbers of the old atoms' id with the new ids"""
    mapping = {}
    for new_id, old_id in canon_idx_old_idx:
        if mapping_n := next(
            (m for old, m in map_numbers.items() if old == old_id), None
        ):
            mapping[new_id] = int(mapping_n)
        else:
            mapping[new_id] = 0
    return mapping


def assign_map_numbers(new_mapped_rdmol: Mol, mapping: dict) -> None:
    """To assign map numbers to the atoms of a rdkit Mol"""
    for atom in new_mapped_rdmol.GetAtoms():
        i = atom.GetIdx()
        for aid, d in mapping.items():
            if aid == i:
                atom.SetAtomMapNum(d)


def rdmol_to_bstr(rdmol: rdkit.Chem.rdchem.Mol):
    return rdmol.ToBinary()


def bstr_to_rdmol(rdmol_bstr: str) -> rdkit.Chem.rdchem.Mol:
    return rdkit.Chem.rdchem.Mol(rdmol_bstr)


MolPropertyFunction = Callable[[Mol], str]


def get_mol_property_function(property_name: str) -> MolPropertyFunction:
    mol_property_function_factory = {
        "smiles": compute_mol_smiles,
        "inchi_key": compute_mol_inchi_key,
    }
    return mol_property_function_factory.get(property_name)


def is_mapped_molecule(rdmol: Mol) -> bool:
    """To check whether a rdmol is mapped"""
    mapped_atoms = [a.GetAtomMapNum() for a in rdmol.GetAtoms()]
    if not mapped_atoms or set(mapped_atoms) == {0} or set(mapped_atoms) == {-1}:
        return False
    else:
        return True


def has_enhanced_stereo(rdmol: Mol) -> bool:
    """To check whether a rdmol has enhanced stereochemistry information"""
    if rdmol.GetStereoGroups():
        return True
    else:
        return False


def compute_molecular_weigth(rdmol: Mol) -> float:
    """To compute the molecular weigth of the input rdmol"""
    return Descriptors.ExactMolWt(rdmol)


####################################################################################################
# RDRXN
def rdrxn_from_string(
    input_string: str, inp_fmt: str
) -> rdChemReactions.ChemicalReaction:
    """To build a rdkit rdrxn object from a reaction string"""
    format_function_map = {
        "smiles": partial(Chem.rdChemReactions.ReactionFromSmarts, useSmiles=True),
        "smarts": partial(Chem.rdChemReactions.ReactionFromSmarts, useSmiles=False),
        "rxn_block": Chem.rdChemReactions.ReactionFromRxnBlock,
    }

    func = format_function_map.get(inp_fmt)
    return func(input_string)


def rdrxn_to_string(
    rdrxn: rdChemReactions.ChemicalReaction,
    out_fmt: str,
    use_atom_mapping: bool = False,
) -> str:
    """To build a reaction smiles from a rdkit rdrxn object"""
    if not use_atom_mapping:
        rdrxn_ = copy.deepcopy(rdrxn)
        Chem.rdChemReactions.RemoveMappingNumbersFromReactions(rdrxn_)
    else:
        rdrxn_ = rdrxn
    function_map = {
        "smiles": partial(Chem.rdChemReactions.ReactionToSmiles, canonical=True),
        "smarts": Chem.rdChemReactions.ReactionToSmarts,
        "rxn": partial(
            Chem.rdChemReactions.ReactionToRxnBlock,
            forceV3000=True,
            separateAgents=True,
        ),
        "rxn_blockV2K": Chem.rdChemReactions.ReactionToRxnBlock,
        "rxn_blockV3K": partial(
            Chem.rdChemReactions.ReactionToV3KRxnBlock, separateAgents=True
        ),
    }
    func = function_map.get(out_fmt)
    return func(rdrxn_)


def rdrxn_to_rxn_mol_catalog(
    rdrxn: rdChemReactions.ChemicalReaction,
) -> Dict[str, List[Mol]]:
    """To build from a rdkit rdrxn a dictionary in the form {'role': [Mol]}"""
    return {
        "reactants": list(rdrxn.GetReactants()),
        "reagents": list(rdrxn.GetAgents()),
        "products": list(rdrxn.GetProducts()),
    }


def rdrxn_from_rxn_mol_catalog(
    rxn_mol_catalog: Dict[str, List[Mol]]
) -> rdChemReactions.ChemicalReaction:
    """To build a dictionary in the form {'role': [Mol]} a rdkit rdrxn"""
    reaction_smiles_empty = ">>"
    rdrxn = rdChemReactions.ReactionFromSmarts(reaction_smiles_empty, useSmiles=True)
    for rdmol in rxn_mol_catalog.get("reactants"):
        rdrxn.AddReactantTemplate(rdmol)
    for rdmol in rxn_mol_catalog.get("reagents"):
        rdrxn.AddAgentTemplate(rdmol)
    for rdmol in rxn_mol_catalog.get("products"):
        rdrxn.AddProductTemplate(rdmol)
    rdChemReactions.SanitizeRxn(rdrxn)
    return rdrxn


def rdrxn_to_molecule_catalog(
    rdrxn: rdChemReactions.ChemicalReaction, constructor
) -> dict:
    """To build from a rdkit rdrxn a dictionary in the form {'role': [Molecule]}"""
    reaction_rdmols = rdrxn_to_rxn_mol_catalog(rdrxn=rdrxn)
    mol_catalog = {}
    for role, rdmols in reaction_rdmols.items():
        mol_catalog[role] = [
            constructor.build_from_rdmol(rdmol=rdmol) for rdmol in rdmols
        ]
    return mol_catalog


def has_mapped_products(rdrxn: rdChemReactions.ChemicalReaction) -> bool:
    """To check if a rdrxn has any mapped product"""
    return any(is_mapped_molecule(mol) for mol in list(rdrxn.GetProducts()))


def select_desired_product(mol_catalog: list):
    """
    To select the 'desired product' among the products of a reaction.

    Parameters:
    -----------
    mol_catalog: dict
        The dictionary with Molecule instances {'reactants_reagents': [Molecule], 'products': [Molecule]}

    Returns:
    --------
    desired product: Molecule
        the Molecule instance corresponding to the desired product
    """
    d = {p: sum(atom.GetMass() for atom in p.rdmol.GetAtoms()) for p in mol_catalog}
    return max(d, key=d.get)


def get_heaviest_mol(mol_list: List):
    """
    To select the molecule with the bigger mass among a list of molecules
    """
    d = {p: sum(atom.GetMass() for atom in p.rdmol.GetAtoms()) for p in mol_list}
    return max(d, key=d.get)


def rdrxn_role_reassignment(
    rdrxn: rdChemReactions.ChemicalReaction, desired_product_idx: int = 0
) -> rdChemReactions.ChemicalReaction:
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
    molecule_catalog = {
        "reactants": [
            {
                "rdmol": rdmol,
                "role_molecule_idx": i,
                "matam": {
                    atom.GetIdx(): atom.GetAtomMapNum() for atom in rdmol.GetAtoms()
                },
            }
            for i, rdmol in enumerate(rdrxn.GetReactants())
        ],
        "reagents": [
            {
                "rdmol": rdmol,
                "role_molecule_idx": i,
                "matam": {
                    atom.GetIdx(): atom.GetAtomMapNum() for atom in rdmol.GetAtoms()
                },
            }
            for i, rdmol in enumerate(rdrxn.GetAgents())
        ],
        "products": [
            {
                "rdmol": rdmol,
                "role_molecule_idx": i,
                "matam": {
                    atom.GetIdx(): atom.GetAtomMapNum() for atom in rdmol.GetAtoms()
                },
            }
            for i, rdmol in enumerate(rdrxn.GetProducts())
        ],
    }

    rxn_mol_catalog = {
        "reactants": [],
        "reagents": [],
        "products": [item.get("rdmol") for item in molecule_catalog.get("products")],
    }

    keys_output = set(
        molecule_catalog.get("products")[desired_product_idx].get("matam").values()
    )

    for input_item in molecule_catalog.get("reactants") + molecule_catalog.get(
        "reagents"
    ):
        keys_input = set(input_item.get("matam").values())
        x = keys_input.intersection(keys_output) - {0} - {-1}
        rdmol = input_item.get("rdmol")
        if len(x) == 0:
            rxn_mol_catalog.get("reagents").append(rdmol)
        else:
            rxn_mol_catalog.get("reactants").append(rdmol)
    return rdrxn_from_rxn_mol_catalog(rxn_mol_catalog=rxn_mol_catalog)


def role_reassignment(reaction_mols: dict, ratam, desired_product) -> Union[dict, None]:
    """
    To reassign the roles of reactants and reagents based on the mapping on the desired product.

    Parameters:
    ------------
    reaction_mols: dict
        Dictionary in the form {'reactants_reagents': [Molecule], 'products': [Molecule]}
    ratam: Ratam
        Contains the mapping information of the reaction of interest
    desired_product: Molecule
        The desired product of the chemical reaction

    Returns:
    ---------
    new role map: dict
        Dictionary in the form {'reactants': [Molecule], 'reagents': [Molecule], 'products': [Molecule]}
        with the new roles assigned based on atom mapping

    """
    if desired_product not in reaction_mols["products"]:
        logger.error("The selected product is not among the reaction products.")
        return None
    desired_product_map_nums = [
        at.map_num
        for at in ratam.atom_transformations
        if at.product_uid == desired_product.uid
    ]
    # generating the new full_map_info dictionary
    full_map_info_new = generate_full_map_info_new(
        ratam.full_map_info, desired_product_map_nums
    )

    # cleaning up the new full_map_info dictionary
    full_map_info_new = clean_full_map_info(full_map_info_new)

    # the ratam's full_map_info dictionary is replaced by the new, more detailed, one
    ratam.full_map_info = full_map_info_new
    return {
        "reactants": sorted(list(full_map_info_new["reactants"].keys())),
        "reagents": sorted(list(full_map_info_new["reagents"].keys())),
        "products": sorted(list(full_map_info_new["products"].keys())),
    }


def generate_full_map_info_new(
    full_map_info: dict, desired_product_map_nums: list
) -> dict:
    """To build a full_map_info dictionary with detailed information about reactants and reagents"""
    # initialization of the new "full_map_info" dictionary
    full_map_info_new = {
        "reactants": {},
        "reagents": {},
        "products": full_map_info["products"],
    }

    # For each reactant or reagent
    for uid, map_list in full_map_info["reactants_reagents"].items():
        full_map_info_new["reagents"][uid] = []
        full_map_info_new["reactants"][uid] = []
        for a_map in map_list:
            # if at least one atom is mapped onto the desired product, the molecule is considered a reactant
            if [n for n in list(a_map.values()) if n in desired_product_map_nums]:
                full_map_info_new["reactants"][uid].append(a_map)
            # otherwise, it is considered a reagent
            else:
                full_map_info_new["reagents"][uid].append(a_map)
    return full_map_info_new


def clean_full_map_info(full_map_info_new: dict) -> dict:
    """To remove reference of molecules with no mapping"""
    for role in list(full_map_info_new.keys()):
        for uid in list(full_map_info_new[role].keys()):
            if not full_map_info_new[role][uid]:
                del full_map_info_new[role][uid]
    return full_map_info_new


def mapping_diagnosis(chemical_equation, desired_product) -> Union[list, None]:
    """
    To validate the chemical equation with atom mapping.
    To check possible issues in the atom mapping:
    (i) if there are unmapped atoms in the desired product (issues in computing route metrics, missing reactants);
    (ii) if there are unmapped atoms in the reactants (possible hint for leaving groups)

    Parameters:
    ------------
    chemical_equation: ChemicalEquation
        The ChemicalEquation instance of interest
    desired_product: Molecule
        The Molecule instance of the desired product

    Returns:
    ---------
    unmapped_fragments: Union[list, None]
        The list of smiles referring to the unmapped atoms of each reactant
    """
    check_product_mapping(
        desired_product, chemical_equation.mapping.atom_transformations
    )
    reactants = chemical_equation.get_reactants()
    if unmapped_fragments := check_reactants_mapping(reactants):
        return unmapped_fragments


def check_reactants_mapping(reactants: list) -> list:
    """To check if there are unmapped atoms in the reactants"""
    unmapped_fragments = []
    for m in [mol.rdmol_mapped for mol in reactants]:
        if unmapped_atoms := [a for a in m.GetAtoms() if a.GetAtomMapNum() in [0, -1]]:
            atoms_indices = [a.GetIdx() for a in unmapped_atoms]
            fragment = Chem.rdmolfiles.MolFragmentToSmiles(
                m,
                atomsToUse=atoms_indices,
                atomSymbols=[a.GetSymbol() for a in m.GetAtoms()],
            )

            unmapped_fragments.append(fragment)
    return unmapped_fragments


def check_product_mapping(desired_product, atom_transformations: list) -> None:
    """To check if there are unmapped atoms in the desired product"""
    at_desired_product = [
        at for at in atom_transformations if at.product_uid == desired_product.uid
    ]
    if len(at_desired_product) != desired_product.rdmol.GetNumAtoms() or [
        a
        for a in desired_product.rdmol_mapped.GetAtoms()
        if a.GetAtomMapNum() in [0, -1]
    ]:
        logger.warning(
            "Some atoms in the desired product remain unmapped: possible important reactants are missing"
        )


def get_hydrogenation_info(
    disconnection_rdmol: Mol, hydrogenated_atoms: List[dict]
) -> Tuple[list, Mol]:
    """
    It adds new bonds between reacting atoms and hydrogen atoms, if any.

    Parameters:
    ------------
    disconnection_rdmol: Mol to
        The rdkit Mol object to which hydrogen atoms will be added
    hydrogenated_atoms: List[dict]
        The list of dictionaries mapping atoms id with the variation in number of bonded hydrogen, if any

    Returns:
    ---------
    bonds, disconnection_rdmol: Tuple[list, Mol]
        the list of atoms id pairs between which a new bond is formed and the rdkit Mol object
        to which explicit hydrogen atoms are added
    """
    bonds = []
    for hydrogen_info in hydrogenated_atoms:
        if hydrogen_info[1] > 0:
            p_atom_idx = hydrogen_info[0]
            disconnection_rdmol = Chem.AddHs(
                disconnection_rdmol, onlyOnAtoms=[p_atom_idx]
            )
            h_idxs = [
                a.GetIdx()
                for a in disconnection_rdmol.GetAtoms()
                if a.GetSymbol() == "H"
                and disconnection_rdmol.GetBondBetweenAtoms(a.GetIdx(), p_atom_idx)
            ]

            bonds.extend(sorted((i, p_atom_idx)) for i in h_idxs)
    return bonds, disconnection_rdmol


def rdchiral_extract_template(
    reaction_string: str, inp_fmt: str, reaction_id: Union[int, None] = None
):
    if inp_fmt != "smiles":
        raise NotImplementedError
    mapped_smiles_split = reaction_string.split(">")
    rdchiral_input = {
        "_id": reaction_id,
        "reactants": mapped_smiles_split[0],
        "agents": mapped_smiles_split[1],
        "products": mapped_smiles_split[2],
    }
    return template_extractor.extract_from_reaction(reaction=rdchiral_input)


def inject_atom_mapping(mol, full_map_info: dict, role: str) -> list:
    """
    To inject the atom mapping information in an unmapped Mol instance

    Parameters:
    -----------
    mol: Molecule
        The Molecule instance of interest
    full_map_info: dict
        The dictionary containing the information regarding the atoms' map numbers
    role: str
        The role of the molecule (reactants, reagents, products)

    Returns:
    --------
    mapped_rdmols: list
        The list of mapped rdkit's Mol objects

    """
    mol_mappings = full_map_info[role].get(mol.uid)
    mapped_rdmols = []
    for map_dict in mol_mappings:
        rdmol_new = copy.deepcopy(mol.rdmol_mapped)
        for i, map_num in map_dict.items():
            atom = next(a for a in rdmol_new.GetAtoms() if a.GetIdx() == i)
            atom.SetIntProp("molAtomMapNumber", map_num)
        mapped_rdmols.append(canonicalize_mapped_rdmol(rdmol_new))
    return mapped_rdmols


def add_reactant_to_rdrxn(rdmol: Mol, rdrxn: rdChemReactions.ChemicalReaction):
    """To add a reactant to a rdkit ChemicalReaction object"""
    rdrxn.AddReactantTemplate(rdmol)


def add_reagent_to_rdrxn(rdmol: Mol, rdrxn: rdChemReactions.ChemicalReaction):
    """To add a reagent to a rdkit ChemicalReaction object"""
    rdrxn.AddAgentTemplate(rdmol)


def add_product_to_rdrxn(rdmol: Mol, rdrxn: rdChemReactions.ChemicalReaction):
    """To add a product to a rdkit ChemicalReaction object"""
    rdrxn.AddProductTemplate(rdmol)


def build_rdrxn(
    catalog: dict,
    role_map: dict,
    stoichiometry_coefficients: dict,
    use_reagents: bool,
    use_smiles: bool,
    use_atom_mapping: bool,
    mapping=None,
) -> rdChemReactions.ChemicalReaction:
    """
    To build a rdkit Reaction object using the attributes of a ChemicalEquation

    Parameters:
    ------------
    catalog: dict
        The dictionary mapping the uid of the Molecules involved in the reaction with the relative Molecule instances
    role_map: dict
        The dictionary mapping each role with a list of Molecules' uid
    stoichiometry_coefficients: dict
        The nested dictionary that, for each role, maps the Molecules' uid with their stoichiometric coefficient
    use_reagents: bool
        Whether the reagents should be considered while building the rdrxn object
    use_smiles: bool
        Whether the smiles should be used
    use_atom_mapping: bool
        Whether the atom mapping should be conserved in the final rdrxn
    mapping: Optional[Ratam, None]
        The Ratam instance with the mapping information of the reaction; if it is not provided, it is assumed
        that the reaction is not mapped (default None)

    Returns:
    --------
    rdrxn_new: ChemicalReaction
        The rdkit ChemicalReaction object
    """
    rdkit_functions_map = {
        "reactants": add_reactant_to_rdrxn,
        "reagents": add_reagent_to_rdrxn,
        "products": add_product_to_rdrxn,
    }
    # an empty rdkit Reaction is initialized
    reaction_smiles_empty = ">>"
    rdrxn_new = rdChemReactions.ReactionFromSmarts(
        reaction_smiles_empty, useSmiles=use_smiles
    )

    # the rdkit Reaction is populated with the molecules based on their role and their stoichiometry
    for role, func in rdkit_functions_map.items():
        # if reagents should be ignored, this role is skipped and no molecules are added
        if role == "reagents" and not use_reagents:
            continue
        for mol_id in role_map.get(role):
            m = catalog.get(mol_id)
            if mapping:
                populate_rdrxn_with_mapping(m, rdrxn_new, mapping, role, func)

            else:
                populate_rdrxn_with_stoichiometry(
                    m, rdrxn_new, role, stoichiometry_coefficients, func
                )

    if not use_atom_mapping:
        Chem.rdChemReactions.RemoveMappingNumbersFromReactions(rdrxn_new)
    return rdrxn_new


def populate_rdrxn_with_mapping(molecule, rdrxn, mapping, role, rdkit_function):
    """To add mapped molecules to a rdkit ChemicalReaction object with a specific role"""
    mapped_rdmols = inject_atom_mapping(molecule, mapping.full_map_info, role)
    [rdkit_function(rdmol, rdrxn) for rdmol in mapped_rdmols]


def populate_rdrxn_with_stoichiometry(
    molecule, rdrxn, role, stoichiometry_coefficients, rdkit_function
):
    """To add molecules to a rdkit ChemReaction object with a specific role"""
    for _ in range(stoichiometry_coefficients.get(role).get(molecule.uid)):
        rdkit_function(molecule.rdmol, rdrxn)


def canonicalize_rdrxn(
    rdrxn: rdChemReactions.ChemicalReaction,
) -> rdChemReactions.ChemicalReaction:
    # canonicalize a reaction passing through a smiles
    # canonicalize the order of atoms in each molecule
    # canonicalize the order of molecules in the reaction
    rdrxn = rdChemReactions.ReactionFromSmarts(
        rdChemReactions.ReactionToSmiles(rdrxn, canonical=True), useSmiles=True
    )
    return rdrxn


def activate_rdrxn(
    rdrxn: rdChemReactions.ChemicalReaction,
) -> Tuple[rdChemReactions.ChemicalReaction, dict]:
    # TODO: the whole error handling logic can be better
    # http://www.rdkit.org/Python_Docs/rdkit.Chem.SimpleEnum.Enumerator-module.html#PreprocessReaction
    # http://www.rdkit.org/Python_Docs/rdkit.Chem.rdChemReactions-module.html
    exception: Union[str, Exception]
    try:
        rdChemReactions.SanitizeRxn(rdrxn)
        rdrxn.Initialize()
        (
            nWarn,
            nError,
            nReacts,
            nProds,
            reactantLabels,
        ) = rdChemReactions.PreprocessReaction(rdrxn)
        rdrxn_log = {
            "n_warn": nWarn,
            "n_error": nError,
            "n_reactants": nReacts,
            "n_products": nProds,
            "reactant_labels": reactantLabels,
        }
        is_successful = True
        if nError:
            rdrxn = None
            is_successful = False
        exception = ""

    except Exception as e:
        exception = e
        rdrxn = None
        rdrxn_log = {}
        is_successful = False

    log = {
        "is_successful": is_successful,
        "exception": exception,
        "rdrxn_log": rdrxn_log,
    }
    return rdrxn, log


####################################################################################################
# Descriptors
def calculate_atom_oxidation_number_by_EN(atom: Atom) -> int:
    pauling_en_map = {
        1: 2.2,
        3: 0.98,
        4: 1.57,
        5: 2.04,
        6: 2.55,
        7: 3.04,
        8: 3.44,
        9: 3.98,
        11: 0.93,
        12: 1.31,
        13: 1.61,
        14: 1.9,
        15: 2.19,
        16: 2.58,
        17: 3.16,
        19: 0.82,
        20: 1,
        21: 1.36,
        22: 1.54,
        23: 1.63,
        24: 1.66,
        25: 1.55,
        26: 1.83,
        27: 1.88,
        28: 1.91,
        29: 1.9,
        30: 1.65,
        31: 1.81,
        32: 2.01,
        33: 2.18,
        34: 2.55,
        35: 2.96,
        36: 3,
        37: 0.82,
        38: 0.95,
        39: 1.22,
        40: 1.33,
        41: 1.6,
        42: 2.16,
        43: 1.9,
        44: 2.2,
        45: 2.28,
        46: 2.2,
        47: 1.93,
        48: 1.69,
        49: 1.78,
        50: 1.96,
        51: 2.05,
        52: 2.1,
        53: 2.66,
        54: 2.6,
        55: 0.79,
        56: 0.89,
        57: 1.1,
        58: 1.12,
        59: 1.13,
        60: 1.14,
        62: 1.17,
        64: 1.2,
        66: 1.22,
        67: 1.23,
        68: 1.24,
        69: 1.25,
        71: 1.27,
        72: 1.3,
        73: 1.5,
        74: 2.36,
        75: 1.9,
        76: 2.2,
        77: 2.2,
        78: 2.28,
        79: 2.54,
        80: 2,
        81: 1.62,
        82: 2.33,
        83: 2.02,
        84: 2,
        85: 2.2,
        88: 0.9,
        89: 1.1,
        90: 1.3,
        91: 1.5,
        92: 1.38,
        93: 1.36,
        94: 1.28,
        95: 1.3,
        96: 1.3,
        97: 1.3,
        98: 1.3,
        99: 1.3,
        100: 1.3,
        101: 1.3,
        102: 1.39,
    }
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

    neighbors_nonHs_contribution = [
        sf(
            en_map.get(a_atomic_number)
            - en_map.get(bond.GetOtherAtom(atom).GetAtomicNum())
        )
        * bond.GetBondTypeAsDouble()
        for bond in bonds_neighbor
        if bond.GetOtherAtom(atom).GetAtomicNum() != 1
    ]

    neighbors_Hs_contribution = [
        sf(en_map.get(a_atomic_number) - en_map.get(1))
        for x in range(atom.GetTotalNumHs())
    ]

    oxidation_number = int(
        atom.GetFormalCharge()
        + sum(neighbors_nonHs_contribution)
        + sum(neighbors_Hs_contribution)
    )

    return oxidation_number


def compute_oxidation_numbers(rdmol: Mol) -> Mol:
    """
    To compute the oxidation number of the atoms in a molecule

    Parameters:
    ------------
    rdmol: Mol
        The rdkit Mol object for whose atoms the oxidation numbers should be computed

    Returns:
    --------
    rdmol with oxidation number: Mol
        The input Mol object whose atoms are annotated with the OxidationNumber property

    """
    rdmol_ = copy.deepcopy(rdmol)
    Chem.rdmolops.Kekulize(rdmol_)
    atoms = rdmol.GetAtoms()
    atoms_ = rdmol_.GetAtoms()
    for atom, atom_ in zip(atoms, atoms_):
        oxidation_number = calculate_atom_oxidation_number_by_EN(atom_)
        atom.SetIntProp("_OxidationNumber", oxidation_number)
    return rdmol


def smiles_to_anonymus_graph(r):
    r1 = re.sub(r"[a-zA-Z]", r"*", r)  # replace all atoms with * wildcard
    for bt in ["-", ":", "=", "#", "/", "\\"]:  # remove bond information
        r1 = r1.replace(bt, "")
    return r1


def smiles_to_element_graph(r):
    r1 = r.upper()  # remove aromatic info from atoms
    for bt in ["-", ":", "=", "#", "/", "\\"]:  # remove bond information
        r1 = r1.replace(bt, "")
    return r1


if __name__ == "__main__":
    print("main mode")
