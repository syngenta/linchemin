import abc
import copy
import re
from collections import namedtuple
from functools import partial
from typing import Callable, Dict, List

import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import (
    DataStructs,
    Draw,
    rdchem,
    rdChemReactions,
    rdFingerprintGenerator,
)
from rdkit.Chem.Draw import DrawingOptions, rdMolDraw2D
from rdkit.Chem.rdchem import Atom, Mol
from rdkit.Chem.rdMolHash import HashFunction, MolHash

RDLogger.DisableLog("rdApp.*")


# RDMOLECULE
def rdmol_from_string(input_string: str, inp_fmt: str):
    function_map = {"smiles": Chem.MolFromSmiles, "smarts": Chem.MolFromSmarts}
    func = function_map.get(inp_fmt)
    return func(input_string)


def remove_rdmol_atom_mapping(rdmol: Mol) -> Mol:
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
    task_name = "canonicalize_rdmol"
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
    rdmol: rdkit.Chem.rdchem.Mol, is_pattern: bool = False
) -> rdkit.Chem.rdchem.Mol:
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
    inchi = Chem.MolToInchi(rdmol, options="-KET -15T")
    return Chem.InchiToInchiKey(inchi)


def get_mol_property_function(property_name: str) -> MolPropertyFunction:
    mol_property_function_factory = {
        "smiles": compute_mol_smiles,
        "inchi_key": compute_mol_inchi_key,
    }
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
    rdmol = rdmol_from_string(smiles, inp_fmt="smiles")

    DrawingOptions.atomLabelFontSize = 55
    DrawingOptions.dotsPerAngstrom = 100
    DrawingOptions.bondLineWidth = 3.0
    Draw.MolToFile(rdmol, filename)


####################################################################################################
# Reaction / Chemical Equation
# Reaction fingerprint factory


class ReactionFingerprint(metaclass=abc.ABCMeta):
    """Definition of the abstract class for reaction fingerprints"""

    @abc.abstractmethod
    def compute_reac_fingerprint(self, rdrxn, params):
        pass


class DiffReactionFingerprint(ReactionFingerprint):
    def compute_reac_fingerprint(self, rdrxn, params):
        # Setting the parameters of the reaction fingerprint; if they are not specified, the default parameters as
        # specified in the link below are used:
        # https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/ChemReactions/ReactionFingerprints.cpp#L123
        if params is None:
            params = {}
        fp_params = rdChemReactions.ReactionFingerprintParams()
        fp_params.includeAgents = params.get("includeAgents", True)
        fp_params.fpSize = params.get("fpSize", 2048)
        fp_params.nonAgentWeight = params.get("nonAgentWeight", 10)
        fp_params.agentWeight = params.get("agentWeight", 1)
        fp_params.bitRatioAgents = params.get("bitRatioAgents", 0.0)
        fp_params.fpType = params.get(
            "fpType", rdChemReactions.FingerprintType.AtomPairFP
        )

        return rdChemReactions.CreateDifferenceFingerprintForReaction(
            rdrxn, ReactionFingerPrintParams=fp_params
        )


class StructReactionFingerprint(ReactionFingerprint):
    def compute_reac_fingerprint(self, rdrxn, params):
        # Setting the parameters of the reaction fingerprint; if they are not specified, the default parameters as
        # specified in the link below are used:
        # https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/ChemReactions/ReactionFingerprints.cpp#L123
        if params is None:
            params = {}
        fp_params = rdChemReactions.ReactionFingerprintParams()
        fp_params.includeAgents = params.get("includeAgents", True)
        fp_params.fpSize = params.get("fpSize", 4096)
        fp_params.nonAgentWeight = params.get("nonAgentWeight", 1)
        fp_params.agentWeight = params.get("agentWeight", 1)
        fp_params.bitRatioAgents = params.get("bitRatioAgents", 0.2)
        fp_params.fpType = params.get(
            "fpType", rdChemReactions.FingerprintType.PatternFP
        )

        return rdChemReactions.CreateStructuralFingerprintForReaction(
            rdrxn, ReactionFingerPrintParams=fp_params
        )


def compute_reaction_fingerprint(rdrxn, fp_name: str, params=None):
    """To compute the reaction fingerprint with the selected method and the optional parameters.

    Parameters:
        rdrxn: an rdkit rdChemReactions.ChemicalReaction instance
        fp_name: a string representing the selected type of fingerprint to be used
        params: an optional dictionary to change the default parameters

    Returns:
        fp: the fingerprint of the reaction
    """
    # control the behavior of fingerprint generation via the parameters
    #  https://www.rdkit.org/docs/source/rdkit.Chem.rdChemReactions.html#rdkit.Chem.rdChemReactions.ReactionFingerprintParams

    fingerprint_map = {
        "structure_fp": StructReactionFingerprint(),
        "difference_fp": DiffReactionFingerprint(),
    }

    if fp_name in fingerprint_map:
        return fingerprint_map.get(fp_name).compute_reac_fingerprint(rdrxn, params)
    else:
        raise KeyError(
            f"Invalid fingerprint type: {fp_name} is not available.\nAvailable options are: {fingerprint_map.keys()}"
        )


def rdrxn_from_string(
    input_string: str, inp_fmt: str
) -> rdChemReactions.ChemicalReaction:
    format_function_map = {
        "smiles": partial(Chem.rdChemReactions.ReactionFromSmarts, useSmiles=True),
        "smarts": partial(Chem.rdChemReactions.ReactionFromSmarts, useSmiles=False),
        "rxn": Chem.rdChemReactions.ReactionFromRxnBlock,
    }

    func = format_function_map.get(inp_fmt)
    return func(input_string)


def rdrxn_to_string(
    rdrxn: rdChemReactions.ChemicalReaction,
    out_fmt: str,
    use_atom_mapping: bool = False,
) -> str:
    if not use_atom_mapping:
        Chem.rdChemReactions.RemoveMappingNumbersFromReactions(rdrxn)
    function_map = {
        "smiles": partial(Chem.rdChemReactions.ReactionToSmiles, canonical=True),
        "smarts": Chem.rdChemReactions.ReactionToSmarts,
        # 'rxn': partial(Chem.rdChemReactions.ReactionToRxnBlock, forceV3000=True),
        "rxn": Chem.rdChemReactions.ReactionToRxnBlock,
        # 'rxn': Chem.rdChemReactions.ReactionToV3KRxnBlock,
    }
    func = function_map.get(out_fmt)
    return func(rdrxn)


def rdrxn_to_rxn_mol_catalog(
    rdrxn: rdChemReactions.ChemicalReaction,
) -> dict[str, list[Mol]]:
    return {
        "reactants": list(rdrxn.GetReactants()),
        "reagents": list(rdrxn.GetAgents()),
        "products": list(rdrxn.GetProducts()),
    }


def rdrxn_from_rxn_mol_catalog(
    rxn_mol_catalog: dict[str, list[Mol]]
) -> rdChemReactions.ChemicalReaction:
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


def unpack_rdrxn(
    rdrxn: rdChemReactions.ChemicalReaction, identity_property_name: str, constructor
):
    reaction_rdmols = rdrxn_to_rxn_mol_catalog(rdrxn=rdrxn)

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
    return catalog, role_map, stoichiometry_coefficients


def build_rdrxn(
    catalog: dict,
    role_map: dict,
    stoichiometry_coefficients: dict,
    use_smiles: bool,
    use_atom_mapping: bool,
) -> rdChemReactions.ChemicalReaction:
    reaction_smiles_empty = ">>"
    rdrxn_new = rdChemReactions.ReactionFromSmarts(
        reaction_smiles_empty, useSmiles=use_smiles
    )
    for mol_id in role_map.get("reactants"):
        if rdmol_mapped := catalog.get(mol_id).rdmol_mapped:
            for _ in range(stoichiometry_coefficients.get("reactants").get(mol_id)):
                rdrxn_new.AddReactantTemplate(rdmol_mapped)
    for mol_id in role_map.get("reagents"):
        if rdmol_mapped := catalog.get(mol_id).rdmol_mapped:
            for _ in range(stoichiometry_coefficients.get("reagents").get(mol_id)):
                rdrxn_new.AddAgentTemplate(rdmol_mapped)
    for mol_id in role_map.get("products"):
        if rdmol_mapped := catalog.get(mol_id).rdmol_mapped:
            for _ in range(stoichiometry_coefficients.get("products").get(mol_id)):
                rdrxn_new.AddProductTemplate(rdmol_mapped)
    if not use_atom_mapping:
        Chem.rdChemReactions.RemoveMappingNumbersFromReactions(rdrxn_new)
    return rdrxn_new


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
) -> (rdChemReactions.ChemicalReaction, dict):
    # TODO: the whole error handling logic can be better
    # http://www.rdkit.org/Python_Docs/rdkit.Chem.SimpleEnum.Enumerator-module.html#PreprocessReaction
    # http://www.rdkit.org/Python_Docs/rdkit.Chem.rdChemReactions-module.html

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


def rdrxn_role_reassignment(
    rdrxn: rdChemReactions.ChemicalReaction, desired_product_idx: int = 0
) -> rdChemReactions.ChemicalReaction:
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
        x = keys_input.intersection(keys_output) - {0}
        rdmol = input_item.get("rdmol")
        if len(x) == 0:
            rxn_mol_catalog.get("reagents").append(rdmol)
        else:
            rxn_mol_catalog.get("reactants").append(rdmol)
    return rdrxn_from_rxn_mol_catalog(rxn_mol_catalog=rxn_mol_catalog)


def compute_similarity(fp1, fp2, similarity_name: str) -> float:
    """
    Computes the chemical similarity between the input pair of fingerprints using the selected similarity algorithm

    Parameters:
        fp1, fp2: pair of molecular or reaction fingerprints
        similarity_name: string indicating the selected similarity algorithm

    Returns:
        a float, output of the similarity algorithm
    """
    similarity_map = {
        "tanimoto": DataStructs.TanimotoSimilarity,
        "kulczynski": DataStructs.KulczynskiSimilarity,
        "dice": DataStructs.DiceSimilarity,
        "mcconnaughey": DataStructs.McConnaugheySimilarity,
    }

    metric = similarity_map.get(similarity_name)
    # The syntax below is not compatible with count vectors fingerprints generated by GetCountFingerprint and with the
    # difference fingerprints for reactions
    # similarity = DataStructs.FingerprintSimilarity(fp1, fp2, metric=metric)
    return metric(fp1, fp2)


# Molecular fingerprints factory
class MolFingerprint(metaclass=abc.ABCMeta):
    """Definition of the abstract class for molecular fingerprints"""

    @abc.abstractmethod
    def compute_molecular_fingerprint(self, rdmol: Mol, parameters, count_fp_vector):
        pass


class RDKitMolFingerprint(MolFingerprint):
    def compute_molecular_fingerprint(self, rdmol: Mol, params, count_fp_vector):
        if params is None:
            params = {}
        fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(
            minPath=params.get("minPath", 1),
            maxPath=params.get("maxPath", 7),
            countSimulation=params.get("countSimulation", False),
            numBitsPerFeature=params.get("numBitsPerFeature", 2),
            fpSize=params.get("fpSize", 2048),
        )
        fp_builder = select_fp_vector(fpgen, count_fp_vector)
        return fp_builder(rdmol)


class MorganMolFingerprint(MolFingerprint):
    def compute_molecular_fingerprint(self, rdmol: Mol, params, count_fp_vector):
        if params is None:
            params = {}
        fpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=params.get("radius", 3),
            useCountSimulation=params.get("countSimulation", False),
            includeChirality=params.get("includeChirality", False),
            useBondTypes=params.get("useBondTypes", True),
            countBounds=params.get("countBounds", None),
            fpSize=params.get("fpSize", 2048),
        )
        fp_builder = select_fp_vector(fpgen, count_fp_vector)
        return fp_builder(rdmol)


class TopologicalMolFingerprint(MolFingerprint):
    def compute_molecular_fingerprint(self, rdmol: Mol, params, count_fp_vector):
        if params is None:
            params = {}
        fpgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
            includeChirality=params.get("includeChirality", False),
            torsionAtomCount=params.get("torsionAtomCount", 4),
            countSimulation=params.get("countSimulation", True),
            countBounds=params.get("countBounds", None),
            fpSize=params.get("fpSize", 2048),
        )

        fp_builder = select_fp_vector(fpgen, count_fp_vector)
        return fp_builder(rdmol)


def compute_mol_fingerprint(
    rdmol: Mol, fp_name: str, parameters=None, count_fp_vector=False
):
    """Takes an rdmol object, the fingerprint type, an optional dictionary to change the default
    parameters and a boolean and returns the selected fingerprint of the given rdmol.

    Parameters:
        rdmol: a molecule as rdkit object
        fp_name: a string representing the name of the fingerprint generator
        parameters: an optional dictionary
        count_fp_vector: an option boolean indicating whether the 'GetCountFingerprint' should be used

    Returns:
        fp: the fingerprint of the molecule
    """
    fp_generators_map = {
        "rdkit": RDKitMolFingerprint(),
        "morgan": MorganMolFingerprint(),
        "topological": TopologicalMolFingerprint(),
    }
    if fp_name in fp_generators_map:
        return fp_generators_map.get(fp_name).compute_molecular_fingerprint(
            rdmol, parameters, count_fp_vector
        )

    else:
        raise KeyError(
            f"Invalid fingerprint type: {fp_name} is not available.\nAvailable options are: {fp_generators_map.keys()}"
        )


def select_fp_vector(fpgen, count_fp_vector):
    return fpgen.GetCountFingerprint if count_fp_vector else fpgen.GetFingerprint


def draw_reaction(smiles: str, filename: str):
    """
    Produces a file with the picture of a reaction using RDKit

    Parameters:
        smiles: string indicating the smiles of the reaction
        filename: string indicated the name of the output file

    Returns:
        None. The image file is produced and stored in the directory
    """
    rxn = rdrxn_from_string(smiles, inp_fmt="smiles")

    DrawingOptions.atomLabelFontSize = 55
    DrawingOptions.dotsPerAngstrom = 100
    DrawingOptions.bondLineWidth = 3.0
    rimage = Draw.ReactionToImage(rxn)
    rimage.save(filename)


###### descriptors
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
    rdmol_ = copy.deepcopy(rdmol)
    Chem.rdmolops.Kekulize(rdmol_)
    atoms = rdmol.GetAtoms()
    atoms_ = rdmol_.GetAtoms()
    for atom, atom_ in zip(atoms, atoms_):
        oxidation_number = calculate_atom_oxidation_number_by_EN(atom_)
        atom.SetIntProp("_OxidationNumber", oxidation_number)
    return rdmol


def calculate_molecular_hash_values(rdmol: Mol, hash_list: list[str] = None) -> dict:
    molhashf = HashFunction.names
    if hash_list:
        hash_list += ["CanonicalSmiles"]
    else:
        hash_list = list(molhashf.keys()) + [
            "inchi_key",
            "inchikey_KET_15T",
            "noiso_smiles",
            "cx_smiles",
        ]

    hash_map = {k: MolHash(rdmol, v) for k, v in molhashf.items() if k in hash_list}

    hash_map["smiles"] = hash_map.get("CanonicalSmiles")
    if "inchi_key" in hash_list:
        hash_map["inchi"] = Chem.MolToInchi(rdmol)
        hash_map["inchi_key"] = Chem.InchiToInchiKey(hash_map["inchi"])
    if "inchikey_KET_15T" in hash_list:
        hash_map["inchi_KET_15T"] = Chem.MolToInchi(rdmol, options="-KET -15T")
        hash_map["inchikey_KET_15T"] = Chem.InchiToInchiKey(hash_map["inchi_KET_15T"])
    if "noiso_smiles" in hash_list:
        hash_map["noiso_smiles"] = Chem.MolToSmiles(rdmol, isomericSmiles=False)
    if "cx_smiles" in hash_list:
        hash_map["cx_smiles"] = Chem.MolToCXSmiles(rdmol)
    """
    
    hash_map['ExtendedMurcko_AG'] = smiles_to_anonymus_graph(hash_map['ExtendedMurcko'])
    hash_map['ExtendedMurcko_EG'] = smiles_to_element_graph(hash_map['ExtendedMurcko'])
    hash_map['MurckoScaffold_AG'] = smiles_to_anonymus_graph(hash_map['MurckoScaffold'])
    hash_map['MurckoScaffold_EG'] = smiles_to_element_graph(hash_map['MurckoScaffold'])
    """

    return hash_map


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
