from abc import ABC, abstractmethod
from typing import Union

import linchemin.cheminfo.functions as cif
from linchemin import settings
from linchemin.utilities import console_logger

"""
Module containing functions and classes for computing chemical fingerprints and similarity
"""
logger = console_logger(__name__)


def generate_rdkit_fp(params: dict):
    return cif.rdFingerprintGenerator.GetRDKitFPGenerator(
        minPath=params.get("minPath", 1),
        maxPath=params.get("maxPath", 7),
        countSimulation=params.get("countSimulation", False),
        numBitsPerFeature=params.get("numBitsPerFeature", 2),
        fpSize=params.get("fpSize", 2048),
    )


def generate_morgan_fp(params: dict):
    return cif.rdFingerprintGenerator.GetMorganGenerator(
        radius=params.get("radius", 3),
        countSimulation=params.get("countSimulation", False),
        includeChirality=params.get("includeChirality", False),
        useBondTypes=params.get("useBondTypes", True),
        countBounds=params.get("countBounds", None),
        fpSize=params.get("fpSize", 2048),
        onlyNonzeroInvariants=params.get("onlyNonzeroInvariants", False),
        includeRingMembership=params.get("includeRingMembership", True),
        atomInvariantsGenerator=params.get("atomInvariantsGenerator", None),
        bondInvariantsGenerator=params.get("bondInvariantsGenerator", None),
    )


def generate_topological_fp(params: dict):
    return cif.rdFingerprintGenerator.GetTopologicalTorsionGenerator(
        includeChirality=params.get("includeChirality", False),
        torsionAtomCount=params.get("torsionAtomCount", 4),
        countSimulation=params.get("countSimulation", True),
        countBounds=params.get("countBounds", None),
        fpSize=params.get("fpSize", 2048),
    )


# Reaction fingerprint factory


class ReactionFingerprint(ABC):
    """Definition of the abstract class for reaction fingerprints"""

    @abstractmethod
    def compute_reac_fingerprint(
        self, rdrxn: cif.rdChemReactions.ChemicalReaction, params: dict
    ) -> Union[
        cif.DataStructs.cDataStructs.UIntSparseIntVect,
        cif.DataStructs.cDataStructs.ExplicitBitVect,
    ]:
        pass


class DiffReactionFingerprint(ReactionFingerprint):
    def compute_reac_fingerprint(
        self, rdrxn: cif.rdChemReactions.ChemicalReaction, params: dict
    ) -> cif.DataStructs.cDataStructs.UIntSparseIntVect:
        # Setting the parameters of the reaction fingerprint; if they are not specified, the default parameters as
        # specified in the link below are used:
        # https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/ChemReactions/ReactionFingerprints.cpp#L123
        if params is None:
            params = {}
        fp_params = cif.rdChemReactions.ReactionFingerprintParams()
        fp_params.includeAgents = params.get(
            "includeAgents", settings.CHEMICAL_SIMILARITY.includeAgents
        )
        fp_params.fpSize = params.get(
            "fpSize", settings.CHEMICAL_SIMILARITY.diff_fp_fpSize
        )
        fp_params.nonAgentWeight = params.get(
            "nonAgentWeight", settings.CHEMICAL_SIMILARITY.diff_fp_nonAgentWeight
        )
        fp_params.agentWeight = params.get(
            "agentWeight", settings.CHEMICAL_SIMILARITY.agentWeight
        )
        fp_params.bitRatioAgents = params.get(
            "bitRatioAgents", settings.CHEMICAL_SIMILARITY.diff_fp_bitRatioAgents
        )
        fp_params.fpType = params.get(
            "fpType", cif.rdChemReactions.FingerprintType.AtomPairFP
        )

        return cif.rdChemReactions.CreateDifferenceFingerprintForReaction(
            rdrxn, ReactionFingerPrintParams=fp_params
        )


class StructReactionFingerprint(ReactionFingerprint):
    def compute_reac_fingerprint(
        self, rdrxn: cif.rdChemReactions.ChemicalReaction, params: dict
    ) -> cif.DataStructs.cDataStructs.ExplicitBitVect:
        # Setting the parameters of the reaction fingerprint; if they are not specified, the default parameters as
        # specified in the link below are used:
        # https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/ChemReactions/ReactionFingerprints.cpp#L123
        if params is None:
            params = {}
        fp_params = cif.rdChemReactions.ReactionFingerprintParams()
        fp_params.includeAgents = params.get(
            "includeAgents", settings.CHEMICAL_SIMILARITY.includeAgents
        )
        fp_params.fpSize = params.get(
            "fpSize", settings.CHEMICAL_SIMILARITY.struct_fp_fpSize
        )
        fp_params.nonAgentWeight = params.get(
            "nonAgentWeight", settings.CHEMICAL_SIMILARITY.struct_fp_nonAgentWeight
        )
        fp_params.agentWeight = params.get(
            "agentWeight", settings.CHEMICAL_SIMILARITY.agentWeight
        )
        fp_params.bitRatioAgents = params.get(
            "bitRatioAgents", settings.CHEMICAL_SIMILARITY.struct_fp_bitRatioAgents
        )
        fp_params.fpType = params.get(
            "fpType", cif.rdChemReactions.FingerprintType.PatternFP
        )

        return cif.rdChemReactions.CreateStructuralFingerprintForReaction(
            rdrxn, ReactionFingerPrintParams=fp_params
        )


def compute_reaction_fingerprint(
    rdrxn: cif.rdChemReactions.ChemicalReaction,
    fp_name: str,
    params: Union[dict, None] = None,
) -> Union[
    cif.DataStructs.cDataStructs.UIntSparseIntVect,
    cif.DataStructs.cDataStructs.ExplicitBitVect,
]:
    """
    To compute the fingerprint of a chemical reaction.

    Parameters:
    -----------
    rdrxn: rdChemReactions.ChemicalReaction
        The rdkit ChemicalReaction instance for which the fingerprint should be computed
    fp_name: str
        The selected type of fingerprint to be used
    params: Optional[Union[dict, None]]
        The dictionary with fingerprint parameters to be used; if it is not provided, the default paremeters are used
        (default None).

    Returns:
    --------
    fp: the fingerprint of the reaction

    Raises:
    -------
    KeyError: if the selected fingerprint is not available

    Example:
    ---------
    >>> smiles = 'CN.CC(O)=O>C>CNC(C)=O'
    >>> rdrxn = cif.rdrxn_from_string(smiles, inp_fmt='smiles')
    >>> fp1 = compute_reaction_fingerprint(rdrxn, fp_name='structure_fp')
    """
    # control the behavior of fingerprint generation via the parameters
    #  https://www.rdkit.org/docs/source/rdkit.Chem.rdChemReactions.html#rdkit.Chem.rdChemReactions.ReactionFingerprintParams

    fingerprint_map = {
        "structure_fp": StructReactionFingerprint(),
        "difference_fp": DiffReactionFingerprint(),
    }

    if fp_name in fingerprint_map:
        return fingerprint_map.get(fp_name).compute_reac_fingerprint(rdrxn, params)
    logger.error(
        f"Invalid fingerprint type: {fp_name} is not available."
        f"Available options are: {fingerprint_map.keys()}"
    )
    raise KeyError


###########################################################################################
# Molecular fingerprints factory
class MolFingerprint(ABC):
    """Definition of the abstract class for molecular fingerprints"""

    @abstractmethod
    def compute_molecular_fingerprint(
        self, rdmol: cif.Mol, parameters: Union[dict, None], count_fp_vector: bool
    ) -> cif.DataStructs.cDataStructs.ExplicitBitVect:
        pass


class RDKitMolFingerprint(MolFingerprint):
    def compute_molecular_fingerprint(
        self, rdmol: cif.Mol, params: Union[dict, None], count_fp_vector: bool
    ) -> cif.DataStructs.cDataStructs.ExplicitBitVect:
        if params is None:
            params = {}
        fpgen = generate_rdkit_fp(params)
        fp_builder = select_fp_vector(fpgen, count_fp_vector)
        return fp_builder(rdmol)


class MorganMolFingerprint(MolFingerprint):
    def compute_molecular_fingerprint(
        self, rdmol: cif.Mol, params: Union[dict, None], count_fp_vector: bool
    ) -> cif.DataStructs.cDataStructs.ExplicitBitVect:
        if params is None:
            params = {}
        fpgen = generate_morgan_fp(params)
        fp_builder = select_fp_vector(fpgen, count_fp_vector)
        return fp_builder(rdmol)


class TopologicalMolFingerprint(MolFingerprint):
    def compute_molecular_fingerprint(
        self, rdmol: cif.Mol, params: Union[dict, None], count_fp_vector: bool
    ) -> cif.DataStructs.cDataStructs.ExplicitBitVect:
        if params is None:
            params = {}
        fpgen = generate_topological_fp(params)
        fp_builder = select_fp_vector(fpgen, count_fp_vector)
        return fp_builder(rdmol)


def compute_mol_fingerprint(
    rdmol: cif.Mol,
    fp_name: str,
    parameters: Union[dict, None] = None,
    count_fp_vector: bool = False,
) -> cif.DataStructs.cDataStructs.ExplicitBitVect:
    """
    To compute the fingerprint of a molecule.

    Parameters:
    ------------
    rdmol: Mol
        The rdkit Mol object for which the fingerprint should be computed
    fp_name: str
        The name of the fingerprint generator
    parameters: Optional[Union[dict, None]]
        The dictionary with fingerprint parameters to be used; if it is not provided, the default paremeters are used
        (default None).
    count_fp_vector: bool
        Whether the 'GetCountFingerprint' should be used (default False)

    Returns:
    --------
    fp: the fingerprint of the molecule

    Raises:
    --------
    KeyError: if the selected fingerprit is not available

    Example:
    ---------
    >>> smiles = 'CNC(C)=O'
    >>> rdmol = cif.rdmol_from_string(smiles, inp_fmt='smiles')
    >>> parameters = {'fpSize': 1024, 'countSimulation': True}
    >>> fp = compute_mol_fingerprint(rdmol, 'rdkit', parameters=parameters)
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

    logger.error(
        f"Invalid fingerprint type: {fp_name} is not available."
        f"Available options are: {fp_generators_map.keys()}"
    )
    raise KeyError


def select_fp_vector(fpgen, count_fp_vector):
    return fpgen.GetCountFingerprint if count_fp_vector else fpgen.GetFingerprint


# Chemical similarity calculation
def compute_similarity(
    fp1: Union[
        cif.DataStructs.cDataStructs.UIntSparseIntVect,
        cif.DataStructs.cDataStructs.ExplicitBitVect,
    ],
    fp2: Union[
        cif.DataStructs.cDataStructs.UIntSparseIntVect,
        cif.DataStructs.cDataStructs.ExplicitBitVect,
    ],
    similarity_name: str,
) -> float:
    """
    To compute the chemical similarity between the input pair of fingerprints using the selected similarity algorithm

    Parameters:
    -------------
    fp1: Union[cif.DataStructs.cDataStructs.UIntSparseIntVect, cif.DataStructs.cDataStructs.ExplicitBitVect]
        The fingerprint of the first chemical object (molecule or chemical reaction)
    fp2: Union[cif.DataStructs.cDataStructs.UIntSparseIntVect, cif.DataStructs.cDataStructs.ExplicitBitVect]
        The fingerprint of the second chemical object (molecule or chemical reaction)
    similarity_name: str
        The similarity algorithm

    Returns:
    ---------
    similarity: float
        The similarity between the two objects

    Raises:
    --------
    KeyError: if the selected similarity algorithm is not available

    Example:
    --------
    >>> smiles1 = 'CN.CC(O)=O>O>CNC(C)=O'
    >>> smiles2 = 'CN.CC(O)=O>C>CNC(C)=O'
    >>> rdrxn1 = cif.rdrxn_from_string(smiles1, inp_fmt='smiles')
    >>> rdrxn2 = cif.rdrxn_from_string(smiles2, inp_fmt='smiles')
    >>> fp1 = compute_reaction_fingerprint(rdrxn1, fp_name='structure_fp')
    >>> fp2 = compute_reaction_fingerprint(rdrxn2, fp_name='structure_fp')
    >>> similarity = compute_similarity(fp1, fp2, similarity_name='tanimoto')
    """
    similarity_map = {
        "tanimoto": cif.DataStructs.TanimotoSimilarity,
        "kulczynski": cif.DataStructs.KulczynskiSimilarity,
        "dice": cif.DataStructs.DiceSimilarity,
        "mcconnaughey": cif.DataStructs.McConnaugheySimilarity,
    }

    if similarity_name in similarity_map:
        metric = similarity_map.get(similarity_name)
    else:
        logger.error(
            f"Invalid similiarity algorithm: {similarity_name} is not available."
            f"Available options are: {similarity_map.keys()}"
        )
        raise KeyError
    # The syntax below is not compatible with count vectors fingerprints generated by GetCountFingerprint and with the
    # difference fingerprints for reactions
    # similarity = DataStructs.FingerprintSimilarity(fp1, fp2, metric=metric)
    return metric(fp1, fp2)


if __name__ == "__main__":
    print("main")
