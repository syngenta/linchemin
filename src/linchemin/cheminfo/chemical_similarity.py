from abc import ABC, abstractmethod
import linchemin.cheminfo.functions as cif
from linchemin import settings

"""
Module containing functions and classes for computing chemical fingerprints and similarity
"""


# Reaction fingerprint factory

class ReactionFingerprint(ABC):
    """ Definition of the abstract class for reaction fingerprints """

    @abstractmethod
    def compute_reac_fingerprint(self, rdrxn, params):
        pass


class DiffReactionFingerprint(ReactionFingerprint):
    def compute_reac_fingerprint(self, rdrxn, params):
        # Setting the parameters of the reaction fingerprint; if they are not specified, the default parameters as
        # specified in the link below are used:
        # https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/ChemReactions/ReactionFingerprints.cpp#L123
        if params is None:
            params = {}
        fp_params = cif.rdChemReactions.ReactionFingerprintParams()
        fp_params.includeAgents = params.get('includeAgents', settings.CHEMICAL_SIMILARITY.includeAgents)
        fp_params.fpSize = params.get('fpSize', settings.CHEMICAL_SIMILARITY.diff_fp_fpSize)
        fp_params.nonAgentWeight = params.get('nonAgentWeight', settings.CHEMICAL_SIMILARITY.diff_fp_nonAgentWeight)
        fp_params.agentWeight = params.get('agentWeight', settings.CHEMICAL_SIMILARITY.agentWeight)
        fp_params.bitRatioAgents = params.get('bitRatioAgents', settings.CHEMICAL_SIMILARITY.diff_fp_bitRatioAgents)
        fp_params.fpType = params.get('fpType', cif.rdChemReactions.FingerprintType.AtomPairFP)

        return cif.rdChemReactions.CreateDifferenceFingerprintForReaction(rdrxn, ReactionFingerPrintParams=fp_params)


class StructReactionFingerprint(ReactionFingerprint):
    def compute_reac_fingerprint(self, rdrxn, params):
        # Setting the parameters of the reaction fingerprint; if they are not specified, the default parameters as
        # specified in the link below are used:
        # https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/ChemReactions/ReactionFingerprints.cpp#L123
        if params is None:
            params = {}
        fp_params = cif.rdChemReactions.ReactionFingerprintParams()
        fp_params.includeAgents = params.get('includeAgents', settings.CHEMICAL_SIMILARITY.includeAgents)
        fp_params.fpSize = params.get('fpSize', settings.CHEMICAL_SIMILARITY.struct_fp_fpSize)
        fp_params.nonAgentWeight = params.get('nonAgentWeight', settings.CHEMICAL_SIMILARITY.struct_fp_nonAgentWeight)
        fp_params.agentWeight = params.get('agentWeight', settings.CHEMICAL_SIMILARITY.agentWeight)
        fp_params.bitRatioAgents = params.get('bitRatioAgents', settings.CHEMICAL_SIMILARITY.struct_fp_bitRatioAgents)
        fp_params.fpType = params.get('fpType', cif.rdChemReactions.FingerprintType.PatternFP)

        return cif.rdChemReactions.CreateStructuralFingerprintForReaction(rdrxn, ReactionFingerPrintParams=fp_params)


def compute_reaction_fingerprint(rdrxn, fp_name: str, params=None):
    """ To compute the reaction fingerprint with the selected method and the optional parameters.

        :param:
            rdrxn: an rdkit rdChemReactions.ChemicalReaction instance

            fp_name: a string representing the selected type of fingerprint to be used

            params: an optional dictionary to change the default parameters

        :return:
            fp: the fingerprint of the reaction
    """
    # control the behavior of fingerprint generation via the parameters
    #  https://www.rdkit.org/docs/source/rdkit.Chem.rdChemReactions.html#rdkit.Chem.rdChemReactions.ReactionFingerprintParams

    fingerprint_map = {'structure_fp': StructReactionFingerprint(),
                       'difference_fp': DiffReactionFingerprint()
                       }

    if fp_name in fingerprint_map:
        return fingerprint_map.get(fp_name).compute_reac_fingerprint(rdrxn, params)
    else:
        raise KeyError(
            f'Invalid fingerprint type: {fp_name} is not available.\nAvailable options are: {fingerprint_map.keys()}')


###########################################################################################
# Molecular fingerprints factory
class MolFingerprint(ABC):
    """ Definition of the abstract class for molecular fingerprints """

    @abstractmethod
    def compute_molecular_fingerprint(self, rdmol, parameters, count_fp_vector):
        pass


class RDKitMolFingerprint(MolFingerprint):
    def compute_molecular_fingerprint(self, rdmol, params, count_fp_vector):
        if params is None:
            params = {}
        fpgen = cif.generate_rdkit_fp(params)
        fp_builder = select_fp_vector(fpgen, count_fp_vector)
        return fp_builder(rdmol)


class MorganMolFingerprint(MolFingerprint):
    def compute_molecular_fingerprint(self, rdmol, params, count_fp_vector):
        if params is None:
            params = {}
        fpgen = cif.generate_morgan_fp(params)
        fp_builder = select_fp_vector(fpgen, count_fp_vector)
        return fp_builder(rdmol)


class TopologicalMolFingerprint(MolFingerprint):
    def compute_molecular_fingerprint(self, rdmol, params, count_fp_vector):
        if params is None:
            params = {}
        fpgen = cif.generate_topological_fp(params)
        fp_builder = select_fp_vector(fpgen, count_fp_vector)
        return fp_builder(rdmol)


def compute_mol_fingerprint(rdmol, fp_name: str, parameters=None, count_fp_vector=False):
    """ Takes a rdmol object, the fingerprint type, an optional dictionary to change the default
        parameters and a boolean and returns the selected fingerprint of the given rdmol.

        :param:
            rdmol: a molecule as rdkit object

            fp_name: a string representing the name of the fingerprint generator

            parameters: an optional dictionary

            count_fp_vector: an option boolean indicating whether the 'GetCountFingerprint' should be used

        :return:
            fp: the fingerprint of the molecule
    """
    fp_generators_map = {'rdkit': RDKitMolFingerprint(),
                         'morgan': MorganMolFingerprint(),
                         'topological': TopologicalMolFingerprint()
                         }
    if fp_name in fp_generators_map:
        return fp_generators_map.get(fp_name).compute_molecular_fingerprint(rdmol, parameters, count_fp_vector)

    else:
        raise KeyError(
            f'Invalid fingerprint type: {fp_name} is not available.\nAvailable options are: {fp_generators_map.keys()}')


def select_fp_vector(fpgen, count_fp_vector):
    return fpgen.GetCountFingerprint if count_fp_vector else fpgen.GetFingerprint


# Chemical similarity calculation
def compute_similarity(fp1, fp2, similarity_name: str) -> float:
    """
    Computes the chemical similarity between the input pair of fingerprints using the selected similarity algorithm

    :param:
        fp1, fp2: pair of molecular or reaction fingerprints

        similarity_name: string indicating the selected similarity algorithm

    :return:
        a float, output of the similarity algorithm
    """
    similarity_map = {'tanimoto': cif.DataStructs.TanimotoSimilarity,
                      'kulczynski': cif.DataStructs.KulczynskiSimilarity,
                      'dice': cif.DataStructs.DiceSimilarity,
                      'mcconnaughey': cif.DataStructs.McConnaugheySimilarity}

    metric = similarity_map.get(similarity_name)
    # The syntax below is not compatible with count vectors fingerprints generated by GetCountFingerprint and with the
    # difference fingerprints for reactions
    # similarity = DataStructs.FingerprintSimilarity(fp1, fp2, metric=metric)
    return metric(fp1, fp2)


if __name__ == '__main__':
    print('main')
