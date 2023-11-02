import pytest
from rdkit.Chem import DataStructs, rdChemReactions

import linchemin.cheminfo.functions as cif
from linchemin.cheminfo.chemical_similarity import (
    compute_mol_fingerprint,
    compute_reaction_fingerprint,
    compute_similarity,
)


def test_rdkit_reaction_fingerprints_basics_non_mapped_molecules():
    # https://www.rdkit.org/docs/cppapi/structRDKit_1_1ReactionFingerprintParams.html

    reactions = {
        0: {"smiles": "CN.CC(O)=O>O>CNC(C)=O"},  # A.B>C>D
        1: {"smiles": "CC(O)=O.CN>O>CNC(C)=O"},
        2: {"smiles": ">>CNC(C)=O"},
        3: {"smiles": "CC(O)=O.CN>>CNC(C)=O"},  #
        4: {"smiles": "CC(O)=O.CN>>"},
        5: {"smiles": "CN.CC(O)=O>>CNC(C)=O"},
        6: {"smiles": "CNC(C)=O>>CN.CC(O)=O"},  # D>>A.B
        7: {"smiles": "CN.CC(O)=O>>CNC(C)=O.O"},  # A.B>>D
        8: {"smiles": "CN.CC(O)=O>>O.CNC(C)=O"},  # A.B>>C.D
        9: {"smiles": "CNC(C)=O>O>CN.CC(O)=O"},  # D>C>A.B
        10: {"smiles": "CNC(C)=O>>CN.CC(O)=O"},
        11: {"smiles": "CN.CC(O)=O.O>>CNC(C)=O"},  # A.B.C>>D
        12: {"smiles": "CN.CC(O)=O>C>CNC(C)=O"},  # A.B>E>D
    }

    rfp_params = rdChemReactions.ReactionFingerprintParams()
    # print(rfp_params.fpType, rfp_params.fpSize, rfp_params.includeAgents, rfp_params.agentWeight,
    # rfp_params.nonAgentWeight, rfp_params.bitRatioAgents)

    default_parameter_values = {
        "fpType": "AtomPairFP",
        "fpSize": 2048,
        "includeAgents": False,
        "agentWeight": 1,
        "nonAgentWeight": 10,
        "bitRatioAgents": 0.2,
    }

    results = {}
    for k, v in reactions.items():
        rdrxn = rdChemReactions.ReactionFromSmarts(v.get("smiles"), useSmiles=True)

        results[k] = {
            "rdrxn": rdrxn,
            "difference_fp": rdChemReactions.CreateDifferenceFingerprintForReaction(
                rdrxn
            ),
            "structural_fp": rdChemReactions.CreateStructuralFingerprintForReaction(
                rdrxn
            ),
        }

    # the function is symmetric
    assert DataStructs.TanimotoSimilarity(
        results.get(0).get("difference_fp"), results.get(1).get("difference_fp")
    ) == DataStructs.TanimotoSimilarity(
        results.get(1).get("difference_fp"), results.get(0).get("difference_fp")
    )
    assert DataStructs.TanimotoSimilarity(
        results.get(0).get("structural_fp"), results.get(1).get("structural_fp")
    ) == DataStructs.TanimotoSimilarity(
        results.get(1).get("structural_fp"), results.get(0).get("structural_fp")
    )

    # similarity with oneself == 1 (it is a similarity not a distance!)
    assert (
        DataStructs.TanimotoSimilarity(
            results.get(0).get("difference_fp"), results.get(0).get("difference_fp")
        )
        == 1.0
    )
    assert (
        DataStructs.TanimotoSimilarity(
            results.get(0).get("structural_fp"), results.get(0).get("structural_fp")
        )
        == 1.0
    )

    # the order of reactants does not matter
    assert (
        DataStructs.TanimotoSimilarity(
            results.get(0).get("difference_fp"), results.get(1).get("difference_fp")
        )
        == 1.0
    )
    assert (
        DataStructs.TanimotoSimilarity(
            results.get(0).get("structural_fp"), results.get(1).get("structural_fp")
        )
        == 1.0
    )
    # the order of products does not matter
    assert (
        DataStructs.TanimotoSimilarity(
            results.get(7).get("difference_fp"), results.get(8).get("difference_fp")
        )
        == 1.0
    )
    assert (
        DataStructs.TanimotoSimilarity(
            results.get(7).get("structural_fp"), results.get(8).get("structural_fp")
        )
        == 1.0
    )

    # the fingerprint works also on partial chemical equations
    assert (
        DataStructs.TanimotoSimilarity(
            results.get(2).get("difference_fp"), results.get(4).get("difference_fp")
        )
        != 1.0
    )
    assert (
        DataStructs.TanimotoSimilarity(
            results.get(2).get("structural_fp"), results.get(4).get("structural_fp")
        )
        != 1.0
    )

    # the difference fingerprint is NOT sensitive the presence of reagents ('includeAgents': False)
    # A.B>C>D == A.B>>D
    assert (
        DataStructs.TanimotoSimilarity(
            results.get(0).get("difference_fp"), results.get(7).get("difference_fp")
        )
        == 1.0
    )

    # the structural fingerprint is sensitive the presence of reagents ('includeAgents': False)
    # A.B>C>D != A.B>>D
    assert (
        DataStructs.TanimotoSimilarity(
            results.get(0).get("structural_fp"), results.get(7).get("structural_fp")
        )
        != 1.0
    )

    # the difference fingerprint is NOT sensitive to the role of constituents (focus on reagent) ('includeAgents': False)
    # A.B>C>D == A.B.C>>D
    assert (
        DataStructs.TanimotoSimilarity(
            results.get(0).get("difference_fp"), results.get(11).get("difference_fp")
        )
        == 1.0
    )

    # the structural_fp fingerprint is sensitive to the role of constituents (focus on reagent) ('includeAgents': False)
    # A.B>C>D != A.B.C>>D
    assert (
        DataStructs.TanimotoSimilarity(
            results.get(0).get("structural_fp"), results.get(11).get("structural_fp")
        )
        != 1.0
    )

    # the structural_fp fingerprint is sensitive to the nature of constituents (focus on reagent) when ReactionFingerprintParams
    # is not initialized
    # A.B>C>D == A.B>E>D
    assert (
        DataStructs.TanimotoSimilarity(
            results.get(0).get("structural_fp"), results.get(12).get("structural_fp")
        )
        != 1.0
    )

    # the difference_fp fingerprint is NOT sensitive to the nature of constituents (focus on reagent) when ReactionFingerprintParams
    # is not initialized
    # A.B>C>D == A.B>E>D
    assert (
        DataStructs.TanimotoSimilarity(
            results.get(0).get("difference_fp"), results.get(12).get("difference_fp")
        )
        == 1.0
    )


def test_reaction_fp_default_params():
    smiles1 = "CN.CC(O)=O>O>CNC(C)=O"
    smiles2 = "CN.CC(O)=O>C>CNC(C)=O"
    rdrxn1 = cif.rdrxn_from_string(smiles1, inp_fmt="smiles")
    rdrxn2 = cif.rdrxn_from_string(smiles2, inp_fmt="smiles")
    # Check structural fingerprints
    fp1 = compute_reaction_fingerprint(rdrxn1, fp_name="structure_fp")
    fp2 = compute_reaction_fingerprint(rdrxn2, fp_name="structure_fp")
    factory_fp_similarity = compute_similarity(fp1, fp2, similarity_name="tanimoto")

    fp1_default = rdChemReactions.CreateStructuralFingerprintForReaction(rdrxn1)
    fp2_default = rdChemReactions.CreateStructuralFingerprintForReaction(rdrxn2)
    default_fp_similarity = DataStructs.TanimotoSimilarity(fp1_default, fp2_default)
    assert factory_fp_similarity == default_fp_similarity

    # Check difference fingerprints
    fp1_dif = compute_reaction_fingerprint(rdrxn1, fp_name="difference_fp")
    fp2_dif = compute_reaction_fingerprint(rdrxn2, fp_name="difference_fp")
    factory_fp_dif_similarity = compute_similarity(
        fp1_dif, fp2_dif, similarity_name="tanimoto"
    )

    fp1_dif_default = rdChemReactions.CreateDifferenceFingerprintForReaction(rdrxn1)
    fp2_dif_default = rdChemReactions.CreateDifferenceFingerprintForReaction(rdrxn2)
    default_fp_dif_similarity = DataStructs.TanimotoSimilarity(
        fp1_dif_default, fp2_dif_default
    )
    assert factory_fp_dif_similarity == default_fp_dif_similarity


def test_compute_reaction_fingerprint_factory():
    smiles = "CN.CC(O)=O>O>CNC(C)=O"
    rdrxn = cif.rdrxn_from_string(smiles, inp_fmt="smiles")

    # 'structure_fp' and 'difference_fp' generate different fingerprint for the same reaction smiles
    fp_struct = compute_reaction_fingerprint(rdrxn, "structure_fp")
    fp_dif = compute_reaction_fingerprint(rdrxn, "difference_fp")
    assert fp_struct != fp_dif

    # different fingerprints are generated is a different fingerprint type is specified
    fp_struct_param = compute_reaction_fingerprint(
        rdrxn,
        "structure_fp",
        params={"fpType": rdChemReactions.FingerprintType.MorganFP},
    )
    assert fp_struct != fp_struct_param

    with pytest.raises(Exception) as ke:
        compute_reaction_fingerprint(rdrxn, "wrong_fp")
    assert "KeyError" in str(ke.type)


def test_mol_fp_factory():
    smiles = "CNC(C)=O"
    rdmol = cif.rdmol_from_string(smiles, inp_fmt="smiles")
    # If an unavailable option for the fingerprint is given, an error is raised
    with pytest.raises(Exception) as ke:
        compute_mol_fingerprint(rdmol, "wrong_fp")
    assert "KeyError" in str(ke.type)

    # Changing the default parameters, a different fingerprint is generated for the same molecule
    fp_rdkit = compute_mol_fingerprint(rdmol, "rdkit")
    parameters = {"fpSize": 1024, "countSimulation": True}
    fp_rdkit_params = compute_mol_fingerprint(rdmol, "rdkit", parameters=parameters)
    assert fp_rdkit_params != fp_rdkit
