import pytest
from rdkit import Chem
from rdkit.Chem import DataStructs, rdChemReactions

from linchemin.cheminfo.functions import (
    bstr_to_rdmol,
    calculate_molecular_hash_values,
    canonicalize_rdmol,
    compute_mol_fingerprint,
    compute_oxidation_numbers,
    compute_reaction_fingerprint,
    compute_similarity,
    draw_mol,
    get_canonical_order,
)
from linchemin.cheminfo.functions import rdkit as rdkit
from linchemin.cheminfo.functions import (
    rdmol_from_string,
    rdmol_to_bstr,
    rdrxn_from_string,
    rdrxn_role_reassignment,
    rdrxn_to_rxn_mol_catalog,
    rdrxn_to_string,
    remove_rdmol_atom_mapping,
)


def test_build_rdrxn_from_smiles():
    kwa = {"input_string": "CN.CC(O)=O>O>CNC(C)=O", "inp_fmt": "smiles"}
    rdrxn = rdrxn_from_string(**kwa)
    assert rdrxn is not None


def test_build_rdrxn_from_smarts():
    kwa = {"input_string": "[C:1]=[O,N:2]>>[C:1][*:2]", "inp_fmt": "smarts"}
    rdrxn = rdrxn_from_string(**kwa)
    assert rdrxn is not None


def test_build_rdrxn_from_rxnblock():
    rxnblockv2k = """$RXN

      RDKit

  3  1
$MOL

     RDKit          2D

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2990    0.7500    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
M  END
$MOL

     RDKit          2D

  4  3  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2990    0.7500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.5981   -0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.2990    2.2500    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  2  4  2  0
M  END
$MOL

     RDKit          2D

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$MOL

     RDKit          2D

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2990    0.7500    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    2.5981   -0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.8971    0.7500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.5981   -1.5000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  3  4  1  0
  3  5  2  0
M  END
        """
    kwa = {"input_string": rxnblockv2k, "inp_fmt": "rxn"}
    rdrxn = rdrxn_from_string(**kwa)
    assert rdrxn is not None


def test_build_smiles_from_rdrxn():
    kwa = {"input_string": "CN.CC(O)=O>O>CNC(C)=O", "inp_fmt": "smiles"}
    rdrxn = rdrxn_from_string(**kwa)

    kwa = {"rdrxn": rdrxn, "out_fmt": "smiles"}
    string = rdrxn_to_string(**kwa)
    assert string is not None


def test_build_rxnblock_from_rdrxn():
    inp = {"input_string": "CN.CC(O)=O>O>CNC(C)=O", "inp_fmt": "smiles"}
    rdrxn = rdrxn_from_string(**inp)

    kwa = {"rdrxn": rdrxn, "out_fmt": "rxn"}
    string = rdrxn_to_string(**kwa)
    assert string is not None


def test_build_smarts_from_rdrxn():
    kwa = {"input_string": "[C:1]=[O,N:2]>>[C:1][*:2]", "inp_fmt": "smarts"}
    rdrxn = rdrxn_from_string(**kwa)

    kwa = {"rdrxn": rdrxn, "out_fmt": "smarts"}
    string = rdrxn_to_string(**kwa)

    assert string is not None


def test_rdmol_serialization():
    # serialize a rdmol to inject it into a database
    # https://github.com/rdkit/rdkit/issues/1999
    # https://stackoverflow.com/questions/15659195/insert-byte-string-into-mongodb
    smiles_string = "c1ncncc1C(=O)[O-]"
    rdmol1 = Chem.MolFromSmiles(smiles_string)
    smiles_canonical1 = Chem.MolToSmiles(rdmol1)
    rdmol_bstr = rdmol_to_bstr(rdmol=rdmol1)
    rdmol2 = bstr_to_rdmol(rdmol_bstr=rdmol_bstr)
    smiles_canonical2 = Chem.MolToSmiles(rdmol2)
    assert smiles_canonical1 == smiles_canonical2


def test_canonicalize_rdmol():
    # m1 and m2 encode the same molecule (same atoms, equivalent connectivity) but the order is different
    # this is not reflected by while generating a smiles because by defualt canonical=True ensures
    # a canonical smiles irrespective of the ordering of the molecular atoms.

    m1 = rdkit.Chem.MolFromSmiles("c1([C@H](C)CC)cccc2ccccc12")
    m2 = rdkit.Chem.MolFromSmiles("c12ccccc1c(ccc2)[C@H](C)CC")

    assert rdkit.Chem.MolToSmiles(m1) == rdkit.Chem.MolToSmiles(m2)
    assert rdkit.Chem.MolToSmiles(m1, canonical=False) != rdkit.Chem.MolToSmiles(
        m2, canonical=False
    )
    m1_canonical_order = get_canonical_order(m1)
    m2_canonical_order = get_canonical_order(m2)
    assert m1_canonical_order != m2_canonical_order

    m1_can, log = canonicalize_rdmol(rdmol=m1)
    m2_can, log = canonicalize_rdmol(rdmol=m2)

    assert rdkit.Chem.MolToSmiles(m1_can) == rdkit.Chem.MolToSmiles(m2_can)
    assert rdkit.Chem.MolToSmiles(m1_can, canonical=False) == rdkit.Chem.MolToSmiles(
        m2_can, canonical=False
    )
    m1_can_canonical_order = get_canonical_order(m1_can)
    m2_can_canonical_order = get_canonical_order(m2_can)
    assert m1_can_canonical_order == m2_can_canonical_order


def test_unpack_rdrxn():
    print("IMPLEMENT: test_unpack_rdrxn")  # TODO


def test_build_rdrxn():
    print("IMPLEMENT: build_rdrxn")  # TODO


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

    rfp_params = rdkit.Chem.rdChemReactions.ReactionFingerprintParams()
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
        rdrxn = rdkit.Chem.rdChemReactions.ReactionFromSmarts(
            v.get("smiles"), useSmiles=True
        )

        results[k] = {
            "rdrxn": rdrxn,
            "difference_fp": rdkit.Chem.rdChemReactions.CreateDifferenceFingerprintForReaction(
                rdrxn
            ),
            "structural_fp": rdkit.Chem.rdChemReactions.CreateStructuralFingerprintForReaction(
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


def test_mol_fp_factory():
    smiles = "CNC(C)=O"
    rdmol = rdmol_from_string(smiles, inp_fmt="smiles")
    # If an unavailable option for the fingerprint is given, an error is raised
    with pytest.raises(Exception) as ke:
        fp_wrong_input = compute_mol_fingerprint(rdmol, "wrong_fp")
    assert "KeyError" in str(ke.type)

    # Changing the default parameters, a different fingerprint is generated for the same molecule
    fp_rdkit = compute_mol_fingerprint(rdmol, "rdkit")
    parameters = {"fpSize": 1024, "countSimulation": True}
    fp_rdkit_params = compute_mol_fingerprint(rdmol, "rdkit", parameters=parameters)
    assert fp_rdkit_params != fp_rdkit


def test_compute_reaction_fingerprint_factory():
    smiles = "CN.CC(O)=O>O>CNC(C)=O"
    rdrxn = rdrxn_from_string(smiles, inp_fmt="smiles")

    # 'structure_fp' and 'difference_fp' generate different fingerprint for the same reaction smiles
    fp_struct = compute_reaction_fingerprint(rdrxn, "structure_fp")
    fp_dif = compute_reaction_fingerprint(rdrxn, "difference_fp")
    assert fp_struct != fp_dif

    # different fingerprints are generated is a different fingerprint type is specified
    fp_struct_param = compute_reaction_fingerprint(
        rdrxn,
        "structure_fp",
        params={"fpType": rdkit.Chem.rdChemReactions.FingerprintType.MorganFP},
    )
    assert fp_struct != fp_struct_param

    with pytest.raises(Exception) as ke:
        fp_wrong_input = compute_reaction_fingerprint(rdrxn, "wrong_fp")
    assert "KeyError" in str(ke.type)


def test_rdrxn_role_reassignment():
    print("IMPLEMENT: rdrxn_role_reassignment")

    test_set = [
        {
            "name": "rnx_1",
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {
                "reactants": ["[CH3:1][C:2]([OH:3])=[O:4]", "[CH3:6][NH2:5]"],
                "reagents": [],
                "products": ["[CH3:6][NH:5][C:2]([CH3:1])=[O:4]", "[OH2:3]"],
            },
        },
        {
            "name": "rnx_2",
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4]>[CH3:6][NH2:5]>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {
                "reactants": ["[CH3:1][C:2]([OH:3])=[O:4]", "[CH3:6][NH2:5]"],
                "reagents": [],
                "products": ["[CH3:6][NH:5][C:2]([CH3:1])=[O:4]", "[OH2:3]"],
            },
        },
        {
            "name": "rnx_3",
            "smiles": "[CH3:6][NH2:5]>[CH3:1][C:2]([OH:3])=[O:4]>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {
                "reactants": ["[CH3:1][C:2]([OH:3])=[O:4]", "[CH3:6][NH2:5]"],
                "reagents": [],
                "products": ["[CH3:6][NH:5][C:2]([CH3:1])=[O:4]", "[OH2:3]"],
            },
        },
        {
            "name": "rnx_4",
            "smiles": ">[CH3:6][NH2:5].[CH3:1][C:2]([OH:3])=[O:4]>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {
                "reactants": ["[CH3:1][C:2]([OH:3])=[O:4]", "[CH3:6][NH2:5]"],
                "reagents": [],
                "products": ["[CH3:6][NH:5][C:2]([CH3:1])=[O:4]", "[OH2:3]"],
            },
        },
        {
            "name": "rnx_5",
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>C>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {
                "reactants": ["[CH3:1][C:2]([OH:3])=[O:4]", "[CH3:6][NH2:5]"],
                "reagents": ["C"],
                "products": ["[CH3:6][NH:5][C:2]([CH3:1])=[O:4]", "[OH2:3]"],
            },
        },
        {
            "name": "rnx_6",
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>CN>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {
                "reactants": ["[CH3:1][C:2]([OH:3])=[O:4]", "[CH3:6][NH2:5]"],
                "reagents": ["CN"],
                "products": ["[CH3:6][NH:5][C:2]([CH3:1])=[O:4]", "[OH2:3]"],
            },
        },
        {
            "name": "rnx_7",
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5].CN>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {
                "reactants": ["[CH3:1][C:2]([OH:3])=[O:4]", "[CH3:6][NH2:5]"],
                "reagents": ["CN"],
                "products": ["[CH3:6][NH:5][C:2]([CH3:1])=[O:4]", "[OH2:3]"],
            },
        },
        {
            "name": "rnx_8",
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4].CN>[CH3:6][NH2:5]>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {
                "reactants": ["[CH3:1][C:2]([OH:3])=[O:4]", "[CH3:6][NH2:5]"],
                "reagents": ["CN"],
                "products": ["[CH3:6][NH:5][C:2]([CH3:1])=[O:4]", "[OH2:3]"],
            },
        },
        {
            "name": "rnx_9",
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4].C[O:3]>[CH3:6][NH2:5]>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {
                "reactants": ["[CH3:1][C:2]([OH:3])=[O:4]", "[CH3:6][NH2:5]"],
                "reagents": ["CN"],
                "products": ["[CH3:6][NH:5][C:2]([CH3:1])=[O:4]", "[OH2:3]"],
            },
        },
    ]

    for item in test_set:
        rdrxn = rdrxn_from_string(input_string=item.get("smiles"), inp_fmt="smiles")
        assert rdrxn
        rdrxn_new = rdrxn_role_reassignment(rdrxn=rdrxn, desired_product_idx=0)
        rdChemReactions.SanitizeRxn(rdrxn_new)
        rdrxn_new.Initialize()
        # rdChemReactions.RemoveMappingNumbersFromReactions(rdrxn_new)
        rxn_mol_catalog = rdrxn_to_rxn_mol_catalog(rdrxn=rdrxn_new)

        print(
            f"\n{item.get('name')} {Chem.rdChemReactions.ReactionToSmiles(rdrxn_new)}"
        )
        for smiles in item.get("expected").get("reactants"):
            rdmol = remove_rdmol_atom_mapping(rdmol=Chem.MolFromSmiles(smiles))
            pass

        for smiles in item.get("expected").get("reagents"):
            pass


def test_reaction_fp_default_params():
    smiles1 = "CN.CC(O)=O>O>CNC(C)=O"
    smiles2 = "CN.CC(O)=O>C>CNC(C)=O"
    rdrxn1 = rdrxn_from_string(smiles1, inp_fmt="smiles")
    rdrxn2 = rdrxn_from_string(smiles2, inp_fmt="smiles")
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


def test_compute_oxidation_numbers():
    examples_andraos = [
        # {'name': '', 'smiles': '', 'expected': {}},
        {
            "name": "A-001",
            "smiles": "O=S(=O)(O)O",
            "expected": {0: -2, 1: 6, 2: -2, 3: -2, 4: -2},
            "andraos_expected": {1: 6},
        },
        {
            "name": "A-002",
            "smiles": "NS(=O)(=O)O",
            "expected": {0: -3, 1: 6, 2: -2, 3: -2, 4: -2},
            "andraos_expected": {0: -1, 1: 4},
        },  # no match
        {
            "name": "A-003",
            "smiles": "NS(=O)(=O)c1ccccc1",
            "expected": {
                0: -3,
                1: 4,
                2: -2,
                3: -2,
                4: 1,
                5: -1,
                6: -1,
                7: -1,
                8: -1,
                9: -1,
            },
            "andraos_expected": {0: -3, 1: 2},
        },  # no match
        {
            "name": "A-004",
            "smiles": "O=S(=O)(O)c1ccccc1",
            "expected": {
                0: -2,
                1: 4,
                2: -2,
                3: -2,
                4: 1,
                5: -1,
                6: -1,
                7: -1,
                8: -1,
                9: -1,
            },
            "andraos_expected": {1: 4},
        },
        {
            "name": "A-005",
            "smiles": "O=S(=O)(Cl)c1ccccc1",
            "expected": {
                0: -2,
                1: 4,
                2: -2,
                3: -1,
                4: 1,
                5: -1,
                6: -1,
                7: -1,
                8: -1,
                9: -1,
            },
            "andraos_expected": {1: 2},
        },
        # no match
        {
            "name": "A-006",
            "smiles": "S",
            "expected": {0: -2},
            "andraos_expected": {0: -2},
        },
        {
            "name": "A-007",
            "smiles": "CSC",
            "expected": {0: -2, 1: -2, 2: -2},
            "andraos_expected": {1: -2},
        },
        {
            "name": "A-008",
            "smiles": "CS(C)=O",
            "expected": {0: -2, 1: 0, 2: -2, 3: -2},
            "andraos_expected": {1: 0},
        },
        {
            "name": "A-009",
            "smiles": "CS(C)(=O)=O",
            "expected": {0: -2, 1: 2, 2: -2, 3: -2, 4: -2},
            "andraos_expected": {1: 2},
        },
        {
            "name": "A-010",
            "smiles": "COS(=O)(=O)OC",
            "expected": {0: -2, 1: -2, 2: 6, 3: -2, 4: -2, 5: -2, 6: -2},
            "andraos_expected": {2: 6},
        },
        {
            "name": "A-011",
            "smiles": "COP(=O)(OC)OC",
            "expected": {0: -2, 1: -2, 2: 5, 3: -2, 4: -2, 5: -2, 6: -2, 7: -2},
            "andraos_expected": {2: 5},
        },
        {
            "name": "A-012",
            "smiles": "COP(OC)OC",
            "expected": {0: -2, 1: -2, 2: 3, 3: -2, 4: -2, 5: -2, 6: -2},
            "andraos_expected": {2: 3},
        },
        {
            "name": "A-013",
            "smiles": "COP(=O)(C#N)OC",
            "expected": {0: -2, 1: -2, 2: 5, 3: -2, 4: 2, 5: -3, 6: -2, 7: -2},
            "andraos_expected": {2: 3},
        },  # no match [C-P electronegativity]
        {
            "name": "A-014",
            "smiles": "CCP(=O)(CC)CC",
            "expected": {0: -3, 1: -3, 2: 5, 3: -2, 4: -3, 5: -3, 6: -3, 7: -3},
            "andraos_expected": {2: -1},
        },  # no match [C-P electronegativity]
        {
            "name": "A-015",
            "smiles": "CCP(CC)CC",
            "expected": {0: -3, 1: -3, 2: 3, 3: -3, 4: -3, 5: -3, 6: -3},
            "andraos_expected": {2: -3},
        },  # no match [C-P electronegativity]
        {
            "name": "A-016",
            "smiles": "CC[P+](CC)(CC)CC",
            "expected": {0: -3, 1: -3, 2: 5, 3: -3, 4: -3, 5: -3, 6: -3, 7: -3, 8: -3},
            "andraos_expected": {2: -3},
        },  # no match [C-P electronegativity]
        {
            "name": "A-017",
            "smiles": "c1ccncc1",
            "expected": {0: -1, 1: -1, 2: 1, 3: -3, 4: 0, 5: -1},
            "andraos_expected": {3: -3},
        },
        {
            "name": "A-018",
            "smiles": "C[n+]1ccccc1",
            "expected": {0: -2, 1: -3, 2: 1, 3: -1, 4: -1, 5: -1, 6: 0},
            "andraos_expected": {1: -3},
        },
        {
            "name": "A-019",
            "smiles": "[O-][n+]1ccccc1",
            "expected": {0: -2, 1: -1, 2: 1, 3: -1, 4: -1, 5: -1, 6: 0},
            "andraos_expected": {1: -1},
        },
        {
            "name": "A-020",
            "smiles": "C[C-](C)[n+]1ccccc1",
            "expected": {0: -3, 1: 0, 2: -3, 3: -3, 4: 1, 5: -1, 6: -1, 7: -1, 8: 0},
            "andraos_expected": {3: -3},
        },
        {
            "name": "A-021",
            "smiles": "C1CCNCC1",
            "expected": {0: -2, 1: -2, 2: -1, 3: -3, 4: -1, 5: -2},
            "andraos_expected": {3: -3},
        },
        {
            "name": "A-022",
            "smiles": "[O]N1CCCCC1",
            "expected": {0: -1, 1: -1, 2: -1, 3: -2, 4: -2, 5: -2, 6: -1},
            "andraos_expected": {1: -1},
        },
        {
            "name": "A-023",
            "smiles": "N",
            "expected": {0: -3},
            "andraos_expected": {0: -3},
        },
        {
            "name": "A-024",
            "smiles": "CN(C)C",
            "expected": {0: -2, 1: -3, 2: -2, 3: -2},
            "andraos_expected": {1: -3},
        },
        {
            "name": "A-025",
            "smiles": "NO",
            "expected": {0: -1, 1: -2},
            "andraos_expected": {0: -1},
        },
        {
            "name": "A-026",
            "smiles": "[NH4+]",
            "expected": {0: -3},
            "andraos_expected": {0: -3},
        },
        {
            "name": "A-027",
            "smiles": "C[N+](C)(C)C",
            "expected": {0: -2, 1: -3, 2: -2, 3: -2, 4: -2},
            "andraos_expected": {1: -3},
        },
        {
            "name": "A-028",
            "smiles": "C[N+](C)(C)[O-]",
            "expected": {0: -2, 1: -1, 2: -2, 3: -2, 4: -2},
            "andraos_expected": {1: -1},
        },
        {
            "name": "A-029",
            "smiles": "[SiH4]",
            "expected": {0: 4},
            "andraos_expected": {0: 4},
        },
        {
            "name": "A-030",
            "smiles": "C[Si](C)(C)C",
            "expected": {0: -4, 1: 4, 2: -4, 3: -4, 4: -4},
            "andraos_expected": {1: 4},
        },
        {
            "name": "A-031",
            "smiles": "C[Si](C)(C)Cl",
            "expected": {0: -4, 1: 4, 2: -4, 3: -4, 4: -1},
            "andraos_expected": {1: 4},
        },
        {
            "name": "A-032",
            "smiles": "C[Si](C)(C)O",
            "expected": {0: -4, 1: 4, 2: -4, 3: -4, 4: -2},
            "andraos_expected": {1: 4},
        },
        {
            "name": "A-033",
            "smiles": "C",
            "expected": {0: -4},
            "andraos_expected": {0: -4},
        },
        {
            "name": "A-034",
            "smiles": "CO",
            "expected": {0: -2, 1: -2},
            "andraos_expected": {0: -2},
        },
        {
            "name": "A-035",
            "smiles": "C=O",
            "expected": {0: 0, 1: -2},
            "andraos_expected": {0: 0},
        },
        {
            "name": "A-036",
            "smiles": "O=CO",
            "expected": {0: -2, 1: 2, 2: -2},
            "andraos_expected": {1: 2},
        },
        {
            "name": "A-037",
            "smiles": "O=C(O)O",
            "expected": {0: -2, 1: 4, 2: -2, 3: -2},
            "andraos_expected": {1: 4},
        },
        {
            "name": "A-038",
            "smiles": "O=C=O",
            "expected": {0: -2, 1: 4, 2: -2},
            "andraos_expected": {1: 4},
        },
        {
            "name": "A-039",
            "smiles": "[C-]#[O+]",
            "expected": {0: 2, 1: -2},
            "andraos_expected": {0: 2},
        },
        {
            "name": "A-041",
            "smiles": "CI",
            "expected": {0: -2, 1: -1},
            "andraos_expected": {0: -2},
        },
        {
            "name": "A-042",
            "smiles": "ICI",
            "expected": {0: -1, 1: 0, 2: -1},
            "andraos_expected": {1: 0},
        },
        {
            "name": "A-043",
            "smiles": "IC(I)I",
            "expected": {0: -1, 1: 2, 2: -1, 3: -1},
            "andraos_expected": {1: 2},
        },
        {
            "name": "A-044",
            "smiles": "IC(I)(I)I",
            "expected": {0: -1, 1: 4, 2: -1, 3: -1, 4: -1},
            "andraos_expected": {1: 4},
        },
        {
            "name": "A-045",
            "smiles": "FC(F)(F)I",
            "expected": {0: -1, 1: 4, 2: -1, 3: -1, 4: -1},
            "andraos_expected": {1: 2},
        },  # no wrong?
        {
            "name": "A-046",
            "smiles": "II",
            "expected": {0: 0, 1: 0},
            "andraos_expected": {0: 0},
        },
        {
            "name": "A-047",
            "smiles": "ClI",
            "expected": {0: -1, 1: 1},
            "andraos_expected": {1: 1},
        },
        {
            "name": "A-048",
            "smiles": "[O-][I+3]([O-])([O-])[O-]",
            "expected": {0: -2, 1: 7, 2: -2, 3: -2, 4: -2},
            "andraos_expected": {1: 7},
        },
        {
            "name": "A-049",
            "smiles": "[O-][I+2]([O-])[O-]",
            "expected": {0: -2, 1: 5, 2: -2, 3: -2},
            "andraos_expected": {1: 5},
        },
        {
            "name": "A-050",
            "smiles": "O=[I+]([O-])c1ccccc1",
            "expected": {0: -2, 1: 3, 2: -2, 3: 1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1},
            "andraos_expected": {1: 3},
        },
        {
            "name": "A-051",
            "smiles": "Ic1ccccc1",
            "expected": {0: -1, 1: 1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1},
            "andraos_expected": {0: -1},
        },
        {
            "name": "A-052",
            "smiles": "CC(=O)OI1(OC(C)=O)(OC(C)=O)OC(=O)c2ccccc21",
            "expected": {
                0: -3,
                1: 3,
                2: -2,
                3: -2,
                4: 3,
                5: -2,
                6: 3,
                7: -3,
                8: -2,
                9: -2,
                10: 3,
                11: -3,
                12: -2,
                13: -2,
                14: 3,
                15: -2,
                16: 0,
                17: -1,
                18: -1,
                19: -1,
                20: -1,
                21: 1,
            },
            "andraos_expected": {4: 3},
        },
        {
            "name": "A-053",
            "smiles": "[Cl-]",
            "expected": {0: -1},
            "andraos_expected": {0: -1},
        },
        {
            "name": "A-054",
            "smiles": "ClCl",
            "expected": {0: 0, 1: 0},
            "andraos_expected": {0: 0},
        },
        {
            "name": "A-055",
            "smiles": "[O-]Cl",
            "expected": {0: -2, 1: 1},
            "andraos_expected": {1: 1},
        },
        {
            "name": "A-056",
            "smiles": "[O-][Cl+][O-]",
            "expected": {0: -2, 1: 3, 2: -2},
            "andraos_expected": {1: 3},
        },
        {
            "name": "A-057",
            "smiles": "[O-][Cl+2]([O-])[O-]",
            "expected": {0: -2, 1: 5, 2: -2, 3: -2},
            "andraos_expected": {1: 5},
        },
        {
            "name": "A-58",
            "smiles": "[O-][Cl+3]([O-])([O-])[O-]",
            "expected": {0: -2, 1: 7, 2: -2, 3: -2, 4: -2},
            "andraos_expected": {1: 7},
        },
    ]
    algorithm_assessment_failures = []
    andraos_assesment_failures = []
    for item in examples_andraos:
        smiles = item.get("smiles")
        rdmol_inp = Chem.MolFromSmiles(smiles)
        rdmol_oxn = compute_oxidation_numbers(rdmol=rdmol_inp)
        onxidation_numbers = {
            atom.GetIdx(): atom.GetIntProp("_OxidationNumber")
            for atom in rdmol_oxn.GetAtoms()
        }
        algorithm_assessment = onxidation_numbers == item.get("expected")

        if andraos_expected := item.get("andraos_expected"):
            andraos_actual = {
                k: onxidation_numbers.get(k) for k in andraos_expected.keys()
            }
            andraos_assesment = andraos_expected == andraos_actual
        else:
            andraos_assesment = None
            andraos_actual = None

        if andraos_assesment is False:
            # print(item.get('name'), algorithm_assessment, andraos_assesment, andraos_expected, andraos_actual)
            pass
            andraos_assesment_failures.append({**item, **{"andraos": andraos_expected}})
        if not algorithm_assessment:
            algorithm_assessment_failures.append(
                {**item, **{"calculated": algorithm_assessment}}
            )

    assert len(algorithm_assessment_failures) == 0


def test_molecular_hashing():
    examples = [
        {"name": "ra1", "smiles": "Cc1ccc2c(C(=O)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1"},
        {"name": "ra2", "smiles": "Cc1ccc2c(C(=O)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1"},
        {"name": "ra3", "smiles": "CCCCCn1cc(C(=O)c2cccc3ccccc23)c2ccccc21"},
        {"name": "ra4", "smiles": "CC1COCCN1CCn1cc(C(=O)c2cccc3ccccc23)c2ccccc21"},
        {"name": "ra5", "smiles": "Cc1ccc(C(=O)c2cn(CCN3CCOCC3)c3ccccc23)c2ccccc12"},
        {"name": "ra6", "smiles": "Cc1c(CCN2CCOCC2)c2ccccc2n1C(=O)c1cccc2ccccc12"},
        {"name": "ra7", "smiles": "CN1CCN(C)C(Cn2cc(C(=O)c3cccc4ccccc34)c3ccccc32)C1"},
        {"name": "ma1", "smiles": "CCC1CC(C(=O)c2cccc(C)c2)CCC1=O"},
        {"name": "ma2", "smiles": "CCC1CC(C(=O)c2ccccc2)CCC1=O"},
        {"name": "ma3", "smiles": "C=C(C)C(c1ccccc1)S(=O)CC(N)=O"},
        {"name": "ma4", "smiles": "Cc1cccc(C(C2CCC(N)CC2)C(F)(F)F)c1"},
        {"name": "ma5", "smiles": "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21"},
        {"name": "ma6", "smiles": "CCCOC(c1ccc(Cl)cc1)C1CCCCC1"},
        {"name": "ta1", "smiles": "OC1=NCCC1"},
        {"name": "ta2", "smiles": "O=C1CCCN1"},
        {"name": "sa1", "smiles": "CC[C@@H](C)[C@H](O)Cl"},
        {"name": "sa2", "smiles": "CC[C@@H](C)[C@@H](O)Cl"},
        {"name": "sa3", "smiles": "CC[C@@H](C)C(O)Cl"},
        {"name": "sa4", "smiles": "CC[C@H](C)[C@H](O)Cl"},
        {"name": "sa5", "smiles": "CC[C@H](C)[C@@H](O)Cl"},
        {"name": "sa6", "smiles": "CC[C@H](C)C(O)Cl"},
        {"name": "sa7", "smiles": "CCC(C)[C@H](O)Cl"},
        {"name": "sa8", "smiles": "CCC(C)[C@@H](O)Cl"},
        {"name": "sa9", "smiles": "CCC(C)C(O)Cl"},
        {"name": "tb1", "smiles": r"C/N=C(\C)C1C(=O)CCC(C)C1=O"},
        {"name": "tb2", "smiles": r"C/N=C(\C)C1=C(O)CCC(C)C1=O"},
        {"name": "tb3", "smiles": r"C/N=C(\C)C1=C(O)C(C)CCC1=O"},
        {"name": "tc1", "smiles": "CC(=O)C1=C(O)C(C)CCC1=O"},
        {"name": "tc2", "smiles": "CC(=O)C1C(=O)CCC(C)C1=O"},
        {"name": "tc3", "smiles": "CC(=O)C1=C(O)CCC(C)C1=O"},
        {"name": "tc4", "smiles": r"C/C(O)=C1\C(=O)CCC(C)C1=O"},
    ]

    reference = {
        "ma1": {
            "AnonymousGraph": "***1**(*(*)*2****(*)*2)***1*",
            "ArthorSubstructureOrder": "001200130100100002000070000000",
            "AtomBondCounts": "18,19",
            "CanonicalSmiles": "CCC1CC(C(=O)c2cccc(C)c2)CCC1=O",
            "DegreeVector": "0,6,8,4",
            "ElementGraph": "CCC1CC(C(O)C2CCCC(C)C2)CCC1O",
            "ExtendedMurcko": "*c1cccc(C(=*)C2CCC(=*)C(*)C2)c1",
            "HetAtomProtomer": "CCC1CC([C]([O])[C]2[CH][CH][CH][C](C)[CH]2)CC[C]1[O]_0",
            "HetAtomTautomer": "CCC1CC([C]([O])[C]2[CH][CH][CH][C](C)[CH]2)CC[C]1[O]_0_0",
            "Mesomer": "CCC1CC([C]([O])[C]2[CH][CH][CH][C](C)[CH]2)CC[C]1[O]_0",
            "MolFormula": "C16H20O2",
            "MurckoScaffold": "c1ccc(CC2CCCCC2)cc1",
            "NetCharge": "0",
            "RedoxPair": "CCC1CC([C]([O])[C]2[CH][CH][CH][C](C)[CH]2)CC[C]1[O]",
            "Regioisomer": "*C.*C(*)=O.*CC.O=C1CCCCC1.c1ccccc1",
            "SmallWorldIndexBR": "B19R2",
            "SmallWorldIndexBRL": "B19R2L8",
            "cx_smiles": "CCC1CC(C(=O)c2cccc(C)c2)CCC1=O",
            "inchi": "InChI=1S/C16H20O2/c1-3-12-10-14(7-8-15(12)17)16(18)13-6-4-5-11(2)9-13/h4-6,9,12,14H,3,7-8,10H2,1-2H3",
            "inchi_KET_15T": "InChI=1/C16H20O2/c1-3-12-10-14(7-8-15(12)17)16(18)13-6-4-5-11(2)9-13/h4-6,9H,3,7,10H2,1-2H3,(H,14,18)(H3,8,12,17)",
            "inchi_key": "MQEBHHFSYXQNPF-UHFFFAOYSA-N",
            "inchikey_KET_15T": "WFPNUSIRQPGLBS-UHFFFAOYNA-N",
            "noiso_smiles": "CCC1CC(C(=O)c2cccc(C)c2)CCC1=O",
            "smiles": "CCC1CC(C(=O)c2cccc(C)c2)CCC1=O",
        },
        "ma2": {
            "AnonymousGraph": "***1**(*(*)*2*****2)***1*",
            "ArthorSubstructureOrder": "0011001201000f000200006a000000",
            "AtomBondCounts": "17,18",
            "CanonicalSmiles": "CCC1CC(C(=O)c2ccccc2)CCC1=O",
            "DegreeVector": "0,5,9,3",
            "ElementGraph": "CCC1CC(C(O)C2CCCCC2)CCC1O",
            "ExtendedMurcko": "*C1CC(C(=*)c2ccccc2)CCC1=*",
            "HetAtomProtomer": "CCC1CC([C]([O])[C]2[CH][CH][CH][CH][CH]2)CC[C]1[O]_0",
            "HetAtomTautomer": "CCC1CC([C]([O])[C]2[CH][CH][CH][CH][CH]2)CC[C]1[O]_0_0",
            "Mesomer": "CCC1CC([C]([O])[C]2[CH][CH][CH][CH][CH]2)CC[C]1[O]_0",
            "MolFormula": "C15H18O2",
            "MurckoScaffold": "c1ccc(CC2CCCCC2)cc1",
            "NetCharge": "0",
            "RedoxPair": "CCC1CC([C]([O])[C]2[CH][CH][CH][CH][CH]2)CC[C]1[O]",
            "Regioisomer": "*C(*)=O.*CC.O=C1CCCCC1.c1ccccc1",
            "SmallWorldIndexBR": "B18R2",
            "SmallWorldIndexBRL": "B18R2L9",
            "cx_smiles": "CCC1CC(C(=O)c2ccccc2)CCC1=O",
            "inchi": "InChI=1S/C15H18O2/c1-2-11-10-13(8-9-14(11)16)15(17)12-6-4-3-5-7-12/h3-7,11,13H,2,8-10H2,1H3",
            "inchi_KET_15T": "InChI=1/C15H18O2/c1-2-11-10-13(8-9-14(11)16)15(17)12-6-4-3-5-7-12/h3-7H,2,8,10H2,1H3,(H,13,17)(H3,9,11,16)",
            "inchi_key": "NGWXTRJFWAYHFL-UHFFFAOYSA-N",
            "inchikey_KET_15T": "NEMNUJJNBNORAW-UHFFFAOYNA-N",
            "noiso_smiles": "CCC1CC(C(=O)c2ccccc2)CCC1=O",
            "smiles": "CCC1CC(C(=O)c2ccccc2)CCC1=O",
        },
        "ma3": {
            "AnonymousGraph": "**(*)**(*)*(*(*)*)*1*****1",
            "ArthorSubstructureOrder": "0010001001000c000400006f000000",
            "AtomBondCounts": "16,16",
            "CanonicalSmiles": "C=C(C)C(c1ccccc1)S(=O)CC(N)=O",
            "DegreeVector": "0,5,6,5",
            "ElementGraph": "CC(C)C(C1CCCCC1)[S](O)CC(N)O",
            "ExtendedMurcko": "*c1ccccc1",
            "HetAtomProtomer": "C=C(C)C([C]1[CH][CH][CH][CH][CH]1)[S]([O])C[C]([N])[O]_2",
            "HetAtomTautomer": "C=C(C)C([C]1[CH][CH][CH][CH][CH]1)[S]([O])C[C]([N])[O]_2_0",
            "Mesomer": "[CH2][C](C)C([C]1[CH][CH][CH][CH][CH]1)[S]([O])C[C](N)[O]_0",
            "MolFormula": "C12H15NO2S",
            "MurckoScaffold": "c1ccccc1",
            "NetCharge": "0",
            "RedoxPair": "[CH2][C](C)C([C]1[CH][CH][CH][CH][CH]1)[S]([O])C[C](N)[O]",
            "Regioisomer": "*CC(=C)C.*S(*)=O.CC(N)=O.c1ccccc1",
            "SmallWorldIndexBR": "B16R1",
            "SmallWorldIndexBRL": "B16R1L6",
            "cx_smiles": "C=C(C)C(c1ccccc1)S(=O)CC(N)=O",
            "inchi": "InChI=1S/C12H15NO2S/c1-9(2)12(16(15)8-11(13)14)10-6-4-3-5-7-10/h3-7,12H,1,8H2,2H3,(H2,13,14)",
            "inchi_KET_15T": "InChI=1/C12H15NO2S/c1-9(2)12(16(15)8-11(13)14)10-6-4-3-5-7-10/h3-7,12H,1H2,2H3,(H4,8,13,14)",
            "inchi_key": "ZFCHMUVIJDAZSM-UHFFFAOYSA-N",
            "inchikey_KET_15T": "DIAQTYFBHNUKSP-UHFFFAOYNA-N",
            "noiso_smiles": "C=C(C)C(c1ccccc1)S(=O)CC(N)=O",
            "smiles": "C=C(C)C(c1ccccc1)S(=O)CC(N)=O",
        },
        "ma4": {
            "AnonymousGraph": "**1***(*(*2****(*)*2)*(*)(*)*)**1",
            "ArthorSubstructureOrder": "0013001401000f000400007c000000",
            "AtomBondCounts": "19,20",
            "CanonicalSmiles": "Cc1cccc(C(C2CCC(N)CC2)C(F)(F)F)c1",
            "DegreeVector": "1,5,8,5",
            "ElementGraph": "CC1CCCC(C(C2CCC(N)CC2)C(F)(F)F)C1",
            "ExtendedMurcko": "*c1cccc(C(*)C2CCC(*)CC2)c1",
            "HetAtomProtomer": "C[C]1[CH][CH][CH][C](C(C2CCC([N])CC2)C(F)(F)F)[CH]1_2",
            "HetAtomTautomer": "C[C]1[CH][CH][CH][C](C(C2CCC([N])CC2)C(F)(F)F)[CH]1_2_0",
            "Mesomer": "C[C]1[CH][CH][CH][C](C(C2CCC(N)CC2)C(F)(F)F)[CH]1_0",
            "MolFormula": "C15H20F3N",
            "MurckoScaffold": "c1ccc(CC2CCCCC2)cc1",
            "NetCharge": "0",
            "RedoxPair": "C[C]1[CH][CH][CH][C](C(C2CCC(N)CC2)C(F)(F)F)[CH]1",
            "Regioisomer": "*C.*C(*)C.*F.*F.*F.*N.C1CCCCC1.c1ccccc1",
            "SmallWorldIndexBR": "B20R2",
            "SmallWorldIndexBRL": "B20R2L8",
            "cx_smiles": "Cc1cccc(C(C2CCC(N)CC2)C(F)(F)F)c1",
            "inchi": "InChI=1S/C15H20F3N/c1-10-3-2-4-12(9-10)14(15(16,17)18)11-5-7-13(19)8-6-11/h2-4,9,11,13-14H,5-8,19H2,1H3",
            "inchi_KET_15T": "InChI=1/C15H20F3N/c1-10-3-2-4-12(9-10)14(15(16,17)18)11-5-7-13(19)8-6-11/h2-4,9,11,13-14H,5-8,19H2,1H3",
            "inchi_key": "JHNUJAVMAYMRLC-UHFFFAOYSA-N",
            "inchikey_KET_15T": "JHNUJAVMAYMRLC-UHFFFAOYNA-N",
            "noiso_smiles": "Cc1cccc(C(C2CCC(N)CC2)C(F)(F)F)c1",
            "smiles": "Cc1cccc(C(C2CCC(N)CC2)C(F)(F)F)c1",
        },
        "ma5": {
            "AnonymousGraph": "***1***(*2***(*)*(*)*2)*2*****12",
            "ArthorSubstructureOrder": "00140016010011000300008f000000",
            "AtomBondCounts": "20,22",
            "CanonicalSmiles": "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21",
            "DegreeVector": "0,7,10,3",
            "ElementGraph": "CNC1CCC(C2CCC(Cl)C(Cl)C2)C2CCCCC12",
            "ExtendedMurcko": "*c1ccc(C2CCC(*)c3ccccc32)cc1*",
            "HetAtomProtomer": "C[N]C1CCC([C]2[CH][CH][C](Cl)[C](Cl)[CH]2)[C]2[CH][CH][CH][CH][C]21_1",
            "HetAtomTautomer": "C[N]C1CCC([C]2[CH][CH][C](Cl)[C](Cl)[CH]2)[C]2[CH][CH][CH][CH][C]21_1_0",
            "Mesomer": "CNC1CCC([C]2[CH][CH][C](Cl)[C](Cl)[CH]2)[C]2[CH][CH][CH][CH][C]21_0",
            "MolFormula": "C17H17Cl2N",
            "MurckoScaffold": "c1ccc(C2CCCc3ccccc32)cc1",
            "NetCharge": "0",
            "RedoxPair": "CNC1CCC([C]2[CH][CH][C](Cl)[C](Cl)[CH]2)[C]2[CH][CH][CH][CH][C]21",
            "Regioisomer": "*Cl.*Cl.*N*.C.c1ccc2c(c1)CCCC2.c1ccccc1",
            "SmallWorldIndexBR": "B22R3",
            "SmallWorldIndexBRL": "B22R3L10",
            "cx_smiles": "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21",
            "inchi": "InChI=1S/C17H17Cl2N/c1-20-17-9-7-12(13-4-2-3-5-14(13)17)11-6-8-15(18)16(19)10-11/h2-6,8,10,12,17,20H,7,9H2,1H3",
            "inchi_KET_15T": "InChI=1/C17H17Cl2N/c1-20-17-9-7-12(13-4-2-3-5-14(13)17)11-6-8-15(18)16(19)10-11/h2-6,8,10,12,17,20H,7,9H2,1H3",
            "inchi_key": "VGKDLMBJGBXTGI-UHFFFAOYSA-N",
            "inchikey_KET_15T": "VGKDLMBJGBXTGI-UHFFFAOYNA-N",
            "noiso_smiles": "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21",
            "smiles": "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21",
        },
        "ma6": {
            "AnonymousGraph": "*****(*1*****1)*1***(*)**1",
            "ArthorSubstructureOrder": "001200130100100002000079000000",
            "AtomBondCounts": "18,19",
            "CanonicalSmiles": "CCCOC(c1ccc(Cl)cc1)C1CCCCC1",
            "DegreeVector": "0,4,12,2",
            "ElementGraph": "CCCOC(C1CCCCC1)C1CCC(Cl)CC1",
            "ExtendedMurcko": "*c1ccc(C(*)C2CCCCC2)cc1",
            "HetAtomProtomer": "CCCOC([C]1[CH][CH][C](Cl)[CH][CH]1)C1CCCCC1_0",
            "HetAtomTautomer": "CCCOC([C]1[CH][CH][C](Cl)[CH][CH]1)C1CCCCC1_0_0",
            "Mesomer": "CCCOC([C]1[CH][CH][C](Cl)[CH][CH]1)C1CCCCC1_0",
            "MolFormula": "C16H23ClO",
            "MurckoScaffold": "c1ccc(CC2CCCCC2)cc1",
            "NetCharge": "0",
            "RedoxPair": "CCCOC([C]1[CH][CH][C](Cl)[CH][CH]1)C1CCCCC1",
            "Regioisomer": "*C*.*Cl.*O*.C1CCCCC1.CCC.c1ccccc1",
            "SmallWorldIndexBR": "B19R2",
            "SmallWorldIndexBRL": "B19R2L12",
            "cx_smiles": "CCCOC(c1ccc(Cl)cc1)C1CCCCC1",
            "inchi": "InChI=1S/C16H23ClO/c1-2-12-18-16(13-6-4-3-5-7-13)14-8-10-15(17)11-9-14/h8-11,13,16H,2-7,12H2,1H3",
            "inchi_KET_15T": "InChI=1/C16H23ClO/c1-2-12-18-16(13-6-4-3-5-7-13)14-8-10-15(17)11-9-14/h8-11,13,16H,2-7,12H2,1H3",
            "inchi_key": "IJTNMUAWUDRFON-UHFFFAOYSA-N",
            "inchikey_KET_15T": "IJTNMUAWUDRFON-UHFFFAOYNA-N",
            "noiso_smiles": "CCCOC(c1ccc(Cl)cc1)C1CCCCC1",
            "smiles": "CCCOC(c1ccc(Cl)cc1)C1CCCCC1",
        },
        "ra1": {
            "AnonymousGraph": "**(*1****2*****12)*1**(***2*****2)*2**(*)***12",
            "ArthorSubstructureOrder": "001e002201001a00040000ba000000",
            "AtomBondCounts": "30,34",
            "CanonicalSmiles": "Cc1ccc2c(C(=O)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1",
            "DegreeVector": "0,10,18,2",
            "ElementGraph": "CC1CCC2C(C(O)C3CCCC4CCCCC34)CN(CCN3CCOCC3)C2C1",
            "ExtendedMurcko": "*c1ccc2c(C(=*)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1",
            "HetAtomProtomer": "C[C]1[CH][CH][C]2[C]([C]([O])[C]3[CH][CH][CH][C]4[CH][CH][CH][CH][C]43)[CH]N(CCN3CCOCC3)[C]2[CH]1_0",
            "HetAtomTautomer": "C[C]1[CH][CH][C]2[C]([C]([O])[C]3[CH][CH][CH][C]4[CH][CH][CH][CH][C]43)[CH]N(CCN3CCOCC3)[C]2[CH]1_0_0",
            "Mesomer": "C[C]1[CH][CH][C]2[C]([C]([O])[C]3[CH][CH][CH][C]4[CH][CH][CH][CH][C]34)[CH]N(CCN3CCOCC3)[C]2[CH]1_0",
            "MolFormula": "C26H26N2O2",
            "MurckoScaffold": "c1ccc2c(Cc3cn(CCN4CCOCC4)c4ccccc34)cccc2c1",
            "NetCharge": "0",
            "RedoxPair": "C[C]1[CH][CH][C]2[C]([C]([O])[C]3[CH][CH][CH][C]4[CH][CH][CH][CH][C]34)[CH]N(CCN3CCOCC3)[C]2[CH]1",
            "Regioisomer": "*C.*C(*)=O.*CC*.C1COCCN1.c1ccc2[nH]ccc2c1.c1ccc2ccccc2c1",
            "SmallWorldIndexBR": "B34R5",
            "SmallWorldIndexBRL": "B34R5L18",
            "cx_smiles": "Cc1ccc2c(C(=O)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1",
            "inchi": "InChI=1S/C26H26N2O2/c1-19-9-10-22-24(26(29)23-8-4-6-20-5-2-3-7-21(20)23)18-28(25(22)17-19)12-11-27-13-15-30-16-14-27/h2-10,17-18H,11-16H2,1H3",
            "inchi_KET_15T": "InChI=1/C26H26N2O2/c1-19-9-10-22-24(26(29)23-8-4-6-20-5-2-3-7-21(20)23)18-28(25(22)17-19)12-11-27-13-15-30-16-14-27/h2-10,17-18H,11-16H2,1H3",
            "inchi_key": "DDVFEKLZEPNGMS-UHFFFAOYSA-N",
            "inchikey_KET_15T": "DDVFEKLZEPNGMS-UHFFFAOYNA-N",
            "noiso_smiles": "Cc1ccc2c(C(=O)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1",
            "smiles": "Cc1ccc2c(C(=O)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1",
        },
        "ra2": {
            "AnonymousGraph": "**(*1****2*****12)*1**(***2*****2)*2**(*)***12",
            "ArthorSubstructureOrder": "001e002201001a00040000ba000000",
            "AtomBondCounts": "30,34",
            "CanonicalSmiles": "Cc1ccc2c(C(=O)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1",
            "DegreeVector": "0,10,18,2",
            "ElementGraph": "CC1CCC2C(C(O)C3CCCC4CCCCC34)CN(CCN3CCOCC3)C2C1",
            "ExtendedMurcko": "*c1ccc2c(C(=*)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1",
            "HetAtomProtomer": "C[C]1[CH][CH][C]2[C]([C]([O])[C]3[CH][CH][CH][C]4[CH][CH][CH][CH][C]43)[CH]N(CCN3CCOCC3)[C]2[CH]1_0",
            "HetAtomTautomer": "C[C]1[CH][CH][C]2[C]([C]([O])[C]3[CH][CH][CH][C]4[CH][CH][CH][CH][C]43)[CH]N(CCN3CCOCC3)[C]2[CH]1_0_0",
            "Mesomer": "C[C]1[CH][CH][C]2[C]([C]([O])[C]3[CH][CH][CH][C]4[CH][CH][CH][CH][C]34)[CH]N(CCN3CCOCC3)[C]2[CH]1_0",
            "MolFormula": "C26H26N2O2",
            "MurckoScaffold": "c1ccc2c(Cc3cn(CCN4CCOCC4)c4ccccc34)cccc2c1",
            "NetCharge": "0",
            "RedoxPair": "C[C]1[CH][CH][C]2[C]([C]([O])[C]3[CH][CH][CH][C]4[CH][CH][CH][CH][C]34)[CH]N(CCN3CCOCC3)[C]2[CH]1",
            "Regioisomer": "*C.*C(*)=O.*CC*.C1COCCN1.c1ccc2[nH]ccc2c1.c1ccc2ccccc2c1",
            "SmallWorldIndexBR": "B34R5",
            "SmallWorldIndexBRL": "B34R5L18",
            "cx_smiles": "Cc1ccc2c(C(=O)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1",
            "inchi": "InChI=1S/C26H26N2O2/c1-19-9-10-22-24(26(29)23-8-4-6-20-5-2-3-7-21(20)23)18-28(25(22)17-19)12-11-27-13-15-30-16-14-27/h2-10,17-18H,11-16H2,1H3",
            "inchi_KET_15T": "InChI=1/C26H26N2O2/c1-19-9-10-22-24(26(29)23-8-4-6-20-5-2-3-7-21(20)23)18-28(25(22)17-19)12-11-27-13-15-30-16-14-27/h2-10,17-18H,11-16H2,1H3",
            "inchi_key": "DDVFEKLZEPNGMS-UHFFFAOYSA-N",
            "inchikey_KET_15T": "DDVFEKLZEPNGMS-UHFFFAOYNA-N",
            "noiso_smiles": "Cc1ccc2c(C(=O)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1",
            "smiles": "Cc1ccc2c(C(=O)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1",
        },
        "ra3": {
            "AnonymousGraph": "******1**(*(*)*2****3*****23)*2*****12",
            "ArthorSubstructureOrder": "001a001d010018000200009f000000",
            "AtomBondCounts": "26,29",
            "CanonicalSmiles": "CCCCCn1cc(C(=O)c2cccc3ccccc23)c2ccccc21",
            "DegreeVector": "0,8,16,2",
            "ElementGraph": "CCCCCN1CC(C(O)C2CCCC3CCCCC23)C2CCCCC21",
            "ExtendedMurcko": "*n1cc(C(=*)c2cccc3ccccc23)c2ccccc21",
            "HetAtomProtomer": "CCCCCN1[CH][C]([C]([O])[C]2[CH][CH][CH][C]3[CH][CH][CH][CH][C]32)[C]2[CH][CH][CH][CH][C]21_0",
            "HetAtomTautomer": "CCCCCN1[CH][C]([C]([O])[C]2[CH][CH][CH][C]3[CH][CH][CH][CH][C]32)[C]2[CH][CH][CH][CH][C]21_0_0",
            "Mesomer": "CCCCCN1[CH][C]([C]([O])[C]2[CH][CH][CH][C]3[CH][CH][CH][CH][C]23)[C]2[CH][CH][CH][CH][C]21_0",
            "MolFormula": "C24H23NO",
            "MurckoScaffold": "c1ccc2c(Cc3c[nH]c4ccccc34)cccc2c1",
            "NetCharge": "0",
            "RedoxPair": "CCCCCN1[CH][C]([C]([O])[C]2[CH][CH][CH][C]3[CH][CH][CH][CH][C]23)[C]2[CH][CH][CH][CH][C]21",
            "Regioisomer": "*C(*)=O.*CCCCC.c1ccc2[nH]ccc2c1.c1ccc2ccccc2c1",
            "SmallWorldIndexBR": "B29R4",
            "SmallWorldIndexBRL": "B29R4L16",
            "cx_smiles": "CCCCCn1cc(C(=O)c2cccc3ccccc23)c2ccccc21",
            "inchi": "InChI=1S/C24H23NO/c1-2-3-8-16-25-17-22(20-13-6-7-15-23(20)25)24(26)21-14-9-11-18-10-4-5-12-19(18)21/h4-7,9-15,17H,2-3,8,16H2,1H3",
            "inchi_KET_15T": "InChI=1/C24H23NO/c1-2-3-8-16-25-17-22(20-13-6-7-15-23(20)25)24(26)21-14-9-11-18-10-4-5-12-19(18)21/h4-7,9-15,17H,2-3,8,16H2,1H3",
            "inchi_key": "JDNLPKCAXICMBW-UHFFFAOYSA-N",
            "inchikey_KET_15T": "JDNLPKCAXICMBW-UHFFFAOYNA-N",
            "noiso_smiles": "CCCCCn1cc(C(=O)c2cccc3ccccc23)c2ccccc21",
            "smiles": "CCCCCn1cc(C(=O)c2cccc3ccccc23)c2ccccc21",
        },
        "ra4": {
            "AnonymousGraph": "**1*****1***1**(*(*)*2****3*****23)*2*****12",
            "ArthorSubstructureOrder": "001e002201001a00040000ba000000",
            "AtomBondCounts": "30,34",
            "CanonicalSmiles": "CC1COCCN1CCn1cc(C(=O)c2cccc3ccccc23)c2ccccc21",
            "DegreeVector": "0,10,18,2",
            "ElementGraph": "CC1COCCN1CCN1CC(C(O)C2CCCC3CCCCC23)C2CCCCC21",
            "ExtendedMurcko": "*C1COCCN1CCn1cc(C(=*)c2cccc3ccccc23)c2ccccc21",
            "HetAtomProtomer": "CC1COCCN1CCN1[CH][C]([C]([O])[C]2[CH][CH][CH][C]3[CH][CH][CH][CH][C]32)[C]2[CH][CH][CH][CH][C]21_0",
            "HetAtomTautomer": "CC1COCCN1CCN1[CH][C]([C]([O])[C]2[CH][CH][CH][C]3[CH][CH][CH][CH][C]32)[C]2[CH][CH][CH][CH][C]21_0_0",
            "Mesomer": "CC1COCCN1CCN1[CH][C]([C]([O])[C]2[CH][CH][CH][C]3[CH][CH][CH][CH][C]23)[C]2[CH][CH][CH][CH][C]21_0",
            "MolFormula": "C26H26N2O2",
            "MurckoScaffold": "c1ccc2c(Cc3cn(CCN4CCOCC4)c4ccccc34)cccc2c1",
            "NetCharge": "0",
            "RedoxPair": "CC1COCCN1CCN1[CH][C]([C]([O])[C]2[CH][CH][CH][C]3[CH][CH][CH][CH][C]23)[C]2[CH][CH][CH][CH][C]21",
            "Regioisomer": "*C.*C(*)=O.*CC*.C1COCCN1.c1ccc2[nH]ccc2c1.c1ccc2ccccc2c1",
            "SmallWorldIndexBR": "B34R5",
            "SmallWorldIndexBRL": "B34R5L18",
            "cx_smiles": "CC1COCCN1CCn1cc(C(=O)c2cccc3ccccc23)c2ccccc21",
            "inchi": "InChI=1S/C26H26N2O2/c1-19-18-30-16-15-27(19)13-14-28-17-24(22-10-4-5-12-25(22)28)26(29)23-11-6-8-20-7-2-3-9-21(20)23/h2-12,17,19H,13-16,18H2,1H3",
            "inchi_KET_15T": "InChI=1/C26H26N2O2/c1-19-18-30-16-15-27(19)13-14-28-17-24(22-10-4-5-12-25(22)28)26(29)23-11-6-8-20-7-2-3-9-21(20)23/h2-12,17,19H,13-16,18H2,1H3",
            "inchi_key": "AZPCJOZKMQFKLE-UHFFFAOYSA-N",
            "inchikey_KET_15T": "AZPCJOZKMQFKLE-UHFFFAOYNA-N",
            "noiso_smiles": "CC1COCCN1CCn1cc(C(=O)c2cccc3ccccc23)c2ccccc21",
            "smiles": "CC1COCCN1CCn1cc(C(=O)c2cccc3ccccc23)c2ccccc21",
        },
        "ra5": {
            "AnonymousGraph": "**(*1***(*)*2*****12)*1**(***2*****2)*2*****12",
            "ArthorSubstructureOrder": "001e002201001a00040000ba000000",
            "AtomBondCounts": "30,34",
            "CanonicalSmiles": "Cc1ccc(C(=O)c2cn(CCN3CCOCC3)c3ccccc23)c2ccccc12",
            "DegreeVector": "0,10,18,2",
            "ElementGraph": "CC1CCC(C(O)C2CN(CCN3CCOCC3)C3CCCCC23)C2CCCCC12",
            "ExtendedMurcko": "*c1ccc(C(=*)c2cn(CCN3CCOCC3)c3ccccc23)c2ccccc12",
            "HetAtomProtomer": "C[C]1[CH][CH][C]([C]([O])[C]2[CH]N(CCN3CCOCC3)[C]3[CH][CH][CH][CH][C]23)[C]2[CH][CH][CH][CH][C]12_0",
            "HetAtomTautomer": "C[C]1[CH][CH][C]([C]([O])[C]2[CH]N(CCN3CCOCC3)[C]3[CH][CH][CH][CH][C]23)[C]2[CH][CH][CH][CH][C]12_0_0",
            "Mesomer": "C[C]1[CH][CH][C]([C]([O])[C]2[CH]N(CCN3CCOCC3)[C]3[CH][CH][CH][CH][C]23)[C]2[CH][CH][CH][CH][C]12_0",
            "MolFormula": "C26H26N2O2",
            "MurckoScaffold": "c1ccc2c(Cc3cn(CCN4CCOCC4)c4ccccc34)cccc2c1",
            "NetCharge": "0",
            "RedoxPair": "C[C]1[CH][CH][C]([C]([O])[C]2[CH]N(CCN3CCOCC3)[C]3[CH][CH][CH][CH][C]23)[C]2[CH][CH][CH][CH][C]12",
            "Regioisomer": "*C.*C(*)=O.*CC*.C1COCCN1.c1ccc2[nH]ccc2c1.c1ccc2ccccc2c1",
            "SmallWorldIndexBR": "B34R5",
            "SmallWorldIndexBRL": "B34R5L18",
            "cx_smiles": "Cc1ccc(C(=O)c2cn(CCN3CCOCC3)c3ccccc23)c2ccccc12",
            "inchi": "InChI=1S/C26H26N2O2/c1-19-10-11-23(21-7-3-2-6-20(19)21)26(29)24-18-28(25-9-5-4-8-22(24)25)13-12-27-14-16-30-17-15-27/h2-11,18H,12-17H2,1H3",
            "inchi_KET_15T": "InChI=1/C26H26N2O2/c1-19-10-11-23(21-7-3-2-6-20(19)21)26(29)24-18-28(25-9-5-4-8-22(24)25)13-12-27-14-16-30-17-15-27/h2-11,18H,12-17H2,1H3",
            "inchi_key": "ICKWPPYMDARCKJ-UHFFFAOYSA-N",
            "inchikey_KET_15T": "ICKWPPYMDARCKJ-UHFFFAOYNA-N",
            "noiso_smiles": "Cc1ccc(C(=O)c2cn(CCN3CCOCC3)c3ccccc23)c2ccccc12",
            "smiles": "Cc1ccc(C(=O)c2cn(CCN3CCOCC3)c3ccccc23)c2ccccc12",
        },
        "ra6": {
            "AnonymousGraph": "**(*1****2*****12)*1*(*)*(***2*****2)*2*****12",
            "ArthorSubstructureOrder": "001e002201001a00040000ba000000",
            "AtomBondCounts": "30,34",
            "CanonicalSmiles": "Cc1c(CCN2CCOCC2)c2ccccc2n1C(=O)c1cccc2ccccc12",
            "DegreeVector": "0,10,18,2",
            "ElementGraph": "CC1C(CCN2CCOCC2)C2CCCCC2N1C(O)C1CCCC2CCCCC12",
            "ExtendedMurcko": "*c1c(CCN2CCOCC2)c2ccccc2n1C(=*)c1cccc2ccccc12",
            "HetAtomProtomer": "C[C]1[C](CCN2CCOCC2)[C]2[CH][CH][CH][CH][C]2N1[C]([O])[C]1[CH][CH][CH][C]2[CH][CH][CH][CH][C]21_0",
            "HetAtomTautomer": "C[C]1[C](CCN2CCOCC2)[C]2[CH][CH][CH][CH][C]2N1[C]([O])[C]1[CH][CH][CH][C]2[CH][CH][CH][CH][C]21_0_0",
            "Mesomer": "C[C]1[C](CCN2CCOCC2)[C]2[CH][CH][CH][CH][C]2N1[C]([O])[C]1[CH][CH][CH][C]2[CH][CH][CH][CH][C]12_0",
            "MolFormula": "C26H26N2O2",
            "MurckoScaffold": "c1ccc2c(Cn3cc(CCN4CCOCC4)c4ccccc43)cccc2c1",
            "NetCharge": "0",
            "RedoxPair": "C[C]1[C](CCN2CCOCC2)[C]2[CH][CH][CH][CH][C]2N1[C]([O])[C]1[CH][CH][CH][C]2[CH][CH][CH][CH][C]12",
            "Regioisomer": "*C.*C(*)=O.*CC*.C1COCCN1.c1ccc2[nH]ccc2c1.c1ccc2ccccc2c1",
            "SmallWorldIndexBR": "B34R5",
            "SmallWorldIndexBRL": "B34R5L18",
            "cx_smiles": "Cc1c(CCN2CCOCC2)c2ccccc2n1C(=O)c1cccc2ccccc12",
            "inchi": "InChI=1S/C26H26N2O2/c1-19-21(13-14-27-15-17-30-18-16-27)23-10-4-5-12-25(23)28(19)26(29)24-11-6-8-20-7-2-3-9-22(20)24/h2-12H,13-18H2,1H3",
            "inchi_KET_15T": "InChI=1/C26H26N2O2/c1-19-21(13-14-27-15-17-30-18-16-27)23-10-4-5-12-25(23)28(19)26(29)24-11-6-8-20-7-2-3-9-22(20)24/h2-12H,13-18H2,1H3",
            "inchi_key": "FKFDKPAJXHDYTK-UHFFFAOYSA-N",
            "inchikey_KET_15T": "FKFDKPAJXHDYTK-UHFFFAOYNA-N",
            "noiso_smiles": "Cc1c(CCN2CCOCC2)c2ccccc2n1C(=O)c1cccc2ccccc12",
            "smiles": "Cc1c(CCN2CCOCC2)c2ccccc2n1C(=O)c1cccc2ccccc12",
        },
        "ra7": {
            "AnonymousGraph": "**1***(*)*(**2**(*(*)*3****4*****34)*3*****23)*1",
            "ArthorSubstructureOrder": "001e002201001a00040000b9000000",
            "AtomBondCounts": "30,34",
            "CanonicalSmiles": "CN1CCN(C)C(Cn2cc(C(=O)c3cccc4ccccc34)c3ccccc32)C1",
            "DegreeVector": "0,11,16,3",
            "ElementGraph": "CN1CCN(C)C(CN2CC(C(O)C3CCCC4CCCCC34)C3CCCCC32)C1",
            "ExtendedMurcko": "*N1CCN(*)C(Cn2cc(C(=*)c3cccc4ccccc34)c3ccccc32)C1",
            "HetAtomProtomer": "CN1CCN(C)C(CN2[CH][C]([C]([O])[C]3[CH][CH][CH][C]4[CH][CH][CH][CH][C]43)[C]3[CH][CH][CH][CH][C]32)C1_0",
            "HetAtomTautomer": "CN1CCN(C)C(CN2[CH][C]([C]([O])[C]3[CH][CH][CH][C]4[CH][CH][CH][CH][C]43)[C]3[CH][CH][CH][CH][C]32)C1_0_0",
            "Mesomer": "CN1CCN(C)C(CN2[CH][C]([C]([O])[C]3[CH][CH][CH][C]4[CH][CH][CH][CH][C]34)[C]3[CH][CH][CH][CH][C]32)C1_0",
            "MolFormula": "C26H27N3O",
            "MurckoScaffold": "c1ccc2c(Cc3cn(CC4CNCCN4)c4ccccc34)cccc2c1",
            "NetCharge": "0",
            "RedoxPair": "CN1CCN(C)C(CN2[CH][C]([C]([O])[C]3[CH][CH][CH][C]4[CH][CH][CH][CH][C]34)[C]3[CH][CH][CH][CH][C]32)C1",
            "Regioisomer": "*C.*C.*C(*)=O.*C*.C1CNCCN1.c1ccc2[nH]ccc2c1.c1ccc2ccccc2c1",
            "SmallWorldIndexBR": "B34R5",
            "SmallWorldIndexBRL": "B34R5L16",
            "cx_smiles": "CN1CCN(C)C(Cn2cc(C(=O)c3cccc4ccccc34)c3ccccc32)C1",
            "inchi": "InChI=1S/C26H27N3O/c1-27-14-15-28(2)20(16-27)17-29-18-24(22-11-5-6-13-25(22)29)26(30)23-12-7-9-19-8-3-4-10-21(19)23/h3-13,18,20H,14-17H2,1-2H3",
            "inchi_KET_15T": "InChI=1/C26H27N3O/c1-27-14-15-28(2)20(16-27)17-29-18-24(22-11-5-6-13-25(22)29)26(30)23-12-7-9-19-8-3-4-10-21(19)23/h3-13,18,20H,14-17H2,1-2H3",
            "inchi_key": "CPHORMQWXVSQFN-UHFFFAOYSA-N",
            "inchikey_KET_15T": "CPHORMQWXVSQFN-UHFFFAOYNA-N",
            "noiso_smiles": "CN1CCN(C)C(Cn2cc(C(=O)c3cccc4ccccc34)c3ccccc32)C1",
            "smiles": "CN1CCN(C)C(Cn2cc(C(=O)c3cccc4ccccc34)c3ccccc32)C1",
        },
        "sa1": {
            "AnonymousGraph": "***(*)*(*)*",
            "ArthorSubstructureOrder": "000700060100050002000037000000",
            "AtomBondCounts": "7,6",
            "CanonicalSmiles": "CC[C@@H](C)[C@H](O)Cl",
            "DegreeVector": "0,2,1,4",
            "ElementGraph": "CC[C@@H](C)[C@H](O)Cl",
            "ExtendedMurcko": "",
            "HetAtomProtomer": "CC[C@@H](C)[C@H]([O])Cl_1",
            "HetAtomTautomer": "CC[C@@H](C)[C@H]([O])Cl_1_0",
            "Mesomer": "CC[C@@H](C)[C@H](O)Cl_0",
            "MolFormula": "C5H11ClO",
            "MurckoScaffold": "",
            "NetCharge": "0",
            "RedoxPair": "CC[C@@H](C)[C@H](O)Cl",
            "Regioisomer": "*Cl.*O.CCC(C)C",
            "SmallWorldIndexBR": "B6R0",
            "SmallWorldIndexBRL": "B6R0L1",
            "cx_smiles": "CC[C@@H](C)[C@H](O)Cl",
            "inchi": "InChI=1S/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4-,5+/m1/s1",
            "inchi_KET_15T": "InChI=1/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4-,5+/m1/s1",
            "inchi_key": "ZXUZVOQOEXOEFU-UHNVWZDZSA-N",
            "inchikey_KET_15T": "ZXUZVOQOEXOEFU-UHNVWZDZNA-N",
            "noiso_smiles": "CCC(C)C(O)Cl",
            "smiles": "CC[C@@H](C)[C@H](O)Cl",
        },
        "sa2": {
            "AnonymousGraph": "***(*)*(*)*",
            "ArthorSubstructureOrder": "000700060100050002000037000000",
            "AtomBondCounts": "7,6",
            "CanonicalSmiles": "CC[C@@H](C)[C@@H](O)Cl",
            "DegreeVector": "0,2,1,4",
            "ElementGraph": "CC[C@@H](C)[C@@H](O)Cl",
            "ExtendedMurcko": "",
            "HetAtomProtomer": "CC[C@@H](C)[C@@H]([O])Cl_1",
            "HetAtomTautomer": "CC[C@@H](C)[C@@H]([O])Cl_1_0",
            "Mesomer": "CC[C@@H](C)[C@@H](O)Cl_0",
            "MolFormula": "C5H11ClO",
            "MurckoScaffold": "",
            "NetCharge": "0",
            "RedoxPair": "CC[C@@H](C)[C@@H](O)Cl",
            "Regioisomer": "*Cl.*O.CCC(C)C",
            "SmallWorldIndexBR": "B6R0",
            "SmallWorldIndexBRL": "B6R0L1",
            "cx_smiles": "CC[C@@H](C)[C@@H](O)Cl",
            "inchi": "InChI=1S/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4-,5-/m1/s1",
            "inchi_KET_15T": "InChI=1/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4-,5-/m1/s1",
            "inchi_key": "ZXUZVOQOEXOEFU-RFZPGFLSSA-N",
            "inchikey_KET_15T": "ZXUZVOQOEXOEFU-RFZPGFLSNA-N",
            "noiso_smiles": "CCC(C)C(O)Cl",
            "smiles": "CC[C@@H](C)[C@@H](O)Cl",
        },
        "sa3": {
            "AnonymousGraph": "***(*)*(*)*",
            "ArthorSubstructureOrder": "000700060100050002000037000000",
            "AtomBondCounts": "7,6",
            "CanonicalSmiles": "CC[C@@H](C)C(O)Cl",
            "DegreeVector": "0,2,1,4",
            "ElementGraph": "CC[C@@H](C)C(O)Cl",
            "ExtendedMurcko": "",
            "HetAtomProtomer": "CC[C@@H](C)C([O])Cl_1",
            "HetAtomTautomer": "CC[C@@H](C)C([O])Cl_1_0",
            "Mesomer": "CC[C@@H](C)C(O)Cl_0",
            "MolFormula": "C5H11ClO",
            "MurckoScaffold": "",
            "NetCharge": "0",
            "RedoxPair": "CC[C@@H](C)C(O)Cl",
            "Regioisomer": "*Cl.*O.CCC(C)C",
            "SmallWorldIndexBR": "B6R0",
            "SmallWorldIndexBRL": "B6R0L1",
            "cx_smiles": "CC[C@@H](C)C(O)Cl",
            "inchi": "InChI=1S/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4-,5?/m1/s1",
            "inchi_KET_15T": "InChI=1/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4-,5?/m1/s1",
            "inchi_key": "ZXUZVOQOEXOEFU-CNZKWPKMSA-N",
            "inchikey_KET_15T": "ZXUZVOQOEXOEFU-CNZKWPKMNA-N",
            "noiso_smiles": "CCC(C)C(O)Cl",
            "smiles": "CC[C@@H](C)C(O)Cl",
        },
        "sa4": {
            "AnonymousGraph": "***(*)*(*)*",
            "ArthorSubstructureOrder": "000700060100050002000037000000",
            "AtomBondCounts": "7,6",
            "CanonicalSmiles": "CC[C@H](C)[C@H](O)Cl",
            "DegreeVector": "0,2,1,4",
            "ElementGraph": "CC[C@H](C)[C@H](O)Cl",
            "ExtendedMurcko": "",
            "HetAtomProtomer": "CC[C@H](C)[C@H]([O])Cl_1",
            "HetAtomTautomer": "CC[C@H](C)[C@H]([O])Cl_1_0",
            "Mesomer": "CC[C@H](C)[C@H](O)Cl_0",
            "MolFormula": "C5H11ClO",
            "MurckoScaffold": "",
            "NetCharge": "0",
            "RedoxPair": "CC[C@H](C)[C@H](O)Cl",
            "Regioisomer": "*Cl.*O.CCC(C)C",
            "SmallWorldIndexBR": "B6R0",
            "SmallWorldIndexBRL": "B6R0L1",
            "cx_smiles": "CC[C@H](C)[C@H](O)Cl",
            "inchi": "InChI=1S/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4-,5-/m0/s1",
            "inchi_KET_15T": "InChI=1/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4-,5-/m0/s1",
            "inchi_key": "ZXUZVOQOEXOEFU-WHFBIAKZSA-N",
            "inchikey_KET_15T": "ZXUZVOQOEXOEFU-WHFBIAKZNA-N",
            "noiso_smiles": "CCC(C)C(O)Cl",
            "smiles": "CC[C@H](C)[C@H](O)Cl",
        },
        "sa5": {
            "AnonymousGraph": "***(*)*(*)*",
            "ArthorSubstructureOrder": "000700060100050002000037000000",
            "AtomBondCounts": "7,6",
            "CanonicalSmiles": "CC[C@H](C)[C@@H](O)Cl",
            "DegreeVector": "0,2,1,4",
            "ElementGraph": "CC[C@H](C)[C@@H](O)Cl",
            "ExtendedMurcko": "",
            "HetAtomProtomer": "CC[C@H](C)[C@@H]([O])Cl_1",
            "HetAtomTautomer": "CC[C@H](C)[C@@H]([O])Cl_1_0",
            "Mesomer": "CC[C@H](C)[C@@H](O)Cl_0",
            "MolFormula": "C5H11ClO",
            "MurckoScaffold": "",
            "NetCharge": "0",
            "RedoxPair": "CC[C@H](C)[C@@H](O)Cl",
            "Regioisomer": "*Cl.*O.CCC(C)C",
            "SmallWorldIndexBR": "B6R0",
            "SmallWorldIndexBRL": "B6R0L1",
            "cx_smiles": "CC[C@H](C)[C@@H](O)Cl",
            "inchi": "InChI=1S/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4-,5+/m0/s1",
            "inchi_KET_15T": "InChI=1/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4-,5+/m0/s1",
            "inchi_key": "ZXUZVOQOEXOEFU-CRCLSJGQSA-N",
            "inchikey_KET_15T": "ZXUZVOQOEXOEFU-CRCLSJGQNA-N",
            "noiso_smiles": "CCC(C)C(O)Cl",
            "smiles": "CC[C@H](C)[C@@H](O)Cl",
        },
        "sa6": {
            "AnonymousGraph": "***(*)*(*)*",
            "ArthorSubstructureOrder": "000700060100050002000037000000",
            "AtomBondCounts": "7,6",
            "CanonicalSmiles": "CC[C@H](C)C(O)Cl",
            "DegreeVector": "0,2,1,4",
            "ElementGraph": "CC[C@H](C)C(O)Cl",
            "ExtendedMurcko": "",
            "HetAtomProtomer": "CC[C@H](C)C([O])Cl_1",
            "HetAtomTautomer": "CC[C@H](C)C([O])Cl_1_0",
            "Mesomer": "CC[C@H](C)C(O)Cl_0",
            "MolFormula": "C5H11ClO",
            "MurckoScaffold": "",
            "NetCharge": "0",
            "RedoxPair": "CC[C@H](C)C(O)Cl",
            "Regioisomer": "*Cl.*O.CCC(C)C",
            "SmallWorldIndexBR": "B6R0",
            "SmallWorldIndexBRL": "B6R0L1",
            "cx_smiles": "CC[C@H](C)C(O)Cl",
            "inchi": "InChI=1S/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4-,5?/m0/s1",
            "inchi_KET_15T": "InChI=1/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4-,5?/m0/s1",
            "inchi_key": "ZXUZVOQOEXOEFU-ROLXFIACSA-N",
            "inchikey_KET_15T": "ZXUZVOQOEXOEFU-ROLXFIACNA-N",
            "noiso_smiles": "CCC(C)C(O)Cl",
            "smiles": "CC[C@H](C)C(O)Cl",
        },
        "sa7": {
            "AnonymousGraph": "***(*)*(*)*",
            "ArthorSubstructureOrder": "000700060100050002000037000000",
            "AtomBondCounts": "7,6",
            "CanonicalSmiles": "CCC(C)[C@H](O)Cl",
            "DegreeVector": "0,2,1,4",
            "ElementGraph": "CCC(C)[C@H](O)Cl",
            "ExtendedMurcko": "",
            "HetAtomProtomer": "CCC(C)[C@H]([O])Cl_1",
            "HetAtomTautomer": "CCC(C)[C@H]([O])Cl_1_0",
            "Mesomer": "CCC(C)[C@H](O)Cl_0",
            "MolFormula": "C5H11ClO",
            "MurckoScaffold": "",
            "NetCharge": "0",
            "RedoxPair": "CCC(C)[C@H](O)Cl",
            "Regioisomer": "*Cl.*O.CCC(C)C",
            "SmallWorldIndexBR": "B6R0",
            "SmallWorldIndexBRL": "B6R0L1",
            "cx_smiles": "CCC(C)[C@H](O)Cl",
            "inchi": "InChI=1S/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4?,5-/m0/s1",
            "inchi_KET_15T": "InChI=1/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4?,5-/m0/s1",
            "inchi_key": "ZXUZVOQOEXOEFU-AKGZTFGVSA-N",
            "inchikey_KET_15T": "ZXUZVOQOEXOEFU-AKGZTFGVNA-N",
            "noiso_smiles": "CCC(C)C(O)Cl",
            "smiles": "CCC(C)[C@H](O)Cl",
        },
        "sa8": {
            "AnonymousGraph": "***(*)*(*)*",
            "ArthorSubstructureOrder": "000700060100050002000037000000",
            "AtomBondCounts": "7,6",
            "CanonicalSmiles": "CCC(C)[C@@H](O)Cl",
            "DegreeVector": "0,2,1,4",
            "ElementGraph": "CCC(C)[C@@H](O)Cl",
            "ExtendedMurcko": "",
            "HetAtomProtomer": "CCC(C)[C@@H]([O])Cl_1",
            "HetAtomTautomer": "CCC(C)[C@@H]([O])Cl_1_0",
            "Mesomer": "CCC(C)[C@@H](O)Cl_0",
            "MolFormula": "C5H11ClO",
            "MurckoScaffold": "",
            "NetCharge": "0",
            "RedoxPair": "CCC(C)[C@@H](O)Cl",
            "Regioisomer": "*Cl.*O.CCC(C)C",
            "SmallWorldIndexBR": "B6R0",
            "SmallWorldIndexBRL": "B6R0L1",
            "cx_smiles": "CCC(C)[C@@H](O)Cl",
            "inchi": "InChI=1S/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4?,5-/m1/s1",
            "inchi_KET_15T": "InChI=1/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4?,5-/m1/s1",
            "inchi_key": "ZXUZVOQOEXOEFU-BRJRFNKRSA-N",
            "inchikey_KET_15T": "ZXUZVOQOEXOEFU-BRJRFNKRNA-N",
            "noiso_smiles": "CCC(C)C(O)Cl",
            "smiles": "CCC(C)[C@@H](O)Cl",
        },
        "sa9": {
            "AnonymousGraph": "***(*)*(*)*",
            "ArthorSubstructureOrder": "000700060100050002000037000000",
            "AtomBondCounts": "7,6",
            "CanonicalSmiles": "CCC(C)C(O)Cl",
            "DegreeVector": "0,2,1,4",
            "ElementGraph": "CCC(C)C(O)Cl",
            "ExtendedMurcko": "",
            "HetAtomProtomer": "CCC(C)C([O])Cl_1",
            "HetAtomTautomer": "CCC(C)C([O])Cl_1_0",
            "Mesomer": "CCC(C)C(O)Cl_0",
            "MolFormula": "C5H11ClO",
            "MurckoScaffold": "",
            "NetCharge": "0",
            "RedoxPair": "CCC(C)C(O)Cl",
            "Regioisomer": "*Cl.*O.CCC(C)C",
            "SmallWorldIndexBR": "B6R0",
            "SmallWorldIndexBRL": "B6R0L1",
            "cx_smiles": "CCC(C)C(O)Cl",
            "inchi": "InChI=1S/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3",
            "inchi_KET_15T": "InChI=1/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3",
            "inchi_key": "ZXUZVOQOEXOEFU-UHFFFAOYSA-N",
            "inchikey_KET_15T": "ZXUZVOQOEXOEFU-UHFFFAOYNA-N",
            "noiso_smiles": "CCC(C)C(O)Cl",
            "smiles": "CCC(C)C(O)Cl",
        },
        "ta1": {
            "AnonymousGraph": "**1****1",
            "ArthorSubstructureOrder": "000600060100040002000027000000",
            "AtomBondCounts": "6,6",
            "CanonicalSmiles": "OC1=NCCC1",
            "DegreeVector": "0,1,4,1",
            "ElementGraph": "OC1CCCN1",
            "ExtendedMurcko": "*C1=NCCC1",
            "HetAtomProtomer": "[O][C]1CCC[N]1_1",
            "HetAtomTautomer": "[O][C]1CCC[N]1_1_0",
            "Mesomer": "O[C]1CCC[N]1_0",
            "MolFormula": "C4H7NO",
            "MurckoScaffold": "C1=NCCC1",
            "NetCharge": "0",
            "RedoxPair": "O[C]1CCC[N]1",
            "Regioisomer": "*O.C1=NCCC1",
            "SmallWorldIndexBR": "B6R1",
            "SmallWorldIndexBRL": "B6R1L4",
            "cx_smiles": "OC1=NCCC1",
            "inchi": "InChI=1S/C4H7NO/c6-4-2-1-3-5-4/h1-3H2,(H,5,6)",
            "inchi_KET_15T": "InChI=1/C4H7NO/c6-4-2-1-3-5-4/h1,3H2,(H3,2,5,6)",
            "inchi_key": "HNJBEVLQSNELDL-UHFFFAOYSA-N",
            "inchikey_KET_15T": "DXJWFBGIRPOVOQ-UHFFFAOYNA-N",
            "noiso_smiles": "OC1=NCCC1",
            "smiles": "OC1=NCCC1",
        },
        "ta2": {
            "AnonymousGraph": "**1****1",
            "ArthorSubstructureOrder": "000600060100040002000027000000",
            "AtomBondCounts": "6,6",
            "CanonicalSmiles": "O=C1CCCN1",
            "DegreeVector": "0,1,4,1",
            "ElementGraph": "OC1CCCN1",
            "ExtendedMurcko": "*=C1CCCN1",
            "HetAtomProtomer": "[O][C]1CCC[N]1_1",
            "HetAtomTautomer": "[O][C]1CCC[N]1_1_0",
            "Mesomer": "[O][C]1CCCN1_0",
            "MolFormula": "C4H7NO",
            "MurckoScaffold": "C1CCNC1",
            "NetCharge": "0",
            "RedoxPair": "[O][C]1CCCN1",
            "Regioisomer": "O=C1CCCN1",
            "SmallWorldIndexBR": "B6R1",
            "SmallWorldIndexBRL": "B6R1L4",
            "cx_smiles": "O=C1CCCN1",
            "inchi": "InChI=1S/C4H7NO/c6-4-2-1-3-5-4/h1-3H2,(H,5,6)",
            "inchi_KET_15T": "InChI=1/C4H7NO/c6-4-2-1-3-5-4/h1,3H2,(H3,2,5,6)",
            "inchi_key": "HNJBEVLQSNELDL-UHFFFAOYSA-N",
            "inchikey_KET_15T": "DXJWFBGIRPOVOQ-UHFFFAOYNA-N",
            "noiso_smiles": "O=C1CCCN1",
            "smiles": "O=C1CCCN1",
        },
        "tb1": {
            "AnonymousGraph": "***(*)*1*(*)***(*)*1*",
            "ArthorSubstructureOrder": "000d000d01000a0003000053000000",
            "AtomBondCounts": "13,13",
            "CanonicalSmiles": "C/N=C(\\C)C1C(=O)CCC(C)C1=O",
            "DegreeVector": "0,5,3,5",
            "ElementGraph": "CNC(C)C1C(O)CCC(C)C1O",
            "ExtendedMurcko": "*C1CCC(=*)C(*)C1=*",
            "HetAtomProtomer": "C[N][C](C)C1[C]([O])CCC(C)[C]1[O]_0",
            "HetAtomTautomer": "C[N][C](C)C1[C]([O])CCC(C)[C]1[O]_0_0",
            "Mesomer": "C[N][C](C)C1[C]([O])CCC(C)[C]1[O]_0",
            "MolFormula": "C10H15NO2",
            "MurckoScaffold": "C1CCCCC1",
            "NetCharge": "0",
            "RedoxPair": "C[N][C](C)C1[C]([O])CCC(C)[C]1[O]",
            "Regioisomer": "*C.*N=C(*)C.C.O=C1CCCC(=O)C1",
            "SmallWorldIndexBR": "B13R1",
            "SmallWorldIndexBRL": "B13R1L3",
            "cx_smiles": "C/N=C(\\C)C1C(=O)CCC(C)C1=O",
            "inchi": "InChI=1S/C10H15NO2/c1-6-4-5-8(12)9(10(6)13)7(2)11-3/h6,9H,4-5H2,1-3H3/b11-7+",
            "inchi_KET_15T": "InChI=1/C10H15NO2/c1-6-4-5-8(12)9(10(6)13)7(2)11-3/h4H2,1-3H3,(H4,5,6,9,11,12,13)",
            "inchi_key": "LFFJNZFRBPZXBS-YRNVUSSQSA-N",
            "inchikey_KET_15T": "SJQNTMZZZVTREP-UHFFFAOYNA-N",
            "noiso_smiles": "CN=C(C)C1C(=O)CCC(C)C1=O",
            "smiles": "C/N=C(\\C)C1C(=O)CCC(C)C1=O",
        },
        "tb2": {
            "AnonymousGraph": "***(*)*1*(*)***(*)*1*",
            "ArthorSubstructureOrder": "000d000d01000a0003000053000000",
            "AtomBondCounts": "13,13",
            "CanonicalSmiles": "C/N=C(\\C)C1=C(O)CCC(C)C1=O",
            "DegreeVector": "0,5,3,5",
            "ElementGraph": "CNC(C)C1C(O)CCC(C)C1O",
            "ExtendedMurcko": "*C1=C(*)C(=*)C(*)CC1",
            "HetAtomProtomer": "C[N][C](C)[C]1[C]([O])CCC(C)[C]1[O]_1",
            "HetAtomTautomer": "C[N][C](C)[C]1[C]([O])CCC(C)[C]1[O]_1_0",
            "Mesomer": "C[N][C](C)[C]1[C](O)CCC(C)[C]1[O]_0",
            "MolFormula": "C10H15NO2",
            "MurckoScaffold": "C1=CCCCC1",
            "NetCharge": "0",
            "RedoxPair": "C[N][C](C)[C]1[C](O)CCC(C)[C]1[O]",
            "Regioisomer": "*C.*N=C(*)C.*O.C.O=C1C=CCCC1",
            "SmallWorldIndexBR": "B13R1",
            "SmallWorldIndexBRL": "B13R1L3",
            "cx_smiles": "C/N=C(\\C)C1=C(O)CCC(C)C1=O",
            "inchi": "InChI=1S/C10H15NO2/c1-6-4-5-8(12)9(10(6)13)7(2)11-3/h6,12H,4-5H2,1-3H3/b11-7+",
            "inchi_KET_15T": "InChI=1/C10H15NO2/c1-6-4-5-8(12)9(10(6)13)7(2)11-3/h4H2,1-3H3,(H4,5,6,9,11,12,13)",
            "inchi_key": "MCKSLMLPGJYCNP-YRNVUSSQSA-N",
            "inchikey_KET_15T": "SJQNTMZZZVTREP-UHFFFAOYNA-N",
            "noiso_smiles": "CN=C(C)C1=C(O)CCC(C)C1=O",
            "smiles": "C/N=C(\\C)C1=C(O)CCC(C)C1=O",
        },
        "tb3": {
            "AnonymousGraph": "***(*)*1*(*)***(*)*1*",
            "ArthorSubstructureOrder": "000d000d01000a0003000053000000",
            "AtomBondCounts": "13,13",
            "CanonicalSmiles": "C/N=C(\\C)C1=C(O)C(C)CCC1=O",
            "DegreeVector": "0,5,3,5",
            "ElementGraph": "CNC(C)C1C(O)CCC(C)C1O",
            "ExtendedMurcko": "*C1=C(*)C(*)CCC1=*",
            "HetAtomProtomer": "C[N][C](C)[C]1[C]([O])CCC(C)[C]1[O]_1",
            "HetAtomTautomer": "C[N][C](C)[C]1[C]([O])CCC(C)[C]1[O]_1_0",
            "Mesomer": "C[N][C](C)[C]1[C]([O])CCC(C)[C]1O_0",
            "MolFormula": "C10H15NO2",
            "MurckoScaffold": "C1=CCCCC1",
            "NetCharge": "0",
            "RedoxPair": "C[N][C](C)[C]1[C]([O])CCC(C)[C]1O",
            "Regioisomer": "*C.*N=C(*)C.*O.C.O=C1C=CCCC1",
            "SmallWorldIndexBR": "B13R1",
            "SmallWorldIndexBRL": "B13R1L3",
            "cx_smiles": "C/N=C(\\C)C1=C(O)C(C)CCC1=O",
            "inchi": "InChI=1S/C10H15NO2/c1-6-4-5-8(12)9(10(6)13)7(2)11-3/h6,13H,4-5H2,1-3H3/b11-7+",
            "inchi_KET_15T": "InChI=1/C10H15NO2/c1-6-4-5-8(12)9(10(6)13)7(2)11-3/h4H2,1-3H3,(H4,5,6,9,11,12,13)",
            "inchi_key": "LQHZFYQSMISUAH-YRNVUSSQSA-N",
            "inchikey_KET_15T": "SJQNTMZZZVTREP-UHFFFAOYNA-N",
            "noiso_smiles": "CN=C(C)C1=C(O)C(C)CCC1=O",
            "smiles": "C/N=C(\\C)C1=C(O)C(C)CCC1=O",
        },
        "tc1": {
            "AnonymousGraph": "**(*)*1*(*)***(*)*1*",
            "ArthorSubstructureOrder": "000c000c010009000300004e000000",
            "AtomBondCounts": "12,12",
            "CanonicalSmiles": "CC(=O)C1=C(O)C(C)CCC1=O",
            "DegreeVector": "0,5,2,5",
            "ElementGraph": "CC(O)C1C(O)CCC(C)C1O",
            "ExtendedMurcko": "*C1=C(*)C(*)CCC1=*",
            "HetAtomProtomer": "C[C]([O])[C]1[C]([O])CCC(C)[C]1[O]_1",
            "HetAtomTautomer": "C[C]([O])[C]1[C]([O])CCC(C)[C]1[O]_1_0",
            "Mesomer": "C[C]([O])[C]1[C]([O])CCC(C)[C]1O_0",
            "MolFormula": "C9H12O3",
            "MurckoScaffold": "C1=CCCCC1",
            "NetCharge": "0",
            "RedoxPair": "C[C]([O])[C]1[C]([O])CCC(C)[C]1O",
            "Regioisomer": "*C.*C(C)=O.*O.O=C1C=CCCC1",
            "SmallWorldIndexBR": "B12R1",
            "SmallWorldIndexBRL": "B12R1L2",
            "cx_smiles": "CC(=O)C1=C(O)C(C)CCC1=O",
            "inchi": "InChI=1S/C9H12O3/c1-5-3-4-7(11)8(6(2)10)9(5)12/h5,12H,3-4H2,1-2H3",
            "inchi_KET_15T": "InChI=1/C9H12O3/c1-5-3-4-7(11)8(6(2)10)9(5)12/h3H2,1-2H3,(H4,4,5,8,10,11,12)",
            "inchi_key": "JKVFZUSRZCBLOO-UHFFFAOYSA-N",
            "inchikey_KET_15T": "XCMTZZNIZOTRIL-UHFFFAOYNA-N",
            "noiso_smiles": "CC(=O)C1=C(O)C(C)CCC1=O",
            "smiles": "CC(=O)C1=C(O)C(C)CCC1=O",
        },
        "tc2": {
            "AnonymousGraph": "**(*)*1*(*)***(*)*1*",
            "ArthorSubstructureOrder": "000c000c010009000300004e000000",
            "AtomBondCounts": "12,12",
            "CanonicalSmiles": "CC(=O)C1C(=O)CCC(C)C1=O",
            "DegreeVector": "0,5,2,5",
            "ElementGraph": "CC(O)C1C(O)CCC(C)C1O",
            "ExtendedMurcko": "*C1CCC(=*)C(*)C1=*",
            "HetAtomProtomer": "C[C]([O])C1[C]([O])CCC(C)[C]1[O]_0",
            "HetAtomTautomer": "C[C]([O])C1[C]([O])CCC(C)[C]1[O]_0_0",
            "Mesomer": "C[C]([O])C1[C]([O])CCC(C)[C]1[O]_0",
            "MolFormula": "C9H12O3",
            "MurckoScaffold": "C1CCCCC1",
            "NetCharge": "0",
            "RedoxPair": "C[C]([O])C1[C]([O])CCC(C)[C]1[O]",
            "Regioisomer": "*C.*C(C)=O.O=C1CCCC(=O)C1",
            "SmallWorldIndexBR": "B12R1",
            "SmallWorldIndexBRL": "B12R1L2",
            "cx_smiles": "CC(=O)C1C(=O)CCC(C)C1=O",
            "inchi": "InChI=1S/C9H12O3/c1-5-3-4-7(11)8(6(2)10)9(5)12/h5,8H,3-4H2,1-2H3",
            "inchi_KET_15T": "InChI=1/C9H12O3/c1-5-3-4-7(11)8(6(2)10)9(5)12/h3H2,1-2H3,(H4,4,5,8,10,11,12)",
            "inchi_key": "MMPTYALDYQTEML-UHFFFAOYSA-N",
            "inchikey_KET_15T": "XCMTZZNIZOTRIL-UHFFFAOYNA-N",
            "noiso_smiles": "CC(=O)C1C(=O)CCC(C)C1=O",
            "smiles": "CC(=O)C1C(=O)CCC(C)C1=O",
        },
        "tc3": {
            "AnonymousGraph": "**(*)*1*(*)***(*)*1*",
            "ArthorSubstructureOrder": "000c000c010009000300004e000000",
            "AtomBondCounts": "12,12",
            "CanonicalSmiles": "CC(=O)C1=C(O)CCC(C)C1=O",
            "DegreeVector": "0,5,2,5",
            "ElementGraph": "CC(O)C1C(O)CCC(C)C1O",
            "ExtendedMurcko": "*C1=C(*)C(=*)C(*)CC1",
            "HetAtomProtomer": "C[C]([O])[C]1[C]([O])CCC(C)[C]1[O]_1",
            "HetAtomTautomer": "C[C]([O])[C]1[C]([O])CCC(C)[C]1[O]_1_0",
            "Mesomer": "C[C]([O])[C]1[C](O)CCC(C)[C]1[O]_0",
            "MolFormula": "C9H12O3",
            "MurckoScaffold": "C1=CCCCC1",
            "NetCharge": "0",
            "RedoxPair": "C[C]([O])[C]1[C](O)CCC(C)[C]1[O]",
            "Regioisomer": "*C.*C(C)=O.*O.O=C1C=CCCC1",
            "SmallWorldIndexBR": "B12R1",
            "SmallWorldIndexBRL": "B12R1L2",
            "cx_smiles": "CC(=O)C1=C(O)CCC(C)C1=O",
            "inchi": "InChI=1S/C9H12O3/c1-5-3-4-7(11)8(6(2)10)9(5)12/h5,11H,3-4H2,1-2H3",
            "inchi_KET_15T": "InChI=1/C9H12O3/c1-5-3-4-7(11)8(6(2)10)9(5)12/h3H2,1-2H3,(H4,4,5,8,10,11,12)",
            "inchi_key": "UYIJIQGWGXARDH-UHFFFAOYSA-N",
            "inchikey_KET_15T": "XCMTZZNIZOTRIL-UHFFFAOYNA-N",
            "noiso_smiles": "CC(=O)C1=C(O)CCC(C)C1=O",
            "smiles": "CC(=O)C1=C(O)CCC(C)C1=O",
        },
        "tc4": {
            "AnonymousGraph": "**1***(*)*(*(*)*)*1*",
            "ArthorSubstructureOrder": "000c000c010009000300004e000000",
            "AtomBondCounts": "12,12",
            "CanonicalSmiles": "C/C(O)=C1\\C(=O)CCC(C)C1=O",
            "DegreeVector": "0,5,2,5",
            "ElementGraph": "CC1CCC(O)C(C(C)O)C1O",
            "ExtendedMurcko": "*C1CCC(=*)C(=*)C1=*",
            "HetAtomProtomer": "C[C]([O])[C]1[C]([O])CCC(C)[C]1[O]_1",
            "HetAtomTautomer": "C[C]([O])[C]1[C]([O])CCC(C)[C]1[O]_1_0",
            "Mesomer": "C[C](O)[C]1[C]([O])CCC(C)[C]1[O]_0",
            "MolFormula": "C9H12O3",
            "MurckoScaffold": "C1CCCCC1",
            "NetCharge": "0",
            "RedoxPair": "C[C](O)[C]1[C]([O])CCC(C)[C]1[O]",
            "Regioisomer": "*C.CC(O)=C1C(=O)CCCC1=O",
            "SmallWorldIndexBR": "B12R1",
            "SmallWorldIndexBRL": "B12R1L2",
            "cx_smiles": "C/C(O)=C1\\C(=O)CCC(C)C1=O",
            "inchi": "InChI=1S/C9H12O3/c1-5-3-4-7(11)8(6(2)10)9(5)12/h5,10H,3-4H2,1-2H3/b8-6-",
            "inchi_KET_15T": "InChI=1/C9H12O3/c1-5-3-4-7(11)8(6(2)10)9(5)12/h3H2,1-2H3,(H4,4,5,8,10,11,12)",
            "inchi_key": "LTGVOMHYBZTAIZ-VURMDHGXSA-N",
            "inchikey_KET_15T": "XCMTZZNIZOTRIL-UHFFFAOYNA-N",
            "noiso_smiles": "CC(O)=C1C(=O)CCC(C)C1=O",
            "smiles": "C/C(O)=C1\\C(=O)CCC(C)C1=O",
        },
    }
    calculated = {}
    for x in examples:
        rdmol = Chem.MolFromSmiles(x.get("smiles"))
        hash_map = calculate_molecular_hash_values(rdmol=rdmol)
        name = x.get("name")
        calculated[name] = hash_map
        print()
        import pprint

        pprint.pprint(calculated)
    # dumb testing
    for k, v in calculated.items():
        assert v == reference.get(k)
    # clever testing


# def test_draw_mol():
#    smiles = 'CNC(C)=O'
#    fname = 'MOLECULE.png'
#    draw_mol(smiles, fname)
#    assert os.path.exists(fname)
#    os.remove(fname)


# def test_draw_reaction():
#    smiles = 'CN.CC(O)=O>O>CNC(C)=O'
#    fname = 'REACTION.png'
#    draw_reaction(smiles, fname)
#    assert os.path.exists(fname)
#    os.remove(fname)
# Test to prove the incompatibility of count vector fingerprints with the similarity syntax
# 'DataStructs.FingerprintSimilarity'
#
# def test_similarity_with_count_fp():
#     smiles = 'CNC(C)=O'
#     rdmol = rdmol_from_string(smiles, inp_fmt='smiles')
#
#     # Morgan fp: id GetCountFingerprint is used, an error is raised; otherwise two identical objects are recognised
#     # as such
#     fp_morgan_count = compute_mol_fingerprint(rdmol, 'morgan', count_fp_vector=True)
#     fp_morgan_count2 = compute_mol_fingerprint(rdmol, 'morgan', count_fp_vector=True)
#     with pytest.raises(AttributeError) as ke:
#         similarity = compute_similarity(fp_morgan_count, fp_morgan_count2, similarity_name='tanimoto')
#     assert "AttributeError" in str(ke.type)
#
#     fp_morgan = compute_mol_fingerprint(rdmol, 'morgan', count_fp_vector=False)
#     fp_morgan2 = compute_mol_fingerprint(rdmol, 'morgan', count_fp_vector=False)
#     assert compute_similarity(fp_morgan, fp_morgan2, similarity_name='tanimoto') == 1.0
#
#     # RDKit fp: id GetCountFingerprint is used, an error is raised; otherwise two identical objects are recognised
#     # as such
#     fp_rdkit_count = compute_mol_fingerprint(rdmol, 'rdkit', count_fp_vector=True)
#     fp_rdkit_count2 = compute_mol_fingerprint(rdmol, 'rdkit', count_fp_vector=True)
#     with pytest.raises(AttributeError) as ke:
#         similarity = compute_similarity(fp_rdkit_count, fp_rdkit_count2, similarity_name='tanimoto')
#     assert "AttributeError" in str(ke.type)
#
#     fp_rdkit = compute_mol_fingerprint(rdmol, 'rdkit', count_fp_vector=False)
#     fp_rdkit2 = compute_mol_fingerprint(rdmol, 'rdkit', count_fp_vector=False)
#     assert compute_similarity(fp_rdkit, fp_rdkit2, similarity_name='tanimoto') == 1.0
#
#     # Topological fp: if GetCountFingerprint is used, an error is raised; otherwise two identical objects
#     # are recognised as such

#     fp_topological_count = compute_mol_fingerprint(rdmol, 'topological', count_fp_vector=True)
#     fp_topological_count2 = compute_mol_fingerprint(rdmol, 'topological', count_fp_vector=True)
#     with pytest.raises(AttributeError) as ke:
#         similarity = compute_similarity(fp_topological_count, fp_topological_count2, similarity_name='tanimoto')
#     assert "AttributeError" in str(ke.type)
#
#     fp_topological = compute_mol_fingerprint(rdmol, 'topological', count_fp_vector=False)
#     fp_topological2 = compute_mol_fingerprint(rdmol, 'topological', count_fp_vector=False)
#     assert compute_similarity(fp_topological, fp_topological2, similarity_name='tanimoto') == 1.0
