import unittest

import pytest
from rdkit import Chem
from rdkit.Chem import rdChemReactions

import linchemin.cheminfo.functions as cif
from linchemin.cheminfo.constructors import ChemicalEquationConstructor
from linchemin.cheminfo.functions import rdkit as rdkit


@pytest.fixture
def mols_w_stereo():
    return {
        # mol with enhanced stereochemistry
        0: {"smiles": "C[C@@H](F)[C@H](C)Cl |&1:1,5,r|", "enhanced_stereo": True},
        # mol with canonical stereochemistry
        1: {"smiles": "C[C@H](Cl)[C@H](F)C", "enhanced_stereo": False},
        # mol with enhanced stereochemistry AND atom mapping
        2: {
            "smiles": "[CH3:6][C@@H:4]([F:5])[C@H:2]([CH3:1])[Cl:3] |&1:1,5,r|",
            "enhanced_stereo": True,
        },
        # flat molecule
        3: {"smiles": "CC(F)C(C)Cl", "enhanced_stereo": False},
    }


def test_build_rdrxn_from_smiles():
    kwa = {"input_string": "CN.CC(O)=O>O>CNC(C)=O", "inp_fmt": "smiles"}
    rdrxn = cif.rdrxn_from_string(**kwa)
    assert rdrxn is not None


def test_build_rdrxn_from_smarts():
    kwa = {"input_string": "[C:1]=[O,N:2]>>[C:1][*:2]", "inp_fmt": "smarts"}
    rdrxn = cif.rdrxn_from_string(**kwa)
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
    kwa = {"input_string": rxnblockv2k, "inp_fmt": "rxn_block"}
    rdrxn = cif.rdrxn_from_string(**kwa)
    assert rdrxn is not None


def test_build_smiles_from_rdrxn():
    kwa = {"input_string": "CN.CC(O)=O>O>CNC(C)=O", "inp_fmt": "smiles"}
    rdrxn = cif.rdrxn_from_string(**kwa)

    kwa = {"rdrxn": rdrxn, "out_fmt": "smiles"}
    string = cif.rdrxn_to_string(**kwa)
    assert string is not None


def test_build_rxnblock_from_rdrxn():
    inp = {"input_string": "CN.CC(O)=O>O>CNC(C)=O", "inp_fmt": "smiles"}
    rdrxn = cif.rdrxn_from_string(**inp)

    kwa = {"rdrxn": rdrxn, "out_fmt": "rxn_blockV2K"}
    string = cif.rdrxn_to_string(**kwa)
    assert string is not None


def test_build_smarts_from_rdrxn():
    kwa = {"input_string": "[C:1]=[O,N:2]>>[C:1][*:2]", "inp_fmt": "smarts"}
    rdrxn = cif.rdrxn_from_string(**kwa)

    kwa = {"rdrxn": rdrxn, "out_fmt": "smarts"}
    string = cif.rdrxn_to_string(**kwa)

    assert string is not None


def test_rdmol_serialization():
    # serialize a rdmol to inject it into a database
    # https://github.com/rdkit/rdkit/issues/1999
    # https://stackoverflow.com/questions/15659195/insert-byte-string-into-mongodb
    smiles_string = "c1ncncc1C(=O)[O-]"
    rdmol1 = cif.Chem.MolFromSmiles(smiles_string)
    smiles_canonical1 = cif.Chem.MolToSmiles(rdmol1)
    rdmol_bstr = cif.rdmol_to_bstr(rdmol=rdmol1)
    rdmol2 = cif.bstr_to_rdmol(rdmol_bstr=rdmol_bstr)
    smiles_canonical2 = cif.Chem.MolToSmiles(rdmol2)
    assert smiles_canonical1 == smiles_canonical2


def test_mapped_molecule():
    m1 = rdkit.Chem.MolFromSmiles("c1([C@H](C)CC)cccc2ccccc12")
    m2 = rdkit.Chem.MolFromSmiles("Cl[C:2]([CH3:1])=[O:3]")
    assert cif.is_mapped_molecule(m1) is False
    assert cif.is_mapped_molecule(m2) is True


def test_enhanced_stereo(mols_w_stereo):
    for item in mols_w_stereo.values():
        rdmol = cif.rdmol_from_string(item["smiles"], "smiles")
        assert cif.has_enhanced_stereo(rdmol) == item["enhanced_stereo"]


def test_mapped_rdrxn():
    smiles1 = "CN.CC(O)=O>O>CNC(C)=O"
    smiles2 = "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]"
    rdrxn1 = cif.rdrxn_from_string(input_string=smiles1, inp_fmt="smiles")
    rdrxn2 = cif.rdrxn_from_string(input_string=smiles2, inp_fmt="smiles")
    assert cif.has_mapped_products(rdrxn1) is False
    assert cif.has_mapped_products(rdrxn2) is True


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
    m1_canonical_order = cif.get_canonical_order(m1)
    m2_canonical_order = cif.get_canonical_order(m2)
    assert m1_canonical_order != m2_canonical_order

    m1_can, log = cif.canonicalize_rdmol(rdmol=m1)
    m2_can, log = cif.canonicalize_rdmol(rdmol=m2)

    assert rdkit.Chem.MolToSmiles(m1_can) == rdkit.Chem.MolToSmiles(m2_can)
    assert rdkit.Chem.MolToSmiles(m1_can, canonical=False) == rdkit.Chem.MolToSmiles(
        m2_can, canonical=False
    )
    m1_can_canonical_order = cif.get_canonical_order(m1_can)
    m2_can_canonical_order = cif.get_canonical_order(m2_can)
    assert m1_can_canonical_order == m2_can_canonical_order


def test_run_rdchiral_wrapper_on_list_of_reactions():
    list_input = [
        {
            "reaction_id": 1,
            "reaction_string": "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "inp_fmt": "smiles",
        },
        {
            "reaction_id": 1,
            "reaction_string": "[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]>>[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]",
            "inp_fmt": "smiles",
        },
    ]

    list_output = [cif.rdchiral_extract_template(**item) for item in list_input]
    expected_list_output = [
        {
            "products": "[C;D1;H3:1]-[C;H0;D3;+0:2](=[O;D1;H0:3])-[NH;D2;+0:6]-[C;D1;H3:5].[OH2;D0;+0:4]",
            "reactants": "[C;D1;H3:1]-[C;H0;D3;+0:2](=[O;D1;H0:3])-[OH;D1;+0:4].[C;D1;H3:5]-[NH2;D1;+0:6]",
            "reaction_smarts": "[C;D1;H3:1]-[C;H0;D3;+0:2](=[O;D1;H0:3])-[NH;D2;+0:6]-[C;D1;H3:5].[OH2;D0;+0:4]>>[C;D1;H3:1]-[C;H0;D3;+0:2](=[O;D1;H0:3])-[OH;D1;+0:4].[C;D1;H3:5]-[NH2;D1;+0:6]",
            "intra_only": False,
            "dimer_only": False,
            "reaction_id": 1,
            "necessary_reagent": "",
        },
        {
            "products": "[C;D1;H3:1]-[C;H0;D3;+0:2](=[O;D1;H0:3])-[OH;D1;+0:6].[C;D1;H3:5]-[NH2;D1;+0:4]",
            "reactants": "[C;D1;H3:1]-[C;H0;D3;+0:2](=[O;D1;H0:3])-[NH;D2;+0:4]-[C;D1;H3:5].[OH2;D0;+0:6]",
            "reaction_smarts": "[C;D1;H3:1]-[C;H0;D3;+0:2](=[O;D1;H0:3])-[OH;D1;+0:6].[C;D1;H3:5]-[NH2;D1;+0:4]>>[C;D1;H3:1]-[C;H0;D3;+0:2](=[O;D1;H0:3])-[NH;D2;+0:4]-[C;D1;H3:5].[OH2;D0;+0:6]",
            "intra_only": False,
            "dimer_only": False,
            "reaction_id": 1,
            "necessary_reagent": "",
        },
    ]

    for a, b in zip(list_output, expected_list_output):
        assert a == b


def test_rdrxn_role_reassignment():
    test_set = [
        {
            "name": "rnx_1",
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {
                "reactants": ["[CH3:1][C:2]([OH:3])=[O:4]", "[NH2:5][CH3:6]"],
                "reagents": [],
                "products": ["[CH3:6][NH:5][C:2]([CH3:1])=[O:4]", "[OH2:3]"],
            },
        },
        {
            "name": "rnx_2",
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4]>[CH3:6][NH2:5]>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {
                "reactants": ["[CH3:1][C:2]([OH:3])=[O:4]", "[NH2:5][CH3:6]"],
                "reagents": [],
                "products": ["[CH3:6][NH:5][C:2]([CH3:1])=[O:4]", "[OH2:3]"],
            },
        },
        {
            "name": "rnx_3",
            "smiles": "[CH3:6][NH2:5]>[CH3:1][C:2]([OH:3])=[O:4]>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {
                "reactants": ["[NH2:5][CH3:6]", "[CH3:1][C:2]([OH:3])=[O:4]"],
                "reagents": [],
                "products": ["[CH3:6][NH:5][C:2]([CH3:1])=[O:4]", "[OH2:3]"],
            },
        },
        {
            "name": "rnx_4",
            "smiles": ">[CH3:6][NH2:5].[CH3:1][C:2]([OH:3])=[O:4]>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {
                "reactants": ["[NH2:5][CH3:6]", "[CH3:1][C:2]([OH:3])=[O:4]"],
                "reagents": [],
                "products": ["[CH3:6][NH:5][C:2]([CH3:1])=[O:4]", "[OH2:3]"],
            },
        },
        {
            "name": "rnx_5",
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>C>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {
                "reactants": ["[CH3:1][C:2]([OH:3])=[O:4]", "[NH2:5][CH3:6]"],
                "reagents": ["C"],
                "products": ["[CH3:6][NH:5][C:2]([CH3:1])=[O:4]", "[OH2:3]"],
            },
        },
        {
            "name": "rnx_6",
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>CN>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {
                "reactants": ["[CH3:1][C:2]([OH:3])=[O:4]", "[NH2:5][CH3:6]"],
                "reagents": ["CN"],
                "products": ["[CH3:6][NH:5][C:2]([CH3:1])=[O:4]", "[OH2:3]"],
            },
        },
        {
            "name": "rnx_7",
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5].CN>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {
                "reactants": ["[CH3:1][C:2]([OH:3])=[O:4]", "[NH2:5][CH3:6]"],
                "reagents": ["CN"],
                "products": ["[CH3:6][NH:5][C:2]([CH3:1])=[O:4]", "[OH2:3]"],
            },
        },
        {
            "name": "rnx_8",
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4].CN>[CH3:6][NH2:5]>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {
                "reactants": ["[CH3:1][C:2]([OH:3])=[O:4]", "[NH2:5][CH3:6]"],
                "reagents": ["CN"],
                "products": ["[CH3:6][NH:5][C:2]([CH3:1])=[O:4]", "[OH2:3]"],
            },
        },
        {
            "name": "rnx_9",
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4].C[OH:3]>[CH3:6][NH2:5]>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {
                "reactants": ["[CH3:1][C:2]([OH:3])=[O:4]", "[NH2:5][CH3:6]"],
                "reagents": ["CO"],
                "products": ["[CH3:6][NH:5][C:2]([CH3:1])=[O:4]", "[OH2:3]"],
            },
        },
    ]

    for item in test_set:
        rdrxn = cif.rdrxn_from_string(input_string=item.get("smiles"), inp_fmt="smiles")
        assert rdrxn
        rdrxn_new = cif.rdrxn_role_reassignment(rdrxn=rdrxn, desired_product_idx=0)
        rdChemReactions.SanitizeRxn(rdrxn_new)
        rdrxn_new.Initialize()
        rxn_mol_catalog = cif.rdrxn_to_rxn_mol_catalog(rdrxn=rdrxn_new)
        reagents_catalog = [
            Chem.MolToSmiles(cif.remove_rdmol_atom_mapping(rdmol))
            for role, list_rdmol in rxn_mol_catalog.items()
            for rdmol in list_rdmol
            if role == "reagents"
        ]
        reactants_catalog = [
            Chem.MolToSmiles(rdmol)
            for role, list_rdmol in rxn_mol_catalog.items()
            for rdmol in list_rdmol
            if role == "reactants"
        ]
        assert item["expected"]["reagents"] == reagents_catalog
        assert item["expected"]["reactants"] == reactants_catalog


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
        rdmol_oxn = cif.compute_oxidation_numbers(rdmol=rdmol_inp)
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


def test_mapped_rdmol_atom_ids_canonicalization():
    # the same molecule with two different mapping
    s1 = "[cH:5]1[cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]2[c:3]([cH:4]1)[C:2](=[O:1])[N:13]=[N+:14]=[N-:15]"
    s2 = "[cH:1]1[cH:6][c:7]2[cH:15][n:9][cH:10][cH:14][c:12]2[c:3]([cH:4]1)[C:2](=[O:5])[N:13]=[N+:11]=[N-:8]"
    mol1 = cif.canonicalize_rdmol(cif.rdmol_from_string(s1, inp_fmt="smiles"))[0]
    mol2 = cif.canonicalize_rdmol(cif.rdmol_from_string(s2, inp_fmt="smiles"))[0]
    d1 = {a.GetIdx(): [a.GetSymbol()] for a in mol1.GetAtoms()}
    d2 = {a.GetIdx(): [a.GetSymbol()] for a in mol2.GetAtoms()}
    # even if canonicalized, the two rdmol have different atom ids
    assert d1 != d2
    # to get the identical atom ids, we need to canonicalize the atoms order after the map numbers have been removed
    mol1_canonical_atoms = cif.canonicalize_mapped_rdmol(
        cif.rdmol_from_string(s1, inp_fmt="smiles")
    )
    mol2_canonical_atoms = cif.canonicalize_mapped_rdmol(
        cif.rdmol_from_string(s2, inp_fmt="smiles")
    )
    d1 = {a.GetIdx(): [a.GetSymbol()] for a in mol1_canonical_atoms.GetAtoms()}
    d2 = {a.GetIdx(): [a.GetSymbol()] for a in mol2_canonical_atoms.GetAtoms()}
    assert d1 == d2


def test_new_canonicalization():
    s1 = "[CH3:1][CH2:2][C@H:3]([F:4])[C@H:5]([CH3:6])[OH:7] |o1:2,4|"
    s2 = "[CH3:4][CH2:2][C@H:3]([F:1])[C@H:5]([CH3:6])[OH:7] |o1:2,4|"
    rdmol1 = cif.rdmol_from_string(s1, "smiles")
    rdmol2 = cif.rdmol_from_string(s2, "smiles")
    stereo_non_canonical1 = {}
    for group in rdmol1.GetStereoGroups():
        stereo = [group.GetGroupType(), [atom.GetIdx() for atom in group.GetAtoms()]]
        stereo_atoms_non_canonical = stereo[-1]
        stereo_non_canonical1[stereo[0]] = stereo[-1]
    atomic_props_non_canonical = {}
    for atom in rdmol1.GetAtoms():
        atom_id = atom.GetIdx()
        props = atom.GetPropsAsDict(True, False)
        atomic_props_non_canonical[atom_id] = props
    rdmol1_canonical = cif.new_molecule_canonicalization(rdmol1)
    rdmol2_canonical = cif.new_molecule_canonicalization(rdmol2)
    for group in rdmol1_canonical.GetStereoGroups():
        stereo = [group.GetGroupType(), [atom.GetIdx() for atom in group.GetAtoms()]]
        stereo_atoms_canonical = stereo[-1]
    atomic_props_canonical = {}
    for atom in rdmol1_canonical.GetAtoms():
        atom_id = atom.GetIdx()
        props = atom.GetPropsAsDict(True, False)
        atomic_props_canonical[atom_id] = props
    assert (
        atomic_props_canonical[stereo_atoms_canonical[0]]
        == atomic_props_non_canonical[stereo_atoms_non_canonical[0]]
    )
    assert (
        atomic_props_canonical[stereo_atoms_canonical[-1]]
        == atomic_props_non_canonical[stereo_atoms_non_canonical[-1]]
    )


def test_atom_ids_canonicalization_w_enhanced_stereo(mols_w_stereo):
    # rdmol created from smiles
    rdmols = {
        i: cif.canonicalize_rdmol_lite(cif.rdmol_from_string(d["smiles"], "smiles"))
        for i, d in mols_w_stereo.items()
    }

    d1 = {a.GetIdx(): [a.GetSymbol()] for a in rdmols[0].GetAtoms()}
    d2 = {a.GetIdx(): [a.GetSymbol()] for a in rdmols[1].GetAtoms()}
    d3 = {a.GetIdx(): [a.GetSymbol()] for a in rdmols[2].GetAtoms()}
    d4 = {a.GetIdx(): [a.GetSymbol()] for a in rdmols[3].GetAtoms()}
    # enhanced stereochemistry info does not affect the atoms' id
    assert d1 == d2
    # atom mapping and different canonical stereo affect the atoms' id
    assert d1 != d3
    assert d4 != d2
    assert d4 != d3

    # rdmols created from V3000 MolBlock
    blocksV3k = {i: cif.compute_mol_blockV3k(rdmol) for i, rdmol in rdmols.items()}
    rdmols_from_blocksV3k = {
        i: cif.canonicalize_rdmol_lite(cif.rdmol_from_string(block, "mol_block"))
        for i, block in blocksV3k.items()
    }
    d1 = {a.GetIdx(): [a.GetSymbol()] for a in rdmols_from_blocksV3k[0].GetAtoms()}
    d2 = {a.GetIdx(): [a.GetSymbol()] for a in rdmols_from_blocksV3k[1].GetAtoms()}
    d3 = {a.GetIdx(): [a.GetSymbol()] for a in rdmols_from_blocksV3k[2].GetAtoms()}
    d4 = {a.GetIdx(): [a.GetSymbol()] for a in rdmols_from_blocksV3k[3].GetAtoms()}
    # enhanced stereochemistry info does not affect the atoms' id
    assert d1 == d2
    # atom mapping affects the atoms' id
    assert d1 != d3
    assert d4 != d2
    assert d4 != d3

    # rdmols created from MolBlock
    blocks = {i: cif.compute_mol_block(rdmol) for i, rdmol in rdmols.items()}
    rdmols_from_blocks = {
        i: cif.canonicalize_rdmol_lite(cif.rdmol_from_string(block, "mol_block"))
        for i, block in blocks.items()
    }
    d1 = {a.GetIdx(): [a.GetSymbol()] for a in rdmols_from_blocks[0].GetAtoms()}
    d2 = {a.GetIdx(): [a.GetSymbol()] for a in rdmols_from_blocks[1].GetAtoms()}
    d3 = {a.GetIdx(): [a.GetSymbol()] for a in rdmols_from_blocks[2].GetAtoms()}
    d4 = {a.GetIdx(): [a.GetSymbol()] for a in rdmols_from_blocks[3].GetAtoms()}
    # enhanced stereochemistry info does not affect the atoms' id
    assert d1 == d2
    # atom mapping affects the atoms' id
    assert d1 != d3
    assert d4 != d2
    assert d4 != d3


def test_mapping_diagnosis():
    smiles = {
        # unmapped atoms in the product --> missing reactants
        1: "[CH3:1][C:2]([OH:3])=[O:4]>O>CN[C:2]([CH3:1])=[O:4]",
        # unmapped atoms in a reactant are bound together --> possible leaving group
        2: "[O:1]=[C:2]([c:3]1[cH:4][cH:5][cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]12)N=[N+]=[N-]>ClCCl.O=C(Cl)C(=O)Cl>O[C:2](=[O:1])[c:3]1[cH:4][cH:5][cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]12.[N-:13]=[N+:14]=[N-:15]",
        3: "Cl[c:8]1[n:9][cH:10][cH:11][c:12]2[c:3]([cH:4][cH:5][cH:6][c:7]12)[C:2](=[O:1])N=[N+]=[N-]>ClCCl.O=C(Cl)C(=O)Cl>O[C:2](=[O:1])[c:3]1[cH:4][cH:5][cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]12",
    }
    ce_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles"
    )
    ce1 = ce_constructor.build_from_reaction_string(
        reaction_string=smiles[1], inp_fmt="smiles"
    )
    desired_prod1 = ce1.get_products()[0]

    # a warning is raised if there are unmapped atoms in the desired product
    with unittest.TestCase().assertLogs(
        "linchemin.cheminfo.functions", level="WARNING"
    ) as w:
        cif.mapping_diagnosis(ce1, desired_prod1)
    unittest.TestCase().assertEqual(len(w.records), 1)
    unittest.TestCase().assertIn("unmapped", w.records[0].getMessage())
    # unmapped atoms in a single reactant are all bounded together: they might indicate a leaving group
    ce2 = ce_constructor.build_from_reaction_string(
        reaction_string=smiles[2], inp_fmt="smiles"
    )
    desired_prod2 = ce2.get_products()[0]
    fragments2 = cif.mapping_diagnosis(ce2, desired_prod2)
    assert len(fragments2) == 1
    assert "." not in fragments2[0]
    # unmapped atoms in a single reactant are only partially bounded together: there might be more than one leaving group
    ce3 = ce_constructor.build_from_reaction_string(
        reaction_string=smiles[3], inp_fmt="smiles"
    )
    desired_prod3 = ce3.get_products()[0]
    fragments3 = cif.mapping_diagnosis(ce3, desired_prod3)
    assert len(fragments3) == 1
    assert "." in fragments3[0]
