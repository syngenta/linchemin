import unittest

import pytest
from rdkit import Chem
from rdkit.Chem import DataStructs, rdChemReactions

from linchemin.cheminfo.constructors import (
    ChemicalEquationConstructor,
    MoleculeConstructor,
)
from linchemin.cheminfo.functions import (
    bstr_to_rdmol,
    canonicalize_mapped_rdmol,
    canonicalize_rdmol,
    canonicalize_rdmol_lite,
    compute_oxidation_numbers,
    get_canonical_order,
    has_mapped_products,
    is_mapped_molecule,
    mapping_diagnosis,
    rdchiral_extract_template,
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


def test_mapped_molecule():
    m1 = rdkit.Chem.MolFromSmiles("c1([C@H](C)CC)cccc2ccccc12")
    m2 = rdkit.Chem.MolFromSmiles("Cl[C:2]([CH3:1])=[O:3]")
    assert is_mapped_molecule(m1) is False
    assert is_mapped_molecule(m2) is True


def test_mapped_rdrxn():
    smiles1 = "CN.CC(O)=O>O>CNC(C)=O"
    smiles2 = "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]"
    rdrxn1 = rdrxn_from_string(input_string=smiles1, inp_fmt="smiles")
    rdrxn2 = rdrxn_from_string(input_string=smiles2, inp_fmt="smiles")
    assert has_mapped_products(rdrxn1) is False
    assert has_mapped_products(rdrxn2) is True


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

    list_output = [rdchiral_extract_template(**item) for item in list_input]
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
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4].C[O:3]>[CH3:6][NH2:5]>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {
                "reactants": ["[CH3:1][C:2]([OH:3])=[O:4]", "[NH2:5][CH3:6]"],
                "reagents": ["CO"],
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
        rxn_mol_catalog = rdrxn_to_rxn_mol_catalog(rdrxn=rdrxn_new)
        reagents_catalog = [
            Chem.MolToSmiles(remove_rdmol_atom_mapping(rdmol))
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


def test_mapped_rdmol_atom_ids_canonicalization():
    # the same molecule with two different mapping
    s1 = "[cH:5]1[cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]2[c:3]([cH:4]1)[C:2](=[O:1])[N:13]=[N+:14]=[N-:15]"
    s2 = "[cH:1]1[cH:6][c:7]2[cH:15][n:9][cH:10][cH:14][c:12]2[c:3]([cH:4]1)[C:2](=[O:5])[N:13]=[N+:11]=[N-:8]"
    mol1 = canonicalize_rdmol(rdmol_from_string(s1, inp_fmt="smiles"))[0]
    mol2 = canonicalize_rdmol(rdmol_from_string(s2, inp_fmt="smiles"))[0]
    d1 = {a.GetIdx(): [a.GetSymbol()] for a in mol1.GetAtoms()}
    d2 = {a.GetIdx(): [a.GetSymbol()] for a in mol2.GetAtoms()}
    # even if canonicalized, the two rdmol have different atom ids
    assert d1 != d2
    # to get the identical atom ids, we need to canonicalized the atoms order after the map numbers have been removed
    mol1_canonical_atoms = canonicalize_mapped_rdmol(
        rdmol_from_string(s1, inp_fmt="smiles")
    )
    mol2_canonical_atoms = canonicalize_mapped_rdmol(
        rdmol_from_string(s2, inp_fmt="smiles")
    )
    d1 = {a.GetIdx(): [a.GetSymbol()] for a in mol1_canonical_atoms.GetAtoms()}
    d2 = {a.GetIdx(): [a.GetSymbol()] for a in mol2_canonical_atoms.GetAtoms()}
    assert d1 == d2


def test_mapping_diagnosis():
    smiles = {
        1: "[CH3:1][C:2]([OH:3])=[O:4]>O>CN[C:2]([CH3:1])=[O:4]",
        2: "[O:1]=[C:2]([c:3]1[cH:4][cH:5][cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]12)N=[N+]=[N-]>ClCCl.O=C(Cl)C(=O)Cl>O[C:2](=[O:1])[c:3]1[cH:4][cH:5][cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]12.[N-:13]=[N+:14]=[N-:15]",
        3: "Cl[c:8]1[n:9][cH:10][cH:11][c:12]2[c:3]([cH:4][cH:5][cH:6][c:7]12)[C:2](=[O:1])N=[N+]=[N-]>ClCCl.O=C(Cl)C(=O)Cl>O[C:2](=[O:1])[c:3]1[cH:4][cH:5][cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]12",
    }
    ce_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles"
    )
    chemical_equations = {
        n: ce_constructor.build_from_reaction_string(
            reaction_string=s, inp_fmt="smiles"
        )
        for n, s in smiles.items()
    }
    desired_prod1 = [
        mol
        for uid, mol in chemical_equations.get(1).catalog.items()
        if uid == chemical_equations.get(1).role_map["products"][0]
    ][0]
    # a warning is raised if there are unmapped atoms in the desired product
    with unittest.TestCase().assertLogs(
        "linchemin.cheminfo.functions", level="WARNING"
    ) as w:
        mapping_diagnosis(chemical_equations.get(1), desired_prod1)
    unittest.TestCase().assertEqual(len(w.records), 1)
    unittest.TestCase().assertIn("unmapped", w.records[0].getMessage())

    # unmapped atoms in a single reactant are all bounded together: they might indicate a leaving group
    desired_prod2 = [
        mol
        for uid, mol in chemical_equations.get(2).catalog.items()
        if uid == chemical_equations.get(2).role_map["products"][0]
    ][0]
    fragments2 = mapping_diagnosis(chemical_equations.get(2), desired_prod2)
    assert len(fragments2) == 1
    assert "." not in fragments2[0]

    # unmapped atoms in a single reactant are only partially bounded together: there might be more than one leaving group
    desired_prod3 = [
        mol
        for uid, mol in chemical_equations.get(3).catalog.items()
        if uid == chemical_equations.get(3).role_map["products"][0]
    ][0]
    fragments3 = mapping_diagnosis(chemical_equations.get(3), desired_prod3)
    assert len(fragments3) == 1
    assert "." in fragments3[0]


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
