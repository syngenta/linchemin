import unittest
from itertools import combinations

import pytest

import linchemin.cheminfo.depiction as cid
import linchemin.cheminfo.functions as cif
from linchemin.cheminfo.chemical_hashes import (
    UnavailableMolIdentifier,
    UnavailableReactionIdentifier,
)
from linchemin.cheminfo.constructors import (
    BadMapping,
    ChemicalEquationConstructor,
    MoleculeConstructor,
    PatternConstructor,
    RatamConstructor,
    TemplateConstructor,
)
from linchemin.cheminfo.models import Template
from linchemin.utilities import create_hash


# Molecule tests
def test_molecular_constructor():
    smiles = "CC(C)=O"
    with pytest.raises(UnavailableMolIdentifier) as ke:
        MoleculeConstructor(molecular_identity_property_name="something")
    assert "UnavailableMolIdentifier" in str(ke.type)
    with unittest.TestCase().assertLogs(
        "linchemin.cheminfo.chemical_hashes", level="WARNING"
    ):
        molecule_constructor = MoleculeConstructor(
            molecular_identity_property_name="smiles", hash_list=["something"]
        )
        mol = molecule_constructor.build_from_molecule_string(
            molecule_string=smiles, inp_fmt="smiles"
        )
    assert "something" not in mol.hash_map


def test_molecule_equality():
    mols = {
        0: {"smiles": "CN"},  # M1
        1: {"smiles": "CN"},  # M1
        2: {"smiles": "NC"},  # M1
        3: {"smiles": "CC"},  # M2
        4: {"smiles": "CC(C)=O"},  # M3_T1
        5: {"smiles": "CC(O)=C"},  # M3_T2
        6: {"smiles": "CC(O)=N"},  # M4_T1
        7: {"smiles": "CC(N)=O"},  # M4_T2
        8: {"smiles": "CCC(C)=O"},  # M5_T1
        9: {"smiles": r"C\C=C(\C)O"},  # M5_T2
        10: {"smiles": "Cl[C:2]([CH3:1])=[O:3]"},  # M6_atom_mapping_1
        11: {"smiles": "Cl[C:1]([CH3:2])=[O:5]"},  # M6_atom_mapping_2
        12: {
            "smiles": "[cH:5]1[cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]2[c:3]([cH:4]1)[C:2](=[O:1])[N:13]=[N+:14]=[N-:15]"
        },
        13: {
            "smiles": "[cH:1]1[cH:6][c:7]2[cH:15][n:9][cH:10][cH:14][c:12]2[c:3]([cH:4]1)[C:2](=[O:5])[N:13]=[N+:11]=[N-:8]"
        },
    }
    # initialize the constructor to use smiles as identity property
    molecule_constructor = MoleculeConstructor(
        molecular_identity_property_name="smiles"
    )
    # using smiles
    ms1 = {
        k: molecule_constructor.build_from_molecule_string(
            molecule_string=v.get("smiles"), inp_fmt="smiles"
        )
        for k, v in mols.items()
    }

    assert ms1.get(0) == ms1.get(1)  # identical molecule, identical input string
    assert ms1.get(0) == ms1.get(
        2
    )  # identical molecule, different input string; assess the canonicalization mechanism
    assert ms1.get(0) != ms1.get(3)  # different molecules
    assert ms1.get(4) != ms1.get(
        5
    )  # same molecule, but different tautomers: smiles fails to capture identity
    assert ms1.get(6) != ms1.get(
        7
    )  # same molecule, but different tautomers: smiles fails to capture identity
    assert ms1.get(8) != ms1.get(
        9
    )  # same molecule, but different tautomers: smiles fails to capture identity
    assert ms1.get(10) == ms1.get(11)  # same molecule, but different atom mapping

    mol1 = ms1.get(12).rdmol_mapped
    mol2 = ms1.get(13).rdmol_mapped
    d1 = {a.GetIdx(): [a.GetSymbol()] for a in mol1.GetAtoms()}
    d2 = {a.GetIdx(): [a.GetSymbol()] for a in mol2.GetAtoms()}
    assert d1 == d2
    assert ms1.get(14) == ms1.get(15)

    # initialize the constructor to use inchi_key as identity property
    molecule_constructor = MoleculeConstructor(
        molecular_identity_property_name="inchi_key"
    )
    ms2 = {
        k: molecule_constructor.build_from_molecule_string(
            molecule_string=v.get("smiles"), inp_fmt="smiles"
        )
        for k, v in mols.items()
    }
    assert ms2.get(0) == ms2.get(1)  # identical molecule, identical input string
    assert ms2.get(0) == ms2.get(
        2
    )  # identical molecule, different input string; assess the canonicalization mechanism
    assert ms2.get(0) != ms2.get(3)  # different molecules
    # assert ms2.get(4) == ms2.get(5)  # same molecule, but different tautomers: inchi_key succeeds to capture identity # TODO: it does not work inchi are different!!!!!
    assert ms2.get(6) == ms2.get(
        7
    )  # same molecule, but different tautomers: inchi_key succeeds to capture identity
    assert ms2.get(10) == ms2.get(11)  # same molecule, but different atom mapping


def test_molecule_from_molblocl():
    smiles = "C[C@H](Cl)[C@H](F)C |&1:2,3|"
    rdmol = cif.Chem.MolFromSmiles(smiles)
    block = cif.Chem.MolToMolBlock(rdmol)
    # if the selected molecular identity property is smiles,
    # absolute stereochemistry info does not participate in the identity of the Molecule object
    mol_constructor = MoleculeConstructor("smiles")
    molecule = mol_constructor.build_from_molecule_string(block, "mol_block")
    molecule_from_smiles = mol_constructor.build_from_molecule_string(smiles, "smiles")
    assert molecule == molecule_from_smiles

    mol_constructor = MoleculeConstructor("cx_smiles")
    molecule = mol_constructor.build_from_molecule_string(block, "mol_block")
    molecule_from_smiles = mol_constructor.build_from_molecule_string(smiles, "smiles")
    assert molecule != molecule_from_smiles


def test_mol_w_smiles_cx_smiles():
    # flat molecule
    s1 = "CCC(F)C(C)O"
    # molecule with standard stereochemistry
    s2 = "CC[C@H](F)[C@H](C)O"
    # molecule with enhanced stereochemistry
    s3 = "CC[C@H](F)[C@H](C)O |o1:2,4|"
    # molecule with enhanced stereochemistry and mapping
    s4 = "[CH3:1][CH2:2][C@H:3]([F:4])[C@H:5]([CH3:6])[OH:7] |o1:2,4|"

    # when the uid is based on canonical smiles, all the molecules are equivalent
    mol_constructor = MoleculeConstructor("smiles", ["smiles", "cx_smiles"])
    mol1 = mol_constructor.build_from_molecule_string(s1, "smiles")
    mol2 = mol_constructor.build_from_molecule_string(s2, "smiles")
    mol3 = mol_constructor.build_from_molecule_string(s3, "smiles")
    mol4 = mol_constructor.build_from_molecule_string(s4, "smiles")
    # the flat molecule is still different from the one with standard stereochemistry
    assert mol1 != mol2
    # all the others are equivalent
    assert mol2 == mol3
    assert mol3 == mol4
    # enhanced stereo info is lost when the uid is based on canonical smiles
    assert not all(cif.has_enhanced_stereo(m.rdmol) for m in [mol1, mol2, mol3, mol4])

    # when the uid is based on cx_smiles, the presence of enhanced stereo information influences the equivalence
    mol_constructor_cxsmiles = MoleculeConstructor("cx_smiles")
    mol1_cx = mol_constructor_cxsmiles.build_from_molecule_string(s1, "smiles")
    mol2_cx = mol_constructor_cxsmiles.build_from_molecule_string(s2, "smiles")
    mol3_cx = mol_constructor_cxsmiles.build_from_molecule_string(s3, "smiles")
    mol4_cx = mol_constructor_cxsmiles.build_from_molecule_string(s4, "smiles")
    # the molecules without enhanced stereo information are different from the others
    assert mol2_cx != mol1_cx
    assert mol2_cx != mol3_cx
    # the atom mapping does not influence identity
    assert mol3_cx == mol4_cx
    stereo_groups3 = mol3_cx.rdmol.GetStereoGroups()
    stereo_groups4 = mol4_cx.rdmol.GetStereoGroups()
    stereo_groups4_mapped = mol4_cx.rdmol_mapped.GetStereoGroups()
    atoms_w_stereo3 = [
        (atom.GetIdx(), atom.GetSymbol(), atom.GetAtomMapNum())
        for group in stereo_groups3
        for atom in group.GetAtoms()
    ]
    atoms_w_stereo4 = [
        (atom.GetIdx(), atom.GetSymbol(), atom.GetAtomMapNum())
        for group in stereo_groups4
        for atom in group.GetAtoms()
    ]
    atoms_w_stereo4_mapped = [
        (atom.GetIdx(), atom.GetSymbol(), atom.GetAtomMapNum())
        for group in stereo_groups4_mapped
        for atom in group.GetAtoms()
    ]
    assert sorted(atoms_w_stereo3) == sorted(atoms_w_stereo4)
    # rdmol1 = cif.canonicalize_mapped_rdmol((cif.Chem.MolFromSmiles(s1)))
    # rdmol2 = cif.canonicalize_mapped_rdmol((cif.Chem.MolFromSmiles(s2)))
    # rdmol3 = cif.Chem.MolFromSmiles(s3)
    # print([n for n in cif.Chem.CanonicalRankAtoms(rdmol3)])
    # stereo_groups3 = rdmol3.GetStereoGroups()
    # print([
    #     (atom.GetIdx(), atom.GetSymbol(), atom.GetAtomMapNum())
    #     for group in stereo_groups3
    #     for atom in group.GetAtoms()
    # ])
    # rdmol_canon = cif.canonicalize_mapped_rdmol(rdmol3)
    # print([n for n in cif.Chem.CanonicalRankAtoms(rdmol_canon)])
    # stereo_groups3 = rdmol_canon.GetStereoGroups()
    # print(
    #     [
    #         (atom.GetIdx(), atom.GetSymbol(), atom.GetAtomMapNum())
    #         for group in stereo_groups3
    #         for atom in group.GetAtoms()
    #     ]
    # )
    # print(cif.Chem.MolToCXSmiles(rdmol_canon))


# ChemicalEquation tests
def test_chemical_equation_hashing():
    reactions = {
        0: {"smiles": "CN.CC(O)=O>O>CNC(C)=O"},
        1: {"smiles": "CC(O)=O.CN>O>CNC(C)=O"},
        2: {"smiles": ">>CNC(C)=O"},
        3: {"smiles": "CC(O)=O.CN>>CNC(C)=O"},
        # 4: {'smiles': 'CC(O)=O.CN>>'},
        5: {"smiles": "CN.CC(O)=O>>CNC(C)=O"},
        6: {"smiles": "CNC(C)=O>>CN.CC(O)=O"},
        7: {"smiles": "CN.CC(O)=O>>CNC(C)=O.O"},
        8: {"smiles": "CN.CC(O)=O>>O.CNC(C)=O"},
        9: {"smiles": "CNC(C)=O>O>CN.CC(O)=O"},
    }
    # initialize the constructor
    chemical_equation_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    results = {}
    for k, v in reactions.items():
        chemical_equation = chemical_equation_constructor.build_from_reaction_string(
            reaction_string=v.get("smiles"), inp_fmt="smiles"
        )

        h = chemical_equation.hash_map
        results[k] = h
        # print(k, h)
    # the hashes are calculated and have a non-null (None) value
    assert results.get(0).get("reactants")
    assert results.get(0).get("reagents")
    assert results.get(0).get("products")
    assert results.get(0).get("r_p")
    assert results.get(0).get("r_r_p")
    assert results.get(0).get("u_r_p")
    assert results.get(0).get("u_r_r_p")

    # the reactant hash is insensitive to the input order of reactants (reaction canonicalization OK)
    assert results.get(0).get("reactants") == results.get(1).get("reactants")
    # the product hash is insensitive to the input order of products (reaction canonicalization OK)
    assert results.get(7).get("products") == results.get(8).get("products")
    # the machinery does not break when the reactants are missing
    assert results.get(2).get("reactants")
    # the machinery does not break when the agents are missing
    assert results.get(3).get("reagents")
    # the machinery does not break when the products are missing
    # assert results.get(4).get('products')
    # there is a special hash for missing roles (it is the hash of an empty string)
    # assert results.get(2).get('reactants') == results.get(3).get('reagents') == results.get(4).get(
    #     'products') == create_hash('')
    # the reactant and products hashes are conserved even when the reagents are missing
    assert results.get(0).get("reactants") == results.get(5).get("reactants")
    assert results.get(0).get("products") == results.get(5).get("products")
    # the agent hash is different if the agents are missing
    assert results.get(0).get("reagents") != results.get(5).get("reagents")
    # the base r>p hash is conserved if the agents are missing in one reaction
    assert results.get(0).get("r_p") == results.get(5).get("r_p")
    # the full r>a>p hash is not conserved if the reagents are missing in one reaction
    assert results.get(0).get("r_r_p") != results.get(5).get("r_r_p")
    # the base r>>p hash is not conserved if the reaction is reversed
    assert results.get(0).get("r_p") != results.get(6).get("r_p")
    # the full r>a>p hash is not conserved if the reaction is reversed
    assert results.get(0).get("r_r_p") != results.get(6).get("r_r_p")
    # the reversible base r<>p hash is  conserved if the reaction is reversed
    assert results.get(0).get("u_r_p") == results.get(9).get("u_r_p")
    assert results.get(3).get("u_r_p") == results.get(6).get("u_r_p")
    # the reversible full r<a>p hash is  conserved if the reaction is reversed
    assert results.get(0).get("u_r_r_p") == results.get(9).get("u_r_r_p")
    assert results.get(3).get("u_r_r_p") == results.get(6).get("u_r_r_p")


def test_ce_from_block():
    smiles = "CN.CC(O)=O>O>CNC(C)=O"
    rxn = cif.rdrxn_from_string(smiles, "smiles")
    # with RxnBlockV3000, reactants and reagents are correctly assigned, if they are correct in the input smiles
    block = cif.rdrxn_to_string(rxn, "rxn_blockV3K")
    ce_constructor = ChemicalEquationConstructor("smiles", "r_r_p")
    ce = ce_constructor.build_from_reaction_string(block, "rxn_block")
    assert ce
    ce_from_smiles = ce_constructor.build_from_reaction_string(smiles, "smiles")
    assert ce_from_smiles == ce

    # if a V2000 RxnBlock is generated, reactants and reagents are mixed
    block2 = cif.rdrxn_to_string(rxn, "rxn_blockV2K")
    ce_from_block = ce_constructor.build_from_reaction_string(block2, "rxn_block")
    assert ce_from_block != ce

    # by forcing V3000 and agents separation, the correct reaction is generated
    block3 = cif.rdrxn_to_string(rxn, "rxn")
    ce_from_block = ce_constructor.build_from_reaction_string(block3, "rxn_block")
    assert ce_from_block == ce


def test_chemical_eq_constructor_arguments():
    # with valid arguments
    ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    # with invalid molecule_identity_property_name
    with pytest.raises(UnavailableMolIdentifier):
        ChemicalEquationConstructor(
            molecular_identity_property_name="not_existing",
            chemical_equation_identity_name="r_p",
        )
    with pytest.raises(UnavailableReactionIdentifier):
        ChemicalEquationConstructor(
            molecular_identity_property_name="smiles",
            chemical_equation_identity_name="not_existing",
        )


def test_instantiate_chemical_equation():
    # the identity property r_r_p considers reactants, reagents ad products
    reaction_smiles_input = "NC.CC(O)=O>O>CNC(C)=O"
    chemical_equation_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    chemical_equation1 = chemical_equation_constructor.build_from_reaction_string(
        reaction_string=reaction_smiles_input, inp_fmt="smiles"
    )
    assert chemical_equation1
    assert chemical_equation1.smiles == "CC(=O)O.CN>O>CNC(C)=O"
    assert list(chemical_equation1.rdrxn.GetAgents())
    # assert molecules are canonicalized
    # assert reaction is canonicalized

    # the identity property r_p only considers reactants and prodcts
    chemical_equation_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles", chemical_equation_identity_name="r_p"
    )
    chemical_equation2 = chemical_equation_constructor.build_from_reaction_string(
        reaction_string=reaction_smiles_input, inp_fmt="smiles"
    )
    assert chemical_equation2.smiles == "CC(=O)O.CN>>CNC(C)=O"
    assert chemical_equation1.smiles != chemical_equation2.smiles
    assert not list(chemical_equation2.rdrxn.GetAgents())
    assert chemical_equation2.disconnection == chemical_equation2.template is None

    # if the input string is mapped, the disconnection and other attributes are built
    mapped_reaction_smiles = (
        "[CH3:6][NH2:5].[CH3:2][C:3]([OH:4])=[O:1]>O>[CH3:6][NH:5][C:3]([CH3:2])=[O:1]"
    )
    chemical_equation = chemical_equation_constructor.build_from_reaction_string(
        reaction_string=mapped_reaction_smiles, inp_fmt="smiles"
    )
    assert (
        chemical_equation.smiles
        == "[NH2:5][CH3:6].[O:1]=[C:3]([CH3:2])[OH:4]>>[O:1]=[C:3]([CH3:2])[NH:5][CH3:6]"
    )
    assert chemical_equation.disconnection
    assert chemical_equation.template


def test_create_reaction_smiles_from_chemical_equation():
    reaction_smiles_input = "CN.CC(O)=O>O>CNC(C)=O"
    chemical_equation_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles", chemical_equation_identity_name="r_p"
    )
    chemical_equation = chemical_equation_constructor.build_from_reaction_string(
        reaction_string=reaction_smiles_input, inp_fmt="smiles"
    )
    reaction_smiles_no_reagents = chemical_equation.build_reaction_smiles(
        use_reagents=False
    )
    reaction_smiles_reagents = chemical_equation.build_reaction_smiles(
        use_reagents=True
    )

    assert reaction_smiles_no_reagents, reaction_smiles_reagents
    assert reaction_smiles_no_reagents != reaction_smiles_reagents


def test_reaction_canonicalization_from_molecules():
    reactions = {
        0: {"smiles": "CN.CC(O)=O>O>CNC(C)=O"},
        1: {"smiles": "CC(O)=O.CN>O>CNC(C)=O"},
        2: {"smiles": ">>CNC(C)=O"},
        3: {"smiles": "CC(O)=O.CN>>CNC(C)=O"},
        # 4: {'smiles': 'CC(O)=O.CN>>'},
        5: {"smiles": "CN.CC(O)=O>>CNC(C)=O"},
        6: {"smiles": "CNC(C)=O>>CN.CC(O)=O"},
        7: {"smiles": "CN.CC(O)=O>>CNC(C)=O.O"},
        8: {"smiles": "CN.CC(O)=O>>O.CNC(C)=O"},
        9: {"smiles": "CNC(C)=O>O>CN.CC(O)=O"},
    }
    # initialize the constructor
    chemical_equation_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles"
    )
    results = {}
    for k, v in reactions.items():
        chemical_equation = chemical_equation_constructor.build_from_reaction_string(
            reaction_string=v.get("smiles"), inp_fmt="smiles"
        )
        results[k] = chemical_equation
        # print(k, h)
    # the reaction smiles is insensitive to the input order of reactants (reaction canonicalization OK)

    assert results.get(0).smiles == results.get(1).smiles


def test_chemical_equation_equality():
    reactions = {
        0: {"smiles": "CN.CC(O)=O>O>CNC(C)=O"},  # R1
        1: {"smiles": "CN.CC(O)=O>O>CNC(C)=O"},  # R1
        2: {"smiles": "NC.CC(O)=O>O>CNC(C)=O"},  # R1
        3: {"smiles": "NC.CC(=O)O>O>CNC(C)=O"},  # R1
        4: {"smiles": "CC(O)=O.CN>O>CNC(C)=O"},  # R1
        5: {
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]"
        },
        6: {
            "smiles": "[CH3:1][C:20]([OH:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:20]([CH3:1])=[O:4].[OH2:3]"
        },
        7: {
            "smiles": "[CH3:6][NH2:5].[CH3:1][C:20]([OH:3])=[O:4]>>[CH3:6][NH:5][C:20]([CH3:1])=[O:4].[OH2:3]"
        },
    }

    ces1 = {}

    # initialize the constructor
    chemical_equation_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )

    for k, v in reactions.items():
        chemical_equation = chemical_equation_constructor.build_from_reaction_string(
            reaction_string=v.get("smiles"), inp_fmt="smiles"
        )
        ces1[k] = chemical_equation

    assert ces1.get(0) == ces1.get(
        1
    )  # same reaction, one reactant has a different smiles:
    # test mol canonicalization
    assert ces1.get(0) == ces1.get(
        2
    )  # same reaction, two reactant have a different smiles:
    # test mol canonicalization
    assert ces1.get(0) == ces1.get(
        3
    )  # same reaction, two reactant have a different smiles:
    # test mol canonicalization
    assert ces1.get(0) == ces1.get(
        4
    )  # same reaction, different reactant ordering: test reaction canonicalization
    assert ces1.get(5) == ces1.get(6)  # same reaction, different atom mapping
    assert ces1.get(5) == ces1.get(7)  # same reaction, different atom mapping,


def test_chemical_equation_stoichiometry():
    reactions = {
        0: {
            "smiles": "ClCl.ClCl.Oc1ccccc1>>Cl.Cl.Oc1ccc(Cl)cc1Cl",
            "stoichiometry": {"reactants": [2, 1], "reagents": [], "products": [2, 1]},
        },
        1: {
            "smiles": "[CH3:1][O:2][c:3]1[cH:4][cH:5][c:6]([CH3:7])[cH:13][c:14]1[NH:15][N:16]=[C:19]([CH3:26])[CH3:20].[ClH:17].[OH2:18].Cl>>[CH3:1][O:2][c:3]1[cH:4][cH:5][c:6]([CH3:7])[cH:13][c:14]1[NH:15][NH2:16].[ClH:17].[CH3:26][C:19]([CH3:20])=[O:18]",
            "stoichiometry": {
                "reactants": [1],
                "reagents": [2, 1],
                "products": [1, 1, 1],
            },
        },
        2: {
            "smiles": "COc1ccc(C)cc1NN=C(C)C>Cl.O>COc1ccc(C)cc1NN.Cl.CC(C)=O",
            "stoichiometry": {
                "reactants": [1],
                "reagents": [1, 1],
                "products": [1, 1, 1],
            },
        },
    }
    chemical_equation_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    for test in reactions.values():
        chemical_equation = chemical_equation_constructor.build_from_reaction_string(
            reaction_string=test["smiles"],
            inp_fmt="smiles",
        )

        for role, d in chemical_equation.stoichiometry_coefficients.items():
            assert list(d.values()) == test["stoichiometry"][role]


def test_chemical_equation_builder():
    reaction_string_reference = "CC(=O)O.CN.CN>O>CNC(C)=O"

    # initialize the constructor
    cec = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )

    for reaction_string_test in [
        "CC(=O)O.CN.CN>O>CNC(C)=O",  # expected smiles
        "CC(=O)O.NC.CN>O>CNC(C)=O",  # test molecule canonicalization: change order of atoms in reactant molecules
        "CN.CC(=O)O.CN>O>CNC(C)=O",  # test reaction canonicalization: change order of molecules in reactants
    ]:
        chemical_equation = cec.build_from_reaction_string(
            reaction_string=reaction_string_test, inp_fmt="smiles"
        )
        reaction_string_calculated = chemical_equation.smiles
        assert reaction_string_calculated == reaction_string_reference


def test_chemical_equation_attributes_are_not_available():
    smiles = "CN.CC(O)=O>O>CNC(C)=O"
    chemical_equation_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles"
    )
    chemical_equation = chemical_equation_constructor.build_from_reaction_string(
        reaction_string=smiles, inp_fmt="smiles"
    )
    disconnection = chemical_equation.disconnection
    assert not disconnection
    template = chemical_equation.template
    assert not template


def test_chemical_equation_attributes_are_available():
    smiles = "O[C:2](=[O:1])[c:3]1[cH:12][cH:7][cH:6][cH:5][cH:4]1.CN>ClCCl.ClC(=O)C(Cl)=O>C[NH][C:2](=[O:1])[c:3]1[cH:12][cH:7][cH:6][cH:5][cH:4]1"
    # initialize the constructor with identity property 'r_r_p'
    chemical_equation_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    ce1 = chemical_equation_constructor.build_from_reaction_string(
        reaction_string=smiles, inp_fmt="smiles"
    )

    assert ce1.mapping
    assert ce1.disconnection
    assert ce1.template
    # initialize the constructor with identity property 'r_p'
    chemical_equation_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles", chemical_equation_identity_name="r_p"
    )
    ce2 = chemical_equation_constructor.build_from_reaction_string(
        reaction_string=smiles, inp_fmt="smiles"
    )

    # disconnection and template are independent of the chemical equation identity property
    assert ce2.disconnection == ce1.disconnection
    assert ce2.disconnection.extract_info() == ce1.disconnection.extract_info()
    assert ce2.template == ce1.template
    assert ce2.mapping.reactants_unmapped_atoms_info["fraction"] == 0.11
    assert ce2.mapping.reactants_unmapped_atoms_info["unmapped_atoms"] != {}
    assert ce2.mapping.desired_product_unmapped_atoms_info["fraction"] == 0.2
    assert next(
        v
        for v in ce2.mapping.desired_product_unmapped_atoms_info[
            "unmapped_atoms"
        ].values()
    ) == [{0, 7}]


def test_chemical_equation_from_db():
    # reaction from Adraos' book
    db_list = [
        {
            "smiles": "CC(=O)Oc1cc(Cl)ccc1Oc2ccc(Cl)cc2Cl",
            "role": "desired_product",
            "stoichiometry": 1,
        },
        {"smiles": "OOC(=O)c1cccc(Cl)c1", "role": "reactant", "stoichiometry": 1},
        {
            "smiles": "CC(=O)c1cc(Cl)ccc1Oc2ccc(Cl)cc2Cl",
            "role": "reactant",
            "stoichiometry": 1,
        },
        {"smiles": "OC(=O)c1cccc(Cl)c1", "role": "by_product", "stoichiometry": 1},
    ]
    chemical_equation_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    ce = chemical_equation_constructor.build_from_db(db_list)
    assert ce.template is None
    smiles = "OOC(=O)c1cccc(Cl)c1.CC(=O)c1cc(Cl)ccc1Oc2ccc(Cl)cc2Cl>>CC(=O)Oc1cc(Cl)ccc1Oc2ccc(Cl)cc2Cl.OC(=O)c1cccc(Cl)c1"
    ce_from_smiles = chemical_equation_constructor.build_from_reaction_string(
        smiles, "smiles"
    )
    assert ce == ce_from_smiles
    db_list2 = [
        {
            "smiles": "[Na+].[Cl-]",
            "role": "by_product",
            "stoichiometry": 2,
        },
        {
            "smiles": "OC(=O)CCl",
            "role": "reactant",
            "stoichiometry": 1,
        },
        {
            "smiles": "Oc1cc(Cl)c(Cl)cc1Cl",
            "role": "reactant",
            "stoichiometry": 1,
        },
        {
            "smiles": "OC(=O)COc1cc(Cl)c(Cl)cc1Cl",
            "role": "desired_product",
            "stoichiometry": 1,
        },
        {
            "smiles": "O",
            "role": "by_product",
            "stoichiometry": 2,
        },
        {
            "smiles": "[OH-].[Na+]",
            "role": "agent",
            "stoichiometry": 2,
        },
        {
            "smiles": "[Cl-].[Na+]",
            "role": "reagent_quench",
            "stoichiometry": 1,
        },
    ]
    ce = chemical_equation_constructor.build_from_db(db_list2)
    assert ce.role_map["reagents"] != []
    na_uid = next(uid for uid, mol in ce.catalog.items() if mol.smiles == "[Na+]")
    assert ce.stoichiometry_coefficients["reagents"][na_uid] == 3


# Ratam tests
def test_ratam_and_role_reassignment():
    test_set = [
        # All initial reactants are actual reactants
        {
            "name": "rnx_1",
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {
                "reactants": ["CC(=O)O", "CN"],
                "reagents": [],
                "products": ["O", "CNC(C)=O"],
            },
        },
        # An initial reagents is actually a reactant
        {
            "name": "rnx_2",
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4]>[CH3:6][NH2:5]>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {
                "reactants": ["CC(=O)O", "CN"],
                "reagents": [],
                "products": ["O", "CNC(C)=O"],
            },
        },
        # A reagent is recognized as such
        {
            "name": "rnx_3",
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>CO>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {
                "reactants": ["CC(=O)O", "CN"],
                "reagents": ["CO"],
                "products": ["O", "CNC(C)=O"],
            },
        },
        # The same molecule appears twice, once as reactant and once as reagent
        {
            "name": "rnx_4",
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>CN>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {
                "reactants": ["CC(=O)O", "CN"],
                "reagents": ["CN"],
                "products": ["O", "CNC(C)=O"],
            },
        },
        # Bad mapping: the same map number is used more than twice
        {
            "name": "rnx_5",
            "smiles": "[CH3:3][C:2]([OH:1])=[O:4].[CH3:3][NH2:5]>CN>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:1]",
            "expected": {
                "reactants": ["CC(=O)O", "CN"],
                "reagents": ["CN"],
                "products": ["CNC(C)=O", "O"],
            },
        },
        # A reactant is missing and all atoms in the desired product
        # have a map number ("complete" option in namerxn)
        {
            "name": "rnx_6",
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4]",
            "expected": {
                "reactants": ["CC(=O)O"],
                "reagents": [],
                "products": ["CNC(C)=O"],
            },
        },
        # A reactant is missing and not all atoms in the desired
        # product have a map number ("matched" option in namerxn)
        {
            "name": "rnx_7",
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4]>>[CH3][NH][C:2]([CH3:1])=[O:4]",
            "expected": {
                "reactants": ["CC(=O)O"],
                "reagents": [],
                "products": ["CNC(C)=O"],
            },
        },
    ]
    mol_constructor = MoleculeConstructor(molecular_identity_property_name="smiles")
    for item in test_set:
        rdrxn = cif.rdrxn_from_string(input_string=item.get("smiles"), inp_fmt="smiles")
        reaction_mols = cif.rdrxn_to_molecule_catalog(rdrxn, mol_constructor)
        catalog = {
            m.uid: m
            for m in set(
                reaction_mols["reactants"]
                + reaction_mols["reagents"]
                + reaction_mols["products"]
            )
        }
        if item["name"] == "rnx_5":
            with pytest.raises(BadMapping) as ke:
                ratam_constructor = RatamConstructor()
                ratam_constructor.create_ratam(
                    reaction_mols, reaction_mols["products"][0]
                )
            assert "BadMapping" in str(ke.type)

        else:
            ratam_constructor = RatamConstructor()
            ratam = ratam_constructor.create_ratam(
                reaction_mols, reaction_mols["products"][0]
            )
            assert ratam
            assert ratam.atom_transformations

            for role, map_info in ratam.full_map_info.items():
                smiles_list = [
                    m.smiles for uid, m in catalog.items() if uid in map_info
                ]
                assert item["expected"][role] == smiles_list
            if item["name"] in ["rnx_6", "rnx_7"]:
                assert ratam.desired_product_unmapped_atoms_info["fraction"] == 0.4
                assert next(
                    v
                    for v in ratam.desired_product_unmapped_atoms_info[
                        "unmapped_atoms"
                    ].values()
                ) == [
                    {
                        0,
                        3,
                    }
                ]


# Pattern tests
def test_pattern_creation():
    test_set = [
        {
            "name": "pattern_1",
            "smarts": "[NH2;D1;+0:4]-[C;H0;D3;+0:1](=[O;D1;H0:2])-[c:3]",
            "expected": {},
        },
        {
            "name": "pattern_2",
            "smarts": "[CH3:6][NH:5][C:2]([CH3:1])=[O:4]",
            "expected": {},
        },
        {
            "name": "pattern_3",
            "smarts": "[CH3:1][c:2]1[cH:3][cH:4][cH:5][cH:6][n:7]1",
            "expected": {},
        },
    ]

    pc = PatternConstructor()
    for item in test_set:
        pattern = pc.build_from_molecule_string(
            molecule_string=item.get("smarts"), inp_fmt="smarts"
        )
        # print(f"\n{item.get('name')} {pattern.to_dict()}")
        assert pattern


# Template tests
def test_template_creation():
    test_set = [
        {
            "name": "rnx_1",
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {},
        },
        {
            "name": "rnx_2",
            "smiles": "[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]>>[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]",
            "expected": {},
        },
    ]

    for item in test_set:
        tc = TemplateConstructor()
        template = tc.build_from_reaction_string(
            reaction_string=item.get("smiles"), inp_fmt="smiles"
        )
        assert isinstance(template, Template)


def test_template_hashing():
    reactions = {
        0: {
            "smiles": "[CH3:1][NH2:2].O[C:3]([CH3:4])=[O:5]>O>[CH3:1][NH:2][C:3]([CH3:4])=[O:5]"
        },
        1: {
            "smiles": "O[C:3]([CH3:4])=[O:5].[CH3:1][NH2:2]>O>[CH3:1][NH:2][C:3]([CH3:4])=[O:5]"
        },
        2: {"smiles": ">>[CH3:1][NH:2][C:3]([CH3:4])=[O:5]"},
        3: {
            "smiles": "[CH3:1][NH2:2].O[C:3]([CH3:4])=[O:5]>>[CH3:1][NH:2][C:3]([CH3:4])=[O:5]"
        },
        4: {"smiles": "[CH3:1][NH2:2].O[C:3]([CH3:4])=[O:5]>>"},
        5: {
            "smiles": "[CH3:1][NH2:2].O[C:3]([CH3:4])=[O:5]>>[CH3:1][NH:2][C:3]([CH3:4])=[O:5]"
        },
        6: {
            "smiles": "[CH3:1][NH:2][C:3]([CH3:4])=[O:5]>>[CH3:1][NH2:2].O[C:3]([CH3:4])=[O:5]"
        },
        7: {
            "smiles": "[CH3:1][NH2:2].[C:3]([CH3:4])(=[O:5])[OH:6]>>[CH3:1][NH:2][C:3]([CH3:4])=[O:5].[OH2:6]"
        },
        8: {
            "smiles": "[C:3]([CH3:4])(=[O:5])[OH:6].[CH3:1][NH2:2]>>[OH2:6].[CH3:1][NH:2][C:3]([CH3:4])=[O:5]"
        },
        9: {
            "smiles": "[CH3:1][NH:2][C:3]([CH3:4])=[O:5]>O>[CH3:1][NH2:2].O[C:3]([CH3:4])=[O:5]"
        },
    }
    # initialize the constructor
    template_constructor = TemplateConstructor(identity_property_name="smarts")
    results = {}
    for k, v in reactions.items():
        if template := template_constructor.build_from_reaction_string(
            reaction_string=v.get("smiles"), inp_fmt="smiles"
        ):
            h = template.hash_map
        else:
            h = None
        results[k] = h
        # print(k, h)

    # the hashes are calculated and have a non-null (None) value
    assert results.get(0).get("reactants")
    assert results.get(0).get("reagents")
    assert results.get(0).get("products")
    assert results.get(0).get("r_p")
    assert results.get(0).get("r_r_p")
    assert results.get(0).get("u_r_p")
    assert results.get(0).get("u_r_r_p")

    # the reactant hash is insensitive to the input order of reactants
    # (reaction canonicalization OK)
    assert results.get(0).get("reactants") == results.get(1).get("reactants")
    # the product hash is insensitive to the input order of products
    # (reaction canonicalization OK)
    assert results.get(7).get("products") == results.get(8).get("products")
    # the machinery does break when the reactants are missing:
    # Template is None
    assert results.get(2) is None
    # the machinery does not break when the agents are missing
    assert results.get(3).get("reagents")
    # the machinery does break when the products are missing:
    # Template is None
    assert results.get(4) is None
    # reagents are happily ignored
    # there is a special hash for missing roles
    # (it is the hash of an empty string)
    assert results.get(3).get("reagents") == create_hash("")
    # the reactant and products hashes are conserved
    # even when the reagents are missing
    assert results.get(0).get("reactants") == results.get(5).get("reactants")
    assert results.get(0).get("products") == results.get(5).get("products")
    # the base r>p hash is conserved if the agents are missing in one reaction
    assert results.get(0).get("r_p") == results.get(5).get("r_p")
    # the full r>a>p hash is conserved  if the reagents are
    # missing in one reaction (reagents are ignored!!)
    assert results.get(0).get("r_r_p") == results.get(5).get("r_r_p")
    # the base r>>p hash is not conserved if the reaction is reversed
    assert results.get(0).get("r_p") != results.get(6).get("r_p")
    # the full r>a>p hash is not conserved if the reaction is reversed
    assert results.get(0).get("r_r_p") != results.get(6).get("r_r_p")
    # the reversible base r<>p hash is not conserved
    # if the reaction is reversed (this comes from rdchiral template extraction)
    # in some special cases it might be true, but it not necessarily is
    assert results.get(0).get("u_r_p") != results.get(9).get("u_r_p")
    assert results.get(3).get("u_r_p") != results.get(6).get("u_r_p")
    # the reversible full r<a>p hash is not conserved if the
    # reaction is reversed (this comes from rdchiral teplate extraction)
    # in some special cases it might be true, but it not necessarily is
    assert results.get(0).get("u_r_r_p") != results.get(9).get("u_r_r_p")
    assert results.get(3).get("u_r_r_p") != results.get(6).get("u_r_r_p")


# Disconnection tests
def test_disconnection_equality():
    test_set = [
        {
            "name": "rnx_1",  # fully balanced amide formation from carboxylic acid and amine
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {},
        },
        {
            "name": "rnx_2",  # fully balanced amide hydrolysis
            "smiles": "[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]>>[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]",
            "expected": {},
        },
        {
            "name": "rnx_3",
            # fully balanced intramolecular michael addition ring forming, one new bond and one changed bond
            "smiles": r"[CH3:1][CH2:2][C:3](=[O:4])[c:5]1[cH:6][cH:7][cH:8][n:9]1[CH2:10]/[CH:11]=[CH:12]\[C:13](=[O:14])[O:15][CH3:16]>>[CH3:1][CH:2]1[CH:11]([CH2:10][n:9]2[cH:8][cH:7][cH:6][c:5]2[C:3]1=[O:4])[CH2:12][C:13](=[O:14])[O:15][CH3:16]",
            "expected": {},
        },
        {
            "name": "rnx_4",  # fully balanced diels-alder product regioisomer 1
            "smiles": "[CH3:6][CH:7]=[CH2:8].[CH3:1][C:2](=[CH2:3])[CH:4]=[CH2:5]>>[CH3:6][CH:7]1[CH2:8][CH2:3][C:2](=[CH:4][CH2:5]1)[CH3:1] ",
            "expected": {},
        },
        {
            "name": "rnx_5",  # fully balanced diels-alder product regioisomer 2
            "smiles": "[CH3:6][CH:7]=[CH2:8].[CH3:1][C:2](=[CH2:3])[CH:4]=[CH2:5]>>[CH3:6][CH:7]1[CH2:8][CH2:5][CH:4]=[C:2]([CH2:3]1)[CH3:1]",
            "expected": {},
        },
        {
            "name": "rnx_6",  # fully balanced amide formation from acyl chloride and amine (same disconnection as rnx_1)
            "smiles": "[CH3:1][C:2]([Cl:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[Cl:3][H]",
            "expected": {},
        },
        {
            "name": "rnx_7",  # not fully balanced reaction
            "smiles": "O[C:2](=[O:1])[c:3]1[cH:4][cH:5][cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]12.CN>ClCCl.ClC(=O)C(Cl)=O>C[NH:13][C:2](=[O:1])[c:3]1[cH:4][cH:5][cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]12",
            "expected": {},
        },
    ]
    ce_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    results = {
        item.get("name"): ce_constructor.build_from_reaction_string(
            reaction_string=item.get("smiles"), inp_fmt="smiles"
        )
        for item in test_set
    }
    # regioisomer products from the same reactants: disconnection is different (fragments might be the same)
    assert results.get("rnx_4").disconnection != results.get("rnx_5").disconnection
    # same product from two sets of equivalent reactants (at synthol level)
    assert results.get("rnx_1").disconnection == results.get("rnx_6").disconnection


def test_disconnection():
    test_set = {
        # fully balanced amide formation from carboxylic acid and amine
        "rxn_1": {
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {},
        },
        # fully balanced amide hydrolysis
        "rxn_2": {
            "smiles": "[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]>>[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]",
            "expected": {},
        },
        # fully balanced intramolecular michael addition ring forming, one new bond and one changed bond
        "rxn_3": {
            "smiles": r"[CH3:1][CH2:2][C:3](=[O:4])[c:5]1[cH:6][cH:7][cH:8][n:9]1[CH2:10]/[CH:11]=[CH:12]\[C:13](=[O:14])[O:15][CH3:16]>>[CH3:1][CH:2]1[CH:11]([CH2:10][n:9]2[cH:8][cH:7][cH:6][c:5]2[C:3]1=[O:4])[CH2:12][C:13](=[O:14])[O:15][CH3:16]",
        },
        # fully balanced diels-alder product regioisomer 1
        "rxn_4": {
            "smiles": "[CH3:6][CH:7]=[CH2:8].[CH3:1][C:2](=[CH2:3])[CH:4]=[CH2:5]>>[CH3:6][CH:7]1[CH2:8][CH2:3][C:2](=[CH:4][CH2:5]1)[CH3:1] ",
        },
        # fully balanced diels-alder product regioisomer 2
        "rxn_5": {
            "smiles": "[CH3:6][CH:7]=[CH2:8].[CH3:1][C:2](=[CH2:3])[CH:4]=[CH2:5]>>[CH3:6][CH:7]1[CH2:8][CH2:5][CH:4]=[C:2]([CH2:3]1)[CH3:1]",
        },
        # fully balanced amide formation from acyl chloride and amine (same disconnection as rxn_1)
        "rxn_6": {
            "smiles": "[CH3:1][C:2]([Cl:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[Cl:3][H]",
        },
        # not fully balanced reaction
        "rxn_7": {
            "smiles": "O[C:2](=[O:1])[c:3]1[cH:12][cH:7][cH:6][cH:5][cH:4]1.CN>ClCCl.ClCC(Cl)=O>C[NH:13][C:2](=[O:1])[c:3]1[cH:12][cH:7][cH:6][cH:5][cH:4]1",
        },
        # ketone reduction to alcohol (two hydrogenated atoms)
        "rxn_9": {
            "smiles": "[CH3:1][C:2](=[O:4])[CH3:3]>>[CH3:1][CH:2]([CH3:3])[OH:4]",
        },
        # double alcohol deprotection (2*ester -> 2*alcohol)
        "rxn_10": {
            "smiles": "C[O:1][CH2:2][CH2:3][CH2:4][O:5]C>>[OH:5][CH2:4][CH2:3][CH2:2][OH:1]",
        },
        # double hydrogenation of a -C#N leading to two H added on N and two on C
        "rxn_11": {"smiles": "[CH3:3][C:2]#[N:1]>>[CH3:3][CH2:2][NH2:1]"},
        # single hydrogenation of a -C#N leading to 1 H added on N and one on C
        "rxn_12": {"smiles": "[CH3:3][C:2]#[N:1]>>[CH3:3][CH:2]=[NH:1]"},
        # amine deprotection: same product as rxn_11 but different disconnection
        "rxn_13": {"smiles": "[CH3:1][CH2:2][NH:3]C(O)=O>>[CH3:1][CH2:2][NH2:3]"},
        # Cl replacement: same product as rxn_11 but different disconnection,
        "rxn_14": {"smiles": "[CH3:2][CH2:3]Cl.[NH3:1]>>[CH3:2][CH2:3][NH2:1]"},
    }

    # initialize the constructor
    chemical_equation_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles", chemical_equation_identity_name="r_p"
    )

    results = {}
    for k, v in test_set.items():
        smiles_input = v.get("smiles")
        chemical_equation = chemical_equation_constructor.build_from_reaction_string(
            reaction_string=smiles_input, inp_fmt="smiles"
        )
        smiles_actual = chemical_equation.smiles
        disconnection_from_ce = chemical_equation.disconnection

        results[k] = {
            "smiles_input": smiles_input,
            "smiles_actual": smiles_actual,
            "chemical_equation": chemical_equation,
            "disconnection_from_ce": disconnection_from_ce,
        }
    # check that the chemical equation is generated for each reaction
    for k, v in results.items():
        chemical_equation = v.get("chemical_equation")
        assert chemical_equation, f"The reaction {k} yields a null chemical equation"

    # check that the disconnection generated from the chemical_equation is not null
    for k, v in results.items():
        disconnection_from_ce = v.get("disconnection_from_ce")
        assert disconnection_from_ce, f"The reaction {k} yields a null disconnection"

    # check that the disconnection is different for reactions giving the same product from different reactants
    reaction_list = ["rxn_11", "rxn_13", "rxn_14"]
    couples = combinations(reaction_list, 2)

    for a, b in couples:
        chemical_equation_a = results.get(a).get("chemical_equation")
        chemical_equation_b = results.get(b).get("chemical_equation")
        disconnection_a = chemical_equation_a.disconnection
        disconnection_b = chemical_equation_b.disconnection
        assert (
            a != b
        ), f"You are comparing the same thing, please review the reaction_list in the test set:  {a} {b}"

        assert disconnection_a.molecule.uid == disconnection_b.molecule.uid, (
            f"This is not a fair comparison because "
            f"the product is not the same for:  {a} {b} "
        )

        assert disconnection_a.uid != disconnection_b.uid, (
            "The disconnection identifier is the same for some "
            "reactions that have the same products but different reactants: \n"
            + f"{a}: {disconnection_a.uid} \n"
            f"{b}: {disconnection_b.uid} \n"
        )
        assert disconnection_a.hash_map.get(
            "disconnection_summary"
        ) != disconnection_b.hash_map.get("disconnection_summary"), (
            "The disconnection summary is the same for some reactions that have the same "
            "products but different reactants: \n"
            + f'{a}: {disconnection_a.hash_map.get("disconnection_summary")} \n'
            f'{b}: {disconnection_b.hash_map.get("disconnection_summary")} \n'
        )

    # check that the disconnection is different for reactions giving the different products from the same reactants
    couples = [["rxn_4", "rxn_5"]]
    for a, b in couples:
        chemical_equation_a = results.get(a).get("chemical_equation")
        chemical_equation_b = results.get(b).get("chemical_equation")
        disconnection_a = chemical_equation_a.disconnection
        disconnection_b = chemical_equation_b.disconnection
        assert (
            a != b
        ), f"You are comparing the same thing, please review the reactions selected from test set:  {a} {b}"

        assert disconnection_a.molecule.uid != disconnection_b.molecule.uid, (
            f"We expect the product to be different "
            f"the product is the same for:  {a} {b} "
        )

        assert disconnection_a.uid != disconnection_b.uid, (
            "The disconnection identifier is the same for some "
            "reactions that have different products but same reactants: \n"
            + f"{a}: {disconnection_a.uid} \n"
            f"{b}: {disconnection_b.uid} \n"
        )
        assert disconnection_a.hash_map.get(
            "disconnection_summary"
        ) != disconnection_b.hash_map.get("disconnection_summary"), (
            "The disconnection summary is the same for some reactions that have the same "
            "products but different reactants: \n"
            + f'{a}: {disconnection_a.hash_map.get("disconnection_summary")} \n'
            f'{b}: {disconnection_b.hash_map.get("disconnection_summary")} \n'
        )

    # check that we have a disconnection for "deprotection" type reactions: a group of atoms is replaced by H
    reaction_list = ["rxn_10", "rxn_13"]
    for item in reaction_list:
        chemical_equation = results.get(item).get("chemical_equation")
        disconnection = chemical_equation.disconnection
        assert len(disconnection.reacting_atoms) > 0, f"No reacting atoms for {item}"
        assert (
            len(disconnection.hydrogenated_atoms) > 0
        ), f"No hydrogenated atoms for {item}"

    # check that we have a disconnection for "hydrogenation" type reactions: addition of H atoms
    reaction_list = ["rxn_9", "rxn_11", "rxn_12"]
    for item in reaction_list:
        chemical_equation = results.get(item).get("chemical_equation")
        disconnection = chemical_equation.disconnection
        assert len(disconnection.reacting_atoms) > 0, f"No reacting atoms for {item}"
        assert (
            len(disconnection.hydrogenated_atoms) > 0
        ), f"No hydrogenated atoms for {item}"

    # check that single and double "hydrogenation" give different disconnections (the product changes)
    couples = [["rxn_11", "rxn_12"]]
    for a, b in couples:
        chemical_equation_a = results.get(a).get("chemical_equation")
        chemical_equation_b = results.get(b).get("chemical_equation")
        disconnection_a = chemical_equation_a.disconnection
        disconnection_b = chemical_equation_b.disconnection
        assert (
            a != b
        ), f"You are comparing the same thing, please review the reactions selected from test set:  {a} {b}"
        assert disconnection_a.uid != disconnection_b.uid, (
            "The disconnection identifier is the same for some "
            "reactions that have different products but same reactants: \n"
            + f"{a}: {disconnection_a.uid} \n"
            f"{b}: {disconnection_b.uid} \n"
        )
        assert disconnection_a.hash_map.get(
            "disconnection_summary"
        ) != disconnection_b.hash_map.get("disconnection_summary"), (
            "The disconnection summary is the same for some reactions that have the same "
            "products but different reactants: \n"
            + f'{a}: {disconnection_a.hash_map.get("disconnection_summary")} \n'
            f'{b}: {disconnection_b.hash_map.get("disconnection_summary")} \n'
        )


def test_correspondence_between_molecule_ce():
    mol_smiles = "C/C=C/C(O)c1ccccc1[N+](=O)[O-]"
    mol = MoleculeConstructor("smiles").build_from_molecule_string(mol_smiles, "smiles")
    ce_smiles = "C/C=C/Br.O=Cc1ccccc1[N+](=O)[O-]>>C/C=C/C(O)c1ccccc1[N+](=O)[O-]"
    ce = ChemicalEquationConstructor("smiles", "r_p").build_from_reaction_string(
        ce_smiles, "smiles"
    )
    assert (
        mol.hash_map["inchikey_KET_15T"]
        == ce.get_products()[0].hash_map["inchikey_KET_15T"]
    )


def test_disconnection_depiction():
    test_set = [
        {
            "name": "rnx_1",  # fully balanced amide formation from carboxylic acid and amine
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {},
        },
        {
            "name": "rnx_2",  # fully balanced amide hydrolysis
            "smiles": "[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]>>[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]",
            "expected": {},
        },
        {
            "name": "rnx_3",
            # fully balanced intramolecular michael addition ring forming, one new bond and one changed bond
            "smiles": r"[CH3:1][CH2:2][C:3](=[O:4])[c:5]1[cH:6][cH:7][cH:8][n:9]1[CH2:10]/[CH:11]=[CH:12]\[C:13](=[O:14])[O:15][CH3:16]>>[CH3:1][CH:2]1[CH:11]([CH2:10][n:9]2[cH:8][cH:7][cH:6][c:5]2[C:3]1=[O:4])[CH2:12][C:13](=[O:14])[O:15][CH3:16]",
            "expected": {},
        },
        {
            "name": "rnx_4",  # fully balanced diels-alder product regioisomer 1
            "smiles": "[CH3:6][CH:7]=[CH2:8].[CH3:1][C:2](=[CH2:3])[CH:4]=[CH2:5]>>[CH3:6][CH:7]1[CH2:8][CH2:3][C:2](=[CH:4][CH2:5]1)[CH3:1] ",
            "expected": {},
        },
        {
            "name": "rnx_5",  # fully balanced diels-alder product regioisomer 2
            "smiles": "[CH3:6][CH:7]=[CH2:8].[CH3:1][C:2](=[CH2:3])[CH:4]=[CH2:5]>>[CH3:6][CH:7]1[CH2:8][CH2:5][CH:4]=[C:2]([CH2:3]1)[CH3:1]",
            "expected": {},
        },
        {
            "name": "rnx_6",  # fully balanced amide formation from acyl chloride and amine (same disconnection as rnx_1)
            "smiles": "[CH3:1][C:2]([Cl:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[Cl:3][H]",
            "expected": {},
        },
        {
            "name": "rnx_7",  # not fully balanced reaction
            "smiles": "[CH3:3][C:2](O)=[O:1].CN>ClCCl.ClC(=O)C(Cl)=O>C[NH:13][C:2]([CH3:3])=[O:1]",
            "expected": {},
        },
        {
            "name": "rxn_8",  # the new bond only involves hydrogen
            "smiles": "[N:8]#[C:7][C:6]1=[CH:5][CH:4]=[CH:3][CH:2]=[CH:1]1>>[NH2:8][CH2:7][C:6]1=[CH:5][CH:4]=[CH:3][CH:2]=[CH:1]1",
            "expected": {},
        },
    ]
    ce_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    results = {
        item.get("name"): ce_constructor.build_from_reaction_string(
            reaction_string=item.get("smiles"), inp_fmt="smiles"
        )
        for item in test_set
    }
    # import linchemin.IO.io as lio
    for name, ce in results.items():
        # print(disconnection.to_dict())
        # rdrxn = cif.rdrxn_from_string(input_string=item.get('smiles'), inp_fmt='smiles')
        depiction_data = cid.draw_disconnection(disconnection=ce.disconnection)
        # lio.write_rdkit_depict(data=depiction_data, file_path=f"{name}_disconnection.png")
        assert depiction_data

        depiction_data = cid.draw_fragments(rdmol=ce.disconnection.rdmol_fragmented)
        assert depiction_data
        # lio.write_rdkit_depict(data=depiction_data, file_path=f"{name}_fragment.png")
        depiction_data = cid.draw_reaction(rdrxn=ce.rdrxn)
        assert depiction_data
        # lio.write_rdkit_depict(data=depiction_data, file_path=f"{name}_reaction.png")
        # print(f"\n{item.get('name')} {disconnection.__dict__}")


def test_real_ces():
    test_data = {
        0: {
            "smiles": "Cl[Cl:1].ClCl.[Cl:11][Cl:14].Cl[Al](Cl)Cl.O.[cH:2]1[cH:3][cH:4][c:5]([O:6][c:7]2[cH:8][cH:9][cH:10][cH:12][cH:13]2)[cH:15][cH:16]1>>[Cl:1][c:2]1[cH:3][cH:4][c:5]([O:6][c:7]2[cH:8][cH:9][c:10]([Cl:11])[cH:12][c:13]2[Cl:14])[cH:15][cH:16]1",
            "expected": "Cl[Cl:1].[Cl:11][Cl:14].[cH:2]1[cH:3][cH:4][c:5]([O:6][c:7]2[cH:8][cH:9][cH:10][cH:12][cH:13]2)[cH:15][cH:16]1>ClCl.Cl[Al](Cl)Cl.O>[Cl:1][c:2]1[cH:3][cH:4][c:5]([O:6][c:7]2[cH:8][cH:9][c:10]([Cl:11])[cH:12][c:13]2[Cl:14])[cH:15][cH:16]1",
        },
        1: {
            "smiles": "CC(=O)[O:1][c:2]1[cH:8][cH:7][cH:6][c:4]([Cl:5])[cH:3]1.CCO.Cl>>[OH:1][c:2]1[cH:8][cH:7][cH:6][c:4]([Cl:5])[cH:3]1",
            "expected": "CC(=O)[O:1][c:2]1[cH:3][c:4]([Cl:5])[cH:6][cH:7][cH:8]1>CCO.Cl>[OH:1][c:2]1[cH:3][c:4]([Cl:5])[cH:6][cH:7][cH:8]1",
        },
        2: {
            "smiles": "[CH3:1][CH2:2][O:7][C:6](=[O:5])[CH2:8][O:9][CH3:10].[OH:3][Na:4]>>[CH3:1][CH2:2][O:3][Na:4].[CH3:10][O:9][CH2:8][C:6]([OH:7])=[O:5]",
            "expected1": "[CH3:1][CH2:2][O:7][C:6](=[O:5])[CH2:8][O:9][CH3:10].[OH:3][Na:4]>>[CH3:1][CH2:2][O:3][Na:4].[O:5]=[C:6]([OH:7])[CH2:8][O:9][CH3:10]",
            "expected2": "[CH3:1][CH2:2][O:7][C:6](=[O:5])[CH2:8][O:9][CH3:10]>[OH:3][Na:4]>[CH3:1][CH2:2][O:3][Na:4].[O:5]=[C:6]([OH:7])[CH2:8][O:9][CH3:10]",
        },
        3: {"smiles": "[CH3:1][CH:7]=[NH:8]>>[CH3:1][CH2:7][NH2:8]"},
        4: {
            "smiles": "[CH3:1][CH2:7][NH:8][C:9]([CH3:10])=[O:11]>>[CH3:1][CH2:7][NH2:8]"
        },
    }

    ce_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    multi_disconnections = []
    for i, test in test_data.items():
        if i == 2:
            # same reaction string but different desired product lead to different ChemicalEquations
            # with different disconnections
            ce1 = ce_constructor.build_from_reaction_string(
                test["smiles"],
                inp_fmt="smiles",
                desired_product="[CH3:1][CH2:2][O:3][Na:4]",
            )
            ce2 = ce_constructor.build_from_reaction_string(
                test["smiles"],
                inp_fmt="smiles",
                desired_product="[CH3:10][O:9][CH2:8][C:6]([OH:7])=[O:5]",
            )
            assert ce1 != ce2
            assert ce1.smiles == test["expected1"]
            assert ce2.smiles == test["expected2"]
            assert ce1.disconnection != ce2.disconnection
            for n, ce in enumerate([ce1, ce2]):
                depiction_data = cid.draw_disconnection(ce.disconnection)
                assert depiction_data
                # lio.write_rdkit_depict(data=depiction_data, file_path=f"disconnection_{n}_{i}.png")
            continue

        ce = ce_constructor.build_from_reaction_string(
            test["smiles"],
            inp_fmt="smiles",
        )
        if i in [3, 4]:
            multi_disconnections.append(ce.disconnection)
            continue
        assert ce.smiles == test["expected"]
        assert ce.disconnection
        depiction_data = cid.draw_disconnection(ce.disconnection)
        # lio.write_rdkit_depict(data=depiction_data, file_path=f"disconnection_{i}.png")
    # multiple disconnection can be depicted on the same product Molecule
    depiction_data = cid.draw_multiple_disconnections(
        disconnections=multi_disconnections
    )
    assert depiction_data
    # lio.write_rdkit_depict(data=depiction_data, file_path="multi_disconnections.png")


def test_chem_equation_w_enhanced_stereo():
    # reaction whose product has enhanced stereo
    reaction_string_w_es = (
        r"[CH3:1][C@H:2]([Cl:3])[C:4]([CH3:6])=O.[FH:5]>>"
        r"[CH3:6][C@@H:4]([F:5])[C@H:2]([CH3:1])[Cl:3] |&1:2,8,12,r|"
    )
    # reaction without enhanced stereo
    reaction_string = "[CH3:1][C@H:2]([Cl:3])[C:4]([CH3:6])=O.[FH:5]>>[CH3:6][C@@H:4]([F:5])[C@H:2]([CH3:1])[Cl:3]"
    ce_constructor = ChemicalEquationConstructor("cx_smiles", "r_r_p")
    ce_w_es = ce_constructor.build_from_reaction_string(reaction_string_w_es, "smiles")
    ce = ce_constructor.build_from_reaction_string(reaction_string, "smiles")

    # ChemicalEquation identity is influenced by enhanced stereochemistry if the molecular uid is cx_smiles
    assert ce != ce_w_es

    # Template is not influenced by enhanced stereochemistry in the desired product
    assert ce.template == ce_w_es.template

    # The disconnection is impacted by enhanced stereochemistry in the desired product
    assert ce.disconnection != ce_w_es.disconnection


def test_chem_equation_w_enhanced_stereo2():
    # reaction whose reactants have enhanced stereochemistry
    s1 = "FC(F)C(F)(Cl)[C@H](F)Cl>>FC(F)C(F)(Cl)C(Cl)=O |o1:3,4,12,13,&6:6,8,15,16,r|"
    # reaction without enhaced stereochemistry
    s2 = "FC(F)C(F)(Cl)[C@H](F)Cl>>FC(F)C(F)(Cl)C(Cl)=O"

    ce_constructor = ChemicalEquationConstructor("cx_smiles", "r_r_p")
    ce_w_es = ce_constructor.build_from_reaction_string(s1, "smiles")
    ce = ce_constructor.build_from_reaction_string(s2, "smiles")
    assert ce_w_es != ce
    # The disconnection is not impacted by enhanced stereochemistry in the reactants/reagents
    assert ce.disconnection == ce_w_es.disconnection


def test_molecule_is_always_sanitized():
    mol_smiles = "O=C(CBr)NC1=CC=C(C2=NOC=N2)C=C1"
    mol_constructor = MoleculeConstructor("smiles")
    mol = mol_constructor.build_from_molecule_string(mol_smiles, "smiles")

    ce_smiles = "COC(=O)CBr.Nc1ccc(-c2ncon2)cc1>>O=C(CBr)NC1=CC=C(C2=NOC=N2)C=C1"
    ce_constructor = ChemicalEquationConstructor("smiles", "r_p")
    ce = ce_constructor.build_from_reaction_string(ce_smiles, "smiles")
    product = ce.get_products()[0]
    assert product == mol
