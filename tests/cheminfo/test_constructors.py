import unittest

import pytest

import linchemin.cheminfo.functions as cif
from linchemin.cheminfo.chemical_hashes import UnavailableMolIdentifier
from linchemin.cheminfo.constructors import (
    ChemicalEquationConstructor,
    InvalidMoleculeInput,
    MoleculeConstructor,
    UnparsableMolecule,
    UnparsableReaction,
)
from linchemin.cheminfo.models import Molecule


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


@pytest.fixture
def molecule_constructor():
    return MoleculeConstructor()


def test_init_default_values():
    constructor = MoleculeConstructor()
    assert constructor.molecular_identity_property_name == "smiles"
    assert constructor.hash_list == {
        "inchikey_KET_15T",
        "smiles",
        "cx_smiles",
        "inchi_key",
    }


def test_init_custom_values():
    constructor = MoleculeConstructor("smiles", ["inchi", "inchikey"])
    assert constructor.molecular_identity_property_name == "smiles"
    assert constructor.hash_list == {"smiles", "inchi", "inchikey"}


def test_build_from_molecule_string_valid(molecule_constructor, monkeypatch):
    # Mock the rdmol_from_string function
    def mock_rdmol_from_string(*args, **kwargs):
        return cif.Chem.MolFromSmiles("CC")

    monkeypatch.setattr(cif, "rdmol_from_string", mock_rdmol_from_string)

    # Mock the Molecule creation
    monkeypatch.setattr(Molecule, "__init__", lambda *args, **kwargs: None)

    result = molecule_constructor.build_from_molecule_string("CC", "smiles")
    assert isinstance(result, Molecule)


def test_build_from_molecule_string_unparsable(molecule_constructor):
    with pytest.raises(UnparsableMolecule):
        molecule_constructor.build_from_molecule_string("Invalid", "smiles")


def test_build_from_rdmol_valid(molecule_constructor, monkeypatch):
    rdmol = cif.Chem.MolFromSmiles("CC")

    monkeypatch.setattr(cif, "remove_rdmol_atom_mapping", lambda rdmol: rdmol)
    monkeypatch.setattr(cif, "new_molecule_canonicalization", lambda rdmol: rdmol)
    monkeypatch.setattr(cif, "compute_mol_smiles", lambda rdmol: "CC")
    monkeypatch.setattr(
        "linchemin.cheminfo.chemical_hashes.calculate_molecular_hash_map",
        lambda **kwargs: {"inchi": "InChI=1S/C2H6/c1-2/h1-2H3"},
    )
    monkeypatch.setattr("linchemin.utilities.create_hash", lambda x: "mocked_uid")

    # Mock the Molecule creation
    monkeypatch.setattr(Molecule, "__init__", lambda *args, **kwargs: None)

    result = molecule_constructor.build_from_rdmol(rdmol)
    assert isinstance(result, Molecule)


def test_build_from_rdmol_invalid_input(molecule_constructor):
    with pytest.raises(
        InvalidMoleculeInput, match="Input must be a valid RDKit Mol object"
    ):
        molecule_constructor.build_from_rdmol("Not a RDKit Mol")


def test_identical_molecules_with_smiles_identity(molecule_constructor):
    mol_smiles = {
        "mol_0": "CN",  # M1
        "mol_1": "CN",  # M1
        "mol_2": "NC",  # M1
        "mol_3": "CC",  # M2
    }
    molecules = {
        name: molecule_constructor.build_from_molecule_string(smiles, "smiles")
        for name, smiles in mol_smiles.items()
    }
    assert (
        molecules["mol_0"] == molecules["mol_1"]
    )  # identical molecule, identical input string
    assert (
        molecules["mol_0"] == molecules["mol_2"]
    )  # identical molecule, different input string; assess the canonicalization mechanism
    assert molecules["mol_0"] != molecules["mol_3"]  # different molecules


def test_tautomers_with_smiles_identity(molecule_constructor):
    tautomers_smiles = {
        "mol_0": "CC(C)=O",  # M3_T1
        "mol_1": "CC(O)=C",  # M3_T2
        "mol_2": "CC(O)=N",  # M4_T1
        "mol_3": "CC(N)=O",  # M4_T2
        "mol_4": "CCC(C)=O",  # M5_T1
        "mol_5": r"C\C=C(\C)O",  # M5_T2}
    }
    molecules = {
        name: molecule_constructor.build_from_molecule_string(smiles, "smiles")
        for name, smiles in tautomers_smiles.items()
    }
    assert (
        molecules["mol_0"] != molecules["mol_1"]
    )  # same molecule, but different tautomers: smiles fails to capture identity
    assert (
        molecules["mol_2"] != molecules["mol_3"]
    )  # same molecule, but different tautomers: smiles fails to capture identity
    assert (
        molecules["mol_4"] != molecules["mol_5"]
    )  # same molecule, but different tautomers: smiles fails to capture identity


def test_atom_mapping_with_smiles_identity(molecule_constructor):
    mol_smiles = {
        "mol_0": "Cl[C:2]([CH3:1])=[O:3]",  # M6_atom_mapping_1
        "mol_1": "Cl[C:1]([CH3:2])=[O:5]",  # M6_atom_mapping_2}
    }

    molecules = {
        name: molecule_constructor.build_from_molecule_string(smiles, "smiles")
        for name, smiles in mol_smiles.items()
    }
    assert (
        molecules["mol_0"] == molecules["mol_1"]
    )  # same molecule, with different atom mapping are equivalent
    mol0 = molecules["mol_0"].rdmol_mapped
    mol1 = molecules["mol_1"].rdmol_mapped
    d1 = {a.GetIdx(): [a.GetSymbol()] for a in mol0.GetAtoms()}
    d2 = {a.GetIdx(): [a.GetSymbol()] for a in mol1.GetAtoms()}
    assert d1 == d2  # the canonicalization of atoms is the same


def test_identical_molecules_with_inchikey_identity():
    mols = {
        0: {"smiles": "CN"},  # M1
        1: {"smiles": "CN"},  # M1
        2: {"smiles": "NC"},  # M1
        3: {"smiles": "CC"},  # M2
        6: {"smiles": "CC(O)=N"},  # M4_T1
        7: {"smiles": "CC(N)=O"},  # M4_T2
        10: {"smiles": "Cl[C:2]([CH3:1])=[O:3]"},  # M6_atom_mapping_1
        11: {"smiles": "Cl[C:1]([CH3:2])=[O:5]"},  # M6_atom_mapping_2
    }
    molecule_constructor = MoleculeConstructor(
        molecular_identity_property_name="inchi_key"
    )
    molecules = {
        name: molecule_constructor.build_from_molecule_string(
            smiles["smiles"], "smiles"
        )
        for name, smiles in mols.items()
    }
    assert molecules.get(0) == molecules.get(
        1
    )  # identical molecule, identical input string
    assert molecules.get(0) == molecules.get(
        2
    )  # identical molecule, different input string; assess the canonicalization mechanism
    assert molecules.get(0) != molecules.get(3)  # different molecules
    assert molecules.get(6) == molecules.get(
        7
    )  # same molecule, but different tautomers: inchi_key succeeds to capture identity
    assert molecules.get(10) == molecules.get(
        11
    )  # same molecule, but different atom mapping


def test_chemical_equation_constructor_init():
    ce_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    assert ce_constructor.molecular_identity_property_name == "smiles"
    assert ce_constructor.chemical_equation_identity_name == "r_r_p"
    assert (
        ce_constructor.molecule_constructor.molecular_identity_property_name == "smiles"
    )


def test_chemical_equation_constructor():
    reaction_string_reference = "CC(=O)O.CN.CN>O>CNC(C)=O"

    # initialize the constructor
    ce_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )

    for reaction_string_test in [
        "CC(=O)O.CN.CN>O>CNC(C)=O",  # expected smiles
        "CC(=O)O.NC.CN>O>CNC(C)=O",  # test molecule canonicalization: change order of atoms in reactant molecules
        "CN.CC(=O)O.CN>O>CNC(C)=O",  # test reaction canonicalization: change order of molecules in reactants
    ]:
        chemical_equation = ce_constructor.build_from_reaction_string(
            reaction_string=reaction_string_test, inp_fmt="smiles"
        )
        reaction_string_calculated = chemical_equation.smiles
        assert reaction_string_calculated == reaction_string_reference


def test_rp_vs_rrp():
    r_w_reagents = "CC(=O)O.CN.CN>O>CNC(C)=O"
    r_without_reagents = "CC(=O)O.CN.CN>>CNC(C)=O"
    ce_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    ce_w_reagents = ce_constructor.build_from_reaction_string(r_w_reagents, "smiles")
    ce_without_reagents = ce_constructor.build_from_reaction_string(
        r_without_reagents, "smiles"
    )
    # if equivalence is determined by r_r_p, 2 reaction with
    # same reactants and products but different reagents are not equivalent
    assert ce_without_reagents != ce_w_reagents

    ce_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_p",
    )
    ce_w_reagents = ce_constructor.build_from_reaction_string(r_w_reagents, "smiles")
    ce_without_reagents = ce_constructor.build_from_reaction_string(
        r_without_reagents, "smiles"
    )
    # if equivalence is determined by r_p, 2 reaction with
    # same reactants and products but different reagents are equivalent
    assert ce_without_reagents == ce_w_reagents
    assert ce_without_reagents.smiles == ce_w_reagents.smiles


def test_from_string_w_desired_product():
    r_string = "CC(=O)O.CN.CN>O>CNC(C)=O.O"
    desired_product = "O"
    ce_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    ce_constructor.build_from_reaction_string(
        r_string, inp_fmt="smiles", desired_product=desired_product
    )
    assert ce_constructor.desired_product == MoleculeConstructor(
        "smiles"
    ).build_from_molecule_string(desired_product, "smiles")


def test_from_string_without_desired_product():
    r_string = "CC(=O)O.CN.CN>O>O.CNC(C)=O"
    ce_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    ce_constructor.build_from_reaction_string(r_string, inp_fmt="smiles")
    assert ce_constructor.desired_product == MoleculeConstructor(
        "smiles"
    ).build_from_molecule_string("CNC(C)=O", "smiles")


def test_unparsable_string():
    ce_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    with pytest.raises(UnparsableReaction):
        ce_constructor.build_from_reaction_string("not_reaction_string", "smiles")


def test_unmapped():
    r_string = "CC(=O)O.CN.CN>O>O.CNC(C)=O"
    ce_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    ce = ce_constructor.build_from_reaction_string(r_string, inp_fmt="smiles")
    assert ce.role_map
    assert len(ce.catalog) > 0
    assert ce.stoichiometry_coefficients
    assert ce.smiles == "CC(=O)O.CN.CN>O>CNC(C)=O.O"
    assert ce.mapping is None
    assert ce.disconnection is None


def test_mapped():
    r_string = "[BrH:4].[CH2:3]=[CH2:2].[ClH:1]>>[Cl:1][CH2:2][CH2:3][Br:4]"
    ce_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    ce = ce_constructor.build_from_reaction_string(r_string, inp_fmt="smiles")
    assert ce.role_map
    assert len(ce.catalog) > 0
    assert ce.stoichiometry_coefficients
    assert ce.smiles
    assert ce.mapping
    assert ce.disconnection

    # mapping information are not used to determine the identity
    r_string_unmapped = "Br.C=C.Cl>>ClCCBr"
    ce_unmapped = ce_constructor.build_from_reaction_string(
        r_string_unmapped, inp_fmt="smiles"
    )
    assert ce == ce_unmapped
    # mapping information is used to create the smiles
    assert ce.smiles != ce_unmapped.smiles


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
