import pytest

from linchemin.cheminfo.constructors import ChemicalEquationConstructor
from linchemin.rem.node_descriptors import (
    NoMapping,
    chemical_equation_descriptor_calculator,
)


def test_factory():
    smile1 = "CC(O)=O.CN>>CNC(C)=O"

    chemical_equation_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles"
    )
    reaction1 = chemical_equation_constructor.build_from_reaction_string(
        reaction_string=smile1, inp_fmt="smiles"
    )
    with pytest.raises(KeyError) as ke:
        chemical_equation_descriptor_calculator(reaction1, "some_score")
    assert "KeyError" in str(ke.type)


def test_CDScores():
    smile1 = "CC(O)=O.CN>>CNC(C)=O"
    chemical_equation_constructor1 = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles"
    )
    reaction1 = chemical_equation_constructor1.build_from_reaction_string(
        reaction_string=smile1, inp_fmt="smiles"
    )
    smile2 = "CN.CC(O)=O>O>CNC(C)=O"
    chemical_equation_constructor2 = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles"
    )
    reaction2 = chemical_equation_constructor2.build_from_reaction_string(
        reaction_string=smile1, inp_fmt="smiles"
    )
    assert chemical_equation_descriptor_calculator(reaction1, "ce_convergence") == 0.5
    assert chemical_equation_descriptor_calculator(reaction2, "ce_convergence") == 0.5

    with pytest.raises(TypeError) as te:
        chemical_equation_descriptor_calculator(smile2, "ce_convergence")
    assert "TypeError" in str(te.type)


def test_atom_effectiveness():
    ce_test_set = {
        # fully efficient reaction (ae = 1)
        0: {
            "smiles": "[N:8]#[C:7][C:6]1=[CH:5][CH:4]=[CH:3][CH:2]=[CH:1]1>>[NH2:8][CH2:7][C:6]1=[CH:5][CH:4]=[CH:3][CH:2]=[CH:1]1",
            "expected": 1.0,
        },
        # reaction with missing reactants (ae > 1)
        1: {
            "smiles": "[CH3:4][NH2:5]>>[CH3:5][NH:4][C:3]([CH3:2])=[O:1]",
            "expected": 2.5,
        },
        # reaction with by/side-products (ea < 1)
        2: {
            "smiles": "[CH3:0][O:0][C:3]([CH3:2])=[O:1].[CH3:4][NH2:5]>>[CH3:4][NH:5][C:3]([CH3:2])=[O:1]",
            "expected": 0.7,
        },
    }
    chemical_equation_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    for d in ce_test_set.values():
        ce = chemical_equation_constructor.build_from_reaction_string(
            reaction_string=d["smiles"], inp_fmt="smiles"
        )
        ae = chemical_equation_descriptor_calculator(ce, "ce_atom_effectiveness")
        assert round(ae, 1) == d["expected"]


def test_hypsicity():
    ce_test_set = {
        # reaction without mapping raises error
        0: {"smiles": "N#CC1=CC=CC=C1>>NCC1=CC=CC=C1", "expected": None},
        # no redox reaction
        1: {
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4]",
            "expected": 0.0,
        },
        # reduction
        2: {
            "smiles": "[CH3:2][C:3]([CH3:4])=[O:1]>>[CH3:2][CH:3]([CH3:4])[OH:1]",
            "expected": -2.0,
        },
        # oxygen exchange without change in ox state
        3: {"smiles": "[O-:3][C:2]([O-:4])=[O:1]>>[O:1]=[C:2]=[O:3]", "expected": 0.0},
        # reduction from triple CN bond to single CN bond
        4: {
            "smiles": "[N:8]#[C:7][C:6]1=[CH:5][CH:4]=[CH:3][CH:2]=[CH:1]1>>[NH2:8][CH2:7][C:6]1=[CH:5][CH:4]=[CH:3][CH:2]=[CH:1]1",
            "expected": -4.0,
        },
        # reductive amination giving an imine -> no change detected
        5: {
            "smiles": "[CH3:1][C:2]([CH3:3])=[O:4].[CH3:5][NH:6][CH3:7]>>[CH3:1][C:2]([CH3:3])=[N+:6]([CH3:5])[CH3:7]",
            "expected": 0.0,
        },
        # reductive amination giving an amine -> change detected
        6: {
            "smiles": "[CH3:1][C:2]([CH3:3])=[O:4].[CH3:5][NH:6][CH3:7]>>[CH3:3][CH:2]([CH3:1])[N:6]([CH3:7])[CH3:5]",
            "expected": -2.0,
        },
        # oxidation from ketone to acid
        7: {
            "smiles": "[CH3:1][C:2]([CH3])=[O:3]>>[CH3:1][C:2]([OH])=[O:3]",
            "expected": 1.0,
        },
        # intramolecular redox
        8: {
            "smiles": "[O-:12][N+:10](=[O:11])[C:5]1=[CH:4][CH:3]=[CH:2][CH:1]=[C:6]1[CH:7]=[O:8]>>[NH2:10][C:5]1=[CH:4][CH:3]=[CH:2][CH:1]=[C:6]1[C:7]([OH:9])=[O:8]",
            "expected": -4.0,
        },
        # redox isomerization of
        9: {
            "smiles": r"[CH3:6][O:5][CH2:4]\[CH:3]=[CH:2]\[CH3:1]>>[CH3:1][CH2:2]\[CH:3]=[CH:4]\[O:5][CH3:6]",
            "expected": 0.0,
        },
    }
    chemical_equation_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    for i, d in ce_test_set.items():
        ce = chemical_equation_constructor.build_from_reaction_string(
            reaction_string=d["smiles"], inp_fmt="smiles"
        )
        if i == 0:
            with pytest.raises(NoMapping) as te:
                chemical_equation_descriptor_calculator(ce, "ce_hypsicity")
            assert "NoMapping" in str(te.type)
        else:
            delta = chemical_equation_descriptor_calculator(ce, "ce_hypsicity")
            assert delta == d["expected"]
