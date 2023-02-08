import pytest

from linchemin.cheminfo.reaction import ChemicalEquation, ChemicalEquationConstructor
from linchemin.rem.node_metrics import node_score_calculator, reaction_mapping


def test_factory():
    smile1 = "CC(O)=O.CN>>CNC(C)=O"

    chemical_equation_constructor = ChemicalEquationConstructor(
        identity_property_name="smiles"
    )
    reaction1 = chemical_equation_constructor.build_from_reaction_string(
        reaction_string=smile1, inp_fmt="smiles"
    )
    with pytest.raises(KeyError) as ke:
        node_score_calculator(reaction1, "some_score")
    assert "KeyError" in str(ke.type)


def test_CDScores():
    smile1 = "CC(O)=O.CN>>CNC(C)=O"
    chemical_equation_constructor1 = ChemicalEquationConstructor(
        identity_property_name="smiles"
    )
    reaction1 = chemical_equation_constructor1.build_from_reaction_string(
        reaction_string=smile1, inp_fmt="smiles"
    )
    smile2 = "CN.CC(O)=O>O>CNC(C)=O"
    chemical_equation_constructor2 = ChemicalEquationConstructor(
        identity_property_name="smiles"
    )
    reaction2 = chemical_equation_constructor2.build_from_reaction_string(
        reaction_string=smile1, inp_fmt="smiles"
    )
    assert node_score_calculator(reaction1, "cdscore") == 0.5
    assert node_score_calculator(reaction2, "cdscore") == 0.5

    with pytest.raises(TypeError) as te:
        node_score_calculator(smile2, "cdscore")
    assert "TypeError" in str(te.type)


def test_reaction_mapping():
    map_dictionaries = {
        "map1": {1: 1, 2: 2, 3: 3, 4: 5, 5: 7, 6: 8, 7: 10},
        "map2": {1: 2, 2: 6, 3: 3, 4: 4, 5: 11, 6: 12},
        "map3": {1: 12, 2: 3, 3: 6, 4: 11, 5: 4, 6: 12},
        "map4": {1: 1, 2: 7, 3: 8},
        "map5": {1: 2, 2: 4, 3: 5, 4: 6, 5: 8, 6: 9},
        "map6": {1: 3, 2: 5, 3: 6, 4: 9, 5: 10},
        "map7": {1: 0, 2: 1, 3: 2, 4: 0},
        "map8": {1: 0, 2: 1, 3: 5},
    }
    # atoms are transferred in the first step
    assert reaction_mapping(map_dictionaries["map1"], map_dictionaries["map2"]) == [
        1,
        3,
    ]
    # all atoms are transferred in the first step
    assert reaction_mapping(map_dictionaries["map2"], map_dictionaries["map3"]) == list(
        map_dictionaries["map2"].keys()
    )
    # no atoms are transferred in the first step
    assert reaction_mapping(map_dictionaries["map2"], map_dictionaries["map4"]) == []
    # atoms are transferred in the first step, but not in the second step
    t1 = reaction_mapping(map_dictionaries["map1"], map_dictionaries["map2"])
    assert (
        reaction_mapping(
            map_dictionaries["map3"], map_dictionaries["map4"], ids_transferred_atoms=t1
        )
        == []
    )
    # atoms are transferred in both the first and the second step
    assert reaction_mapping(
        map_dictionaries["map5"], map_dictionaries["map6"], ids_transferred_atoms=t1
    ) == [2]
    # no atoms are transferred in the first step, but atoms are transferred in the second step
    assert reaction_mapping(
        map_dictionaries["map5"], map_dictionaries["map6"], ids_transferred_atoms=[]
    ) == [2, 3, 4]
    # unmapped atoms (map index = 0) are ignored
    assert reaction_mapping(map_dictionaries["map7"], map_dictionaries["map8"]) == [2]
