from linchemin.rem.node_descriptors import node_descriptor_calculator, reaction_mapping, NoMapping
from linchemin.cheminfo.constructors import ChemicalEquationConstructor
import pytest


def test_factory():
    smile1 = 'CC(O)=O.CN>>CNC(C)=O'

    chemical_equation_constructor = ChemicalEquationConstructor(molecular_identity_property_name='smiles')
    reaction1 = chemical_equation_constructor.build_from_reaction_string(
        reaction_string=smile1,
        inp_fmt='smiles')
    with pytest.raises(KeyError) as ke:
        node_descriptor_calculator(reaction1, 'some_score')
    assert "KeyError" in str(ke.type)


def test_CDScores():
    smile1 = 'CC(O)=O.CN>>CNC(C)=O'
    chemical_equation_constructor1 = ChemicalEquationConstructor(molecular_identity_property_name='smiles')
    reaction1 = chemical_equation_constructor1.build_from_reaction_string(
        reaction_string=smile1,
        inp_fmt='smiles')
    smile2 = 'CN.CC(O)=O>O>CNC(C)=O'
    chemical_equation_constructor2 = ChemicalEquationConstructor(molecular_identity_property_name='smiles')
    reaction2 = chemical_equation_constructor2.build_from_reaction_string(
        reaction_string=smile1,
        inp_fmt='smiles')
    assert node_descriptor_calculator(reaction1, 'cdscore') == 0.5
    assert node_descriptor_calculator(reaction2, 'cdscore') == 0.5

    with pytest.raises(TypeError) as te:
        node_descriptor_calculator(smile2, 'cdscore')
    assert "TypeError" in str(te.type)


def test_reaction_mapping():
    map_dictionaries = {
        'map1': {1: 1, 2: 2, 3: 3, 4: 5, 5: 7, 6: 8, 7: 10},
        'map2': {1: 2, 2: 6, 3: 3, 4: 4, 5: 11, 6: 12},
        'map3': {1: 2, 2: 3, 3: 6, 4: 11, 5: 4, 6: 12},
        'map4': {1: 1, 2: 7, 3: 8},
        'map5': {1: 2, 2: 4, 3: 5, 4: 6, 5: 8, 6: 9},
        'map6': {1: 3, 2: 5, 3: 6, 4: 9, 5: 10},
        'map7': {1: 0, 2: 1, 3: 2, 4: 0},
        'map8': {1: 0, 2: 1, 3: 5}
    }
    # atoms are transferred in the first step
    assert reaction_mapping(map_dictionaries['map1'], map_dictionaries['map2']) == [1, 3]
    # all atoms are transferred in the first step
    assert reaction_mapping(map_dictionaries['map2'], map_dictionaries['map3']) == list(map_dictionaries['map2'].keys())
    # no atoms are transferred in the first step
    assert reaction_mapping(map_dictionaries['map2'], map_dictionaries['map4']) == []
    # atoms are transferred in the first step, but not in the second step
    t1 = reaction_mapping(map_dictionaries['map1'], map_dictionaries['map2'])
    assert reaction_mapping(map_dictionaries['map3'], map_dictionaries['map4'], ids_transferred_atoms=t1) == []
    # atoms are transferred in both the first and the second step
    assert reaction_mapping(map_dictionaries['map5'], map_dictionaries['map6'], ids_transferred_atoms=t1) == [2]
    # no atoms are transferred in the first step, but atoms are transferred in the second step
    assert reaction_mapping(map_dictionaries['map5'], map_dictionaries['map6'], ids_transferred_atoms=[]) == [2, 3, 4]
    # unmapped atoms (map index = 0) are ignored
    assert reaction_mapping(map_dictionaries['map7'], map_dictionaries['map8']) == [2]


def test_atom_efficiency():
    ce_test_set = {
        # fully efficient reaction (ae = 1)
        0: {'smiles': '[N:8]#[C:7][C:6]1=[CH:5][CH:4]=[CH:3][CH:2]=[CH:1]1>>[NH2:8][CH2:7][C:6]1=[CH:5][CH:4]=[CH:3][CH:2]=[CH:1]1',
            'expected': 1.},
        # reaction with missing reactants (ae > 1)
        1: {'smiles': '[CH3:4][NH2:5]>>[CH3:5][NH:4][C:3]([CH3:2])=[O:1]',
            'expected': 2.5},
        # reaction with by/side-products (ea < 1)
        2: {'smiles': '[CH3:0][O:0][C:3]([CH3:2])=[O:1].[CH3:4][NH2:5]>>[CH3:4][NH:5][C:3]([CH3:2])=[O:1]',
            'expected': 0.7},
    }
    chemical_equation_constructor = ChemicalEquationConstructor(molecular_identity_property_name='smiles',
                                                                chemical_equation_identity_name='r_r_p')
    for d in ce_test_set.values():
        ce = chemical_equation_constructor.build_from_reaction_string(
            reaction_string=d['smiles'], inp_fmt='smiles')
        ae = node_descriptor_calculator(ce, 'ce_efficiency')
        assert round(ae, 1) == d['expected']


def test_hypsicity():
    ce_test_set = {
        # reaction without mapping raises error
        0: {'smiles': 'N#CC1=CC=CC=C1>>NCC1=CC=CC=C1',
            'expected': None},
        # no redox reaction
        1: {'smiles': '[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4]',
            'expected': 0.0},
        # reduction
        2: {'smiles': '[CH3:2][C:3]([CH3:4])=[O:1]>>[CH3:2][CH:3]([CH3:4])[OH:1]',
            'expected': -2.0},
        # oxygen exchange without change in ox state
        3: {'smiles': '[O-:3][C:2]([O-:4])=[O:1]>>[O:1]=[C:2]=[O:3]',
            'expected': 0.0},
        # reduction from triple CN bond to single CN bond
        4: {'smiles': '[N:8]#[C:7][C:6]1=[CH:5][CH:4]=[CH:3][CH:2]=[CH:1]1>>[NH2:8][CH2:7][C:6]1=[CH:5][CH:4]=[CH:3][CH:2]=[CH:1]1',
            'expected': -4.0},
        # reductive amination giving an imine -> no change detected
        # 5: {'smiles': '[CH3:1][C:2]([CH3:3])=[O:4].[CH3:5][NH:6][CH3:7]>>[CH3:1][C:2]([CH3:3])=[N+:6]([CH3:5])[CH3:7]',
        #     'expected': 0.0},
        # reductive amination giving an amine -> change detected
        6: {'smiles': '[CH3:1][C:2]([CH3:3])=[O:4].[CH3:5][NH:6][CH3:7]>>[CH3:3][CH:2]([CH3:1])[N:6]([CH3:7])[CH3:5]',
            'expected': -2.0},
        # oxidation from ketone to acid
        7: {'smiles': '[CH3:1][C:2]([CH3])=[O:3]>>[CH3:1][C:2]([OH])=[O:3]',
            'expected': 1.0},
        # intramolecular redox
        # 8: {'smiles': '[O-:12][N+:10](=[O:11])[C:5]1=[CH:4][CH:3]=[CH:2][CH:1]=[C:6]1[CH:7]=[O:8]>>[NH2:10][C:5]1=[CH:4][CH:3]=[CH:2][CH:1]=[C:6]1[C:7]([OH:9])=[O:8]',
        #     'expected': -4.0}
    }
    chemical_equation_constructor = ChemicalEquationConstructor(molecular_identity_property_name='smiles',
                                                                chemical_equation_identity_name='r_r_p')
    for i, d in ce_test_set.items():
        ce = chemical_equation_constructor.build_from_reaction_string(reaction_string=d['smiles'], inp_fmt='smiles')
        if i == 0:
            with pytest.raises(NoMapping) as te:
                node_descriptor_calculator(ce, 'ce_hypsicity')
            assert "NoMapping" in str(te.type)
        else:
            delta = node_descriptor_calculator(ce, 'ce_hypsicity')
            assert delta == d['expected']
