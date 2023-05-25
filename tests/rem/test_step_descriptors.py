import pytest
from linchemin.cgu.syngraph import MonopartiteReacSynGraph
from linchemin.cheminfo.constructors import ChemicalEquationConstructor
from linchemin.rem.step_descriptors import step_descriptor_calculator, get_available_step_descriptors


def test_factory():
    sd = get_available_step_descriptors()
    assert 'step_effectiveness' in sd
    route = MonopartiteReacSynGraph()
    # when a not existing descriptor is requested, an error is raised
    with pytest.raises(KeyError) as ke:
        step_descriptor_calculator('some_score', route, 'step')
    assert "KeyError" in str(ke.type)
    # when an object of the wring type is passed as route, an error is raised
    route = {}
    with pytest.raises(TypeError) as ke:
        step_descriptor_calculator('step_effectiveness', route, 'step')
    assert "TypeError" in str(ke.type)


def test_atom_effectiveness(az_path):
    route_smiles = {
        0: 'O[C:11]([CH2:10][NH:9][c:8]1[c:2]([CH3:1])[cH:3][cH:4][cH:5][c:6]1[CH3:7])=[O:12]>>Cl[C:11]([CH2:10][NH:9][c:8]1[c:2]([CH3:1])[cH:3][cH:4][cH:5][c:6]1[CH3:7])=[O:12]',
        1: 'Cl[C:11]([CH2:10][NH:9][c:8]1[c:2]([CH3:1])[cH:3][cH:4][cH:5][c:6]1[CH3:7])=[O:12].[NH2:13][c:14]1[cH:15][cH:16][c:17](-[c:18]2[n:19][cH:20][o:21][n:22]2)[cH:23][cH:24]1>>[CH3:1][c:2]1[cH:3][cH:4][cH:5][c:6]([CH3:7])[c:8]1[NH:9][CH2:10][C:11](=[O:12])[NH:13][c:14]1[cH:15][cH:16][c:17](-[c:18]2[n:19][cH:20][o:21][n:22]2)[cH:23][cH:24]1',
        2: 'O[C:26](=[O:25])[CH:27]1[CH2:28][CH2:29][S:30](=[O:31])(=[O:32])[CH2:33][CH2:34]1.[CH3:1][c:2]1[cH:3][cH:4][cH:5][c:6]([CH3:7])[c:8]1[NH:9][CH2:10][C:11](=[O:12])[NH:13][c:14]1[cH:15][cH:16][c:17](-[c:18]2[n:19][cH:20][o:21][n:22]2)[cH:23][cH:24]1>>[CH3:1][c:2]1[cH:3][cH:4][cH:5][c:6]([CH3:7])[c:8]1[N:9]([CH2:10][C:11](=[O:12])[NH:13][c:14]1[cH:15][cH:16][c:17](-[c:18]2[n:19][cH:20][o:21][n:22]2)[cH:23][cH:24]1)[C:26](=[O:25])[CH:27]1[CH2:28][CH2:29][S:30](=[O:31])(=[O:32])[CH2:33][CH2:34]1'}
    chemical_equation_constructor = ChemicalEquationConstructor(molecular_identity_property_name='smiles',
                                                                chemical_equation_identity_name='r_r_p')
    route_ce = {0: chemical_equation_constructor.build_from_reaction_string(reaction_string=route_smiles[0],
                                                                            inp_fmt='smiles'),
                1: chemical_equation_constructor.build_from_reaction_string(reaction_string=route_smiles[1],
                                                                            inp_fmt='smiles'),
                2: chemical_equation_constructor.build_from_reaction_string(reaction_string=route_smiles[2],
                                                                            inp_fmt='smiles'),
                }
    syngraph = MonopartiteReacSynGraph()
    syngraph.add_node((route_ce[0], [route_ce[1]]))
    syngraph.add_node((route_ce[1], [route_ce[2]]))
    out = step_descriptor_calculator('step_effectiveness', syngraph, route_ce[0])
    assert round(out.descriptor_value, 2) == 0.35
    assert out.additional_info['contributing_atoms'] == 12
    out = step_descriptor_calculator('step_effectiveness', syngraph, route_ce[1])
    assert round(out.descriptor_value, 2) == 0.71
    assert out.additional_info['contributing_atoms'] == 24
    out = step_descriptor_calculator('step_effectiveness', syngraph, route_ce[2])
    assert round(out.descriptor_value, 2) == 1.0
    assert out.additional_info['contributing_atoms'] == 34


def test_step_hypsicity():
    route_smiles = {
        0: 'O[C:11]([CH2:10][NH:9][c:8]1[c:2]([CH3:1])[cH:3][cH:4][cH:5][c:6]1[CH3:7])=[O:12]>>Cl[C:11]([CH2:10][NH:9][c:8]1[c:2]([CH3:1])[cH:3][cH:4][cH:5][c:6]1[CH3:7])=[O:12]',
        1: 'Cl[C:11]([CH2:10][NH:9][c:8]1[c:2]([CH3:1])[cH:3][cH:4][cH:5][c:6]1[CH3:7])=[O:12].[NH2:13][c:14]1[cH:15][cH:16][c:17](-[c:18]2[n:19][cH:20][o:21][n:22]2)[cH:23][cH:24]1>>[CH3:1][c:2]1[cH:3][cH:4][cH:5][c:6]([CH3:7])[c:8]1[NH:9][CH2:10][C:11](=[O:12])[NH:13][c:14]1[cH:15][cH:16][c:17](-[c:18]2[n:19][cH:20][o:21][n:22]2)[cH:23][cH:24]1',
        2: 'O[C:26](=[O:25])[CH:27]1[CH2:28][CH2:29][S:30](=[O:31])(=[O:32])[CH2:33][CH2:34]1.[CH3:1][c:2]1[cH:3][cH:4][cH:5][c:6]([CH3:7])[c:8]1[NH:9][CH2:10][C:11](=[O:12])[NH:13][c:14]1[cH:15][cH:16][c:17](-[c:18]2[n:19][cH:20][o:21][n:22]2)[cH:23][cH:24]1>>[CH3:1][c:2]1[cH:3][cH:4][cH:5][c:6]([CH3:7])[c:8]1[N:9]([CH2:10][C:11](=[O:12])[NH:13][c:14]1[cH:15][cH:16][c:17](-[c:18]2[n:19][cH:20][o:21][n:22]2)[cH:23][cH:24]1)[C:26](=[O:25])[CH:27]1[CH2:28][CH2:29][S:30](=[O:31])(=[O:32])[CH2:33][CH2:34]1'}
    chemical_equation_constructor = ChemicalEquationConstructor(molecular_identity_property_name='smiles',
                                                                chemical_equation_identity_name='r_r_p')
    route_ce = {0: chemical_equation_constructor.build_from_reaction_string(reaction_string=route_smiles[0],
                                                                            inp_fmt='smiles'),
                1: chemical_equation_constructor.build_from_reaction_string(reaction_string=route_smiles[1],
                                                                            inp_fmt='smiles'),
                2: chemical_equation_constructor.build_from_reaction_string(reaction_string=route_smiles[2],
                                                                            inp_fmt='smiles'),
                }
    syngraph = MonopartiteReacSynGraph()
    syngraph.add_node((route_ce[0], [route_ce[1]]))
    syngraph.add_node((route_ce[1], [route_ce[2]]))
    out = step_descriptor_calculator('step_hypsicity', syngraph, route_ce[0])
    # print(out.descriptor_value)
    # print(out.additional_info)
    out = step_descriptor_calculator('step_hypsicity', syngraph, route_ce[1])
    # print(out.descriptor_value)
    out = step_descriptor_calculator('step_hypsicity', syngraph, route_ce[2])
    # print(out.descriptor_value)
