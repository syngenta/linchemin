import pytest

from linchemin.cgu.syngraph import MonopartiteReacSynGraph
from linchemin.cheminfo.constructors import ChemicalEquationConstructor
from linchemin.rem.step_descriptors import (WrongSmilesType,
                                            get_available_step_descriptors,
                                            step_descriptor_calculator)


def test_factory():
    sd = get_available_step_descriptors()
    assert "step_effectiveness" in sd
    route = MonopartiteReacSynGraph(
        [{"query_id": 0, "output_string": "CC(=O)O.CN.CN>O>CNC(C)=O"}]
    )
    # when a not existing descriptor is requested, an error is raised
    with pytest.raises(KeyError) as ke:
        step_descriptor_calculator("some_score", route, "CC(=O)O.CN.CN>O>CNC(C)=O")
    assert "KeyError" in str(ke.type)

    # when an object of the wrong type is passed as route, an error is raised
    route = {}
    with pytest.raises(TypeError) as ke:
        step_descriptor_calculator(
            "step_effectiveness", route, "CC(=O)O.CN.CN>O>CNC(C)=O"
        )
    assert "TypeError" in str(ke.type)

    # when a smiles not representing a reaction is given, an error is raised
    route = MonopartiteReacSynGraph()
    with pytest.raises(WrongSmilesType) as ke:
        step_descriptor_calculator("step_hypsicity", route, "CO")
    assert "WrongSmilesType" in str(ke.type)


def test_atom_effectiveness():
    route_smiles = [
        {
            "query_id": 0,
            "output_string": "O[C:11]([CH2:10][NH:9][c:8]1[c:2]([CH3:1])[cH:3][cH:4][cH:5][c:6]1[CH3:7])=[O:12]>>Cl[C:11]([CH2:10][NH:9][c:8]1[c:2]([CH3:1])[cH:3][cH:4][cH:5][c:6]1[CH3:7])=[O:12]",
        },
        {
            "query_id": 1,
            "output_string": "Cl[C:11]([CH2:10][NH:9][c:8]1[c:2]([CH3:1])[cH:3][cH:4][cH:5][c:6]1[CH3:7])=[O:12].[NH2:13][c:14]1[cH:15][cH:16][c:17](-[c:18]2[n:19][cH:20][o:21][n:22]2)[cH:23][cH:24]1>>[CH3:1][c:2]1[cH:3][cH:4][cH:5][c:6]([CH3:7])[c:8]1[NH:9][CH2:10][C:11](=[O:12])[NH:13][c:14]1[cH:15][cH:16][c:17](-[c:18]2[n:19][cH:20][o:21][n:22]2)[cH:23][cH:24]1",
        },
        {
            "query_id": 2,
            "output_string": "O[C:26](=[O:25])[CH:27]1[CH2:28][CH2:29][S:30](=[O:31])(=[O:32])[CH2:33][CH2:34]1.[CH3:1][c:2]1[cH:3][cH:4][cH:5][c:6]([CH3:7])[c:8]1[NH:9][CH2:10][C:11](=[O:12])[NH:13][c:14]1[cH:15][cH:16][c:17](-[c:18]2[n:19][cH:20][o:21][n:22]2)[cH:23][cH:24]1>>[CH3:1][c:2]1[cH:3][cH:4][cH:5][c:6]([CH3:7])[c:8]1[N:9]([CH2:10][C:11](=[O:12])[NH:13][c:14]1[cH:15][cH:16][c:17](-[c:18]2[n:19][cH:20][o:21][n:22]2)[cH:23][cH:24]1)[C:26](=[O:25])[CH:27]1[CH2:28][CH2:29][S:30](=[O:31])(=[O:32])[CH2:33][CH2:34]1",
        },
    ]

    syngraph = MonopartiteReacSynGraph(route_smiles)
    out = step_descriptor_calculator(
        "step_effectiveness", syngraph, route_smiles[0]["output_string"]
    )
    assert round(out.descriptor_value, 2) == 0.92
    assert out.additional_info["contributing_atoms"] == 12

    out = step_descriptor_calculator(
        "step_effectiveness", syngraph, route_smiles[1]["output_string"]
    )
    assert round(out.descriptor_value, 2) == 0.96
    assert out.additional_info["contributing_atoms"] == 24

    out = step_descriptor_calculator(
        "step_effectiveness", syngraph, route_smiles[2]["output_string"]
    )
    assert round(out.descriptor_value, 2) == 0.97
    assert out.additional_info["contributing_atoms"] == 34


def test_step_hypsicity():
    route_smiles = [
        {
            "query_id": 0,
            "output_string": "[OH:2][c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1>[Na+:1].[OH-]>[Na+:1].[O-:2][c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1",
        },
        {
            "query_id": 1,
            "output_string": "Cl[CH2:4][C:2](=[O:1])[O-:3].[O-:5][c:6]1[cH:7][cH:8][cH:9][cH:10][cH:11]1>[Na+].[Na+]>[O:1]=[C:2]([OH:3])[CH2:4][O:5][c:6]1[cH:7][cH:8][cH:9][cH:10][cH:11]1",
        },
        {
            "query_id": 2,
            "output_string": "Cl[Cl:10].Cl[Cl:13].[O:1]=[C:2]([OH:3])[CH2:4][O:5][c:6]1[cH:7][cH:8][cH:9][cH:11][cH:12]1>>[O:1]=[C:2]([OH:3])[CH2:4][O:5][c:6]1[cH:7][cH:8][c:9]([Cl:10])[cH:11][c:12]1[Cl:13]",
        },
    ]
    syngraph = MonopartiteReacSynGraph(route_smiles)
    out = step_descriptor_calculator(
        "step_hypsicity", syngraph, route_smiles[0]["output_string"]
    )
    assert out.descriptor_value == -4.0
    out = step_descriptor_calculator(
        "step_hypsicity", syngraph, route_smiles[1]["output_string"]
    )
    assert out.descriptor_value == -4.0
    out = step_descriptor_calculator(
        "step_hypsicity", syngraph, route_smiles[2]["output_string"]
    )
    assert out.descriptor_value == -2.0
