import linchemin.cheminfo.depiction as cid
import linchemin.cheminfo.functions as cif
from linchemin.cheminfo.disconnection import Disconnection, DisconnectionConstructor
from linchemin.IO import io as lio


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
            # fully balanced intramolecular michael addition ring forming, one new bond and one changend bond
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
            "smiles": "[cH:5]1[cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]2[c:3]([cH:4]1)[C:2](=[O:1])O.[N-:13]=[N+:14]=[N-:15]>C(Cl)Cl.C(=O)(C(=O)Cl)Cl>[cH:5]1[cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]2[c:3]([cH:4]1)[C:2](=[O:1])[N:13]=[N+:14]=[N-:15]",
            "expected": {},
        },
    ]

    dc = DisconnectionConstructor(identity_property_name="smiles")

    results = {
        item.get("name"): dc.build_from_reaction_string(
            reaction_string=item.get("smiles"), inp_fmt="smiles"
        )
        for item in test_set
    }

    # regioisomer products from the same reactants: disconnection is different (fragments might be the same)
    assert results.get("rnx_4") != results.get("rnx_5")

    # same product from two sets of equivalent reactants (at synthol level)
    assert results.get("rnx_1") == results.get("rnx_6")


def test_depiction():
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
            # fully balanced intramolecular michael addition ring forming, one new bond and one changend bond
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
            "smiles": "[cH:5]1[cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]2[c:3]([cH:4]1)[C:2](=[O:1])O.[N-:13]=[N+:14]=[N-:15]>C(Cl)Cl.C(=O)(C(=O)Cl)Cl>[cH:5]1[cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]2[c:3]([cH:4]1)[C:2](=[O:1])[N:13]=[N+:14]=[N-:15]",
            "expected": {},
        },
    ]

    dc = DisconnectionConstructor(identity_property_name="smiles")

    results = {
        item.get("name"): dc.build_from_reaction_string(
            reaction_string=item.get("smiles"), inp_fmt="smiles"
        )
        for item in test_set
    }

    for name, disconnection in results.items():
        # print(disconnection.to_dict())
        # rdrxn = cif.rdrxn_from_string(input_string=item.get('smiles'), inp_fmt='smiles')
        depiction_data = cid.draw_disconnection(
            reacting_atoms=disconnection.reacting_atoms,
            new_bonds=disconnection.new_bonds,
            modified_bonds=disconnection.modified_bonds,
            rdmol=disconnection.rdmol,
        )
        # lio.write_rdkit_depict(data=depiction_data, file_path=f"{name}_disconnection.png")

        # depiction_data = cid.draw_fragments(rdmol=disconnection.rdmol_fragmented)
        # lio.write_rdkit_depict(data=depiction_data, file_path=f"{name}_fragment.png")
        # depiction_data = cid.draw_reaction(rdrxn=rdrxn, )
        # lio.write_rdkit_depict(data=depiction_data, file_path=f"{item.get('name')}_reaction.png")
        # print(f"\n{item.get('name')} {disconnection.__dict__}")


def xtest():
    # smiles= '[CH3:6][CH:7]=[CH2:8].[CH3:1][C:2](=[CH2:3])[CH:4]=[CH2:5]>>[CH3:6][CH:7]1[CH2:8][CH2:5][CH:4]=[C:2]([CH2:3]1)[CH3:1]'
    smiles = "[cH:5]1[cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]2[c:3]([cH:4]1)[C:2](=[O:1])O.[N-:13]=[N+:14]=[N-:15]>C(Cl)Cl.C(=O)(C(=O)Cl)Cl>[cH:5]1[cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]2[c:3]([cH:4]1)[C:2](=[O:1])[N:13]=[N+:14]=[N-:15]"

    import linchemin.cheminfo.depiction as cid
    from linchemin.cheminfo.reaction import (
        ChemicalEquation,
        ChemicalEquationConstructor,
    )
    from linchemin.IO import io as lio

    chemical_equation_constructor = ChemicalEquationConstructor(
        identity_property_name="smiles"
    )
    chemical_equation = chemical_equation_constructor.build_from_reaction_string(
        reaction_string=smiles, inp_fmt="smiles"
    )

    ce_rdrxn = chemical_equation.build_rdrxn(use_atom_mapping=True)
    # print(cif.rdrxn_to_string(rdrxn=ce_rdrxn, out_fmt='smiles'))

    dc = DisconnectionConstructor(identity_property_name="smiles")
    disconnection = dc.build_from_reaction_string(
        reaction_string=smiles, inp_fmt="smiles"
    )
    # disconnection = dc.build_from_rdrxn(rdrxn=ce_rdrxn)

    depiction_data = cid.draw_disconnection(
        reacting_atoms=disconnection.reacting_atoms,
        new_bonds=disconnection.new_bonds,
        modified_bonds=disconnection.modified_bonds,
        rdmol=disconnection.rdmol,
    )
    # lio.write_rdkit_depict(data=depiction_data, file_path=f"x_disconnection.png")
