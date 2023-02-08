from linchemin.cheminfo.molecule import Molecule, MoleculeConstructor


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
    }

    # initialize the constructor to use smiles as identity property
    molecule_constructor = MoleculeConstructor(identity_property_name="smiles")

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

    # initialize the constructor to use inchi_key as identity property
    molecule_constructor = MoleculeConstructor(identity_property_name="inchi_key")
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
