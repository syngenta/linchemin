from itertools import combinations
from linchemin.cheminfo.constructors import (
    MoleculeConstructor,
    ChemicalEquationConstructor,
    RatamConstructor,
    BadMapping,
    PatternConstructor,
    TemplateConstructor,
    DisconnectionConstructor,
    calculate_molecular_hash_values,
    UnavailableMolIdentifier,
)
from linchemin.cheminfo.atom_mapping import pipeline_atom_mapping
from linchemin.cheminfo.models import Template
from linchemin.utilities import create_hash
import linchemin.cheminfo.functions as cif
from linchemin.IO import io as lio
import linchemin.cheminfo.depiction as cid
import unittest
import pytest
import pprint


# Molecule tests
def test_molecular_constructor():
    smiles = "CC(C)=O"
    with pytest.raises(UnavailableMolIdentifier) as ke:
        MoleculeConstructor(molecular_identity_property_name="something")
    assert "UnavailableMolIdentifier" in str(ke.type)
    with unittest.TestCase().assertLogs(
        "linchemin.cheminfo.constructors", level="WARNING"
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


def test_molecular_hashing():
    examples = [
        {"name": "ra1", "smiles": "Cc1ccc2c(C(=O)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1"},
        {"name": "ra2", "smiles": "Cc1ccc2c(C(=O)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1"},
        {"name": "ra3", "smiles": "CCCCCn1cc(C(=O)c2cccc3ccccc23)c2ccccc21"},
        {"name": "ra4", "smiles": "CC1COCCN1CCn1cc(C(=O)c2cccc3ccccc23)c2ccccc21"},
        {"name": "ra5", "smiles": "Cc1ccc(C(=O)c2cn(CCN3CCOCC3)c3ccccc23)c2ccccc12"},
        {"name": "ra6", "smiles": "Cc1c(CCN2CCOCC2)c2ccccc2n1C(=O)c1cccc2ccccc12"},
        {"name": "ra7", "smiles": "CN1CCN(C)C(Cn2cc(C(=O)c3cccc4ccccc34)c3ccccc32)C1"},
        {"name": "ma1", "smiles": "CCC1CC(C(=O)c2cccc(C)c2)CCC1=O"},
        {"name": "ma2", "smiles": "CCC1CC(C(=O)c2ccccc2)CCC1=O"},
        {"name": "ma3", "smiles": "C=C(C)C(c1ccccc1)S(=O)CC(N)=O"},
        {"name": "ma4", "smiles": "Cc1cccc(C(C2CCC(N)CC2)C(F)(F)F)c1"},
        {"name": "ma5", "smiles": "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21"},
        {"name": "ma6", "smiles": "CCCOC(c1ccc(Cl)cc1)C1CCCCC1"},
        {"name": "ta1", "smiles": "OC1=NCCC1"},
        {"name": "ta2", "smiles": "O=C1CCCN1"},
        {"name": "sa1", "smiles": "CC[C@@H](C)[C@H](O)Cl"},
        {"name": "sa2", "smiles": "CC[C@@H](C)[C@@H](O)Cl"},
        {"name": "sa3", "smiles": "CC[C@@H](C)C(O)Cl"},
        {"name": "sa4", "smiles": "CC[C@H](C)[C@H](O)Cl"},
        {"name": "sa5", "smiles": "CC[C@H](C)[C@@H](O)Cl"},
        {"name": "sa6", "smiles": "CC[C@H](C)C(O)Cl"},
        {"name": "sa7", "smiles": "CCC(C)[C@H](O)Cl"},
        {"name": "sa8", "smiles": "CCC(C)[C@@H](O)Cl"},
        {"name": "sa9", "smiles": "CCC(C)C(O)Cl"},
        {"name": "tb1", "smiles": r"C/N=C(\C)C1C(=O)CCC(C)C1=O"},
        {"name": "tb2", "smiles": r"C/N=C(\C)C1=C(O)CCC(C)C1=O"},
        {"name": "tb3", "smiles": r"C/N=C(\C)C1=C(O)C(C)CCC1=O"},
        {"name": "tc1", "smiles": "CC(=O)C1=C(O)C(C)CCC1=O"},
        {"name": "tc2", "smiles": "CC(=O)C1C(=O)CCC(C)C1=O"},
        {"name": "tc3", "smiles": "CC(=O)C1=C(O)CCC(C)C1=O"},
        {"name": "tc4", "smiles": r"C/C(O)=C1\C(=O)CCC(C)C1=O"},
    ]

    reference = {
        "ma1": {
            "AnonymousGraph": "***1**(*(*)*2****(*)*2)***1*",
            "ArthorSubstructureOrder": "001200130100100002000070000000",
            "AtomBondCounts": "18,19",
            "CanonicalSmiles": "CCC1CC(C(=O)c2cccc(C)c2)CCC1=O",
            "DegreeVector": "0,6,8,4",
            "ElementGraph": "CCC1CC(C(O)C2CCCC(C)C2)CCC1O",
            "ExtendedMurcko": "*c1cccc(C(=*)C2CCC(=*)C(*)C2)c1",
            "HetAtomProtomer": "CCC1CC([C]([O])[C]2[CH][CH][CH][C](C)[CH]2)CC[C]1[O]_0",
            "HetAtomTautomer": "CCC1CC([C]([O])[C]2[CH][CH][CH][C](C)[CH]2)CC[C]1[O]_0_0",
            "Mesomer": "CCC1CC([C]([O])[C]2[CH][CH][CH][C](C)[CH]2)CC[C]1[O]_0",
            "MolFormula": "C16H20O2",
            "MurckoScaffold": "c1ccc(CC2CCCCC2)cc1",
            "NetCharge": "0",
            "RedoxPair": "CCC1CC([C]([O])[C]2[CH][CH][CH][C](C)[CH]2)CC[C]1[O]",
            "Regioisomer": "*C.*C(*)=O.*CC.O=C1CCCCC1.c1ccccc1",
            "SmallWorldIndexBR": "B19R2",
            "SmallWorldIndexBRL": "B19R2L8",
            "cx_smiles": "CCC1CC(C(=O)c2cccc(C)c2)CCC1=O",
            "inchi": "InChI=1S/C16H20O2/c1-3-12-10-14(7-8-15(12)17)16(18)13-6-4-5-11(2)9-13/h4-6,9,12,14H,3,7-8,10H2,1-2H3",
            "inchi_KET_15T": "InChI=1/C16H20O2/c1-3-12-10-14(7-8-15(12)17)16(18)13-6-4-5-11(2)9-13/h4-6,9H,3,7,10H2,1-2H3,(H,14,18)(H3,8,12,17)",
            "inchi_key": "MQEBHHFSYXQNPF-UHFFFAOYSA-N",
            "inchikey_KET_15T": "WFPNUSIRQPGLBS-UHFFFAOYNA-N",
            "noiso_smiles": "CCC1CC(C(=O)c2cccc(C)c2)CCC1=O",
            "smiles": "CCC1CC(C(=O)c2cccc(C)c2)CCC1=O",
        },
        "ma2": {
            "AnonymousGraph": "***1**(*(*)*2*****2)***1*",
            "ArthorSubstructureOrder": "0011001201000f000200006a000000",
            "AtomBondCounts": "17,18",
            "CanonicalSmiles": "CCC1CC(C(=O)c2ccccc2)CCC1=O",
            "DegreeVector": "0,5,9,3",
            "ElementGraph": "CCC1CC(C(O)C2CCCCC2)CCC1O",
            "ExtendedMurcko": "*C1CC(C(=*)c2ccccc2)CCC1=*",
            "HetAtomProtomer": "CCC1CC([C]([O])[C]2[CH][CH][CH][CH][CH]2)CC[C]1[O]_0",
            "HetAtomTautomer": "CCC1CC([C]([O])[C]2[CH][CH][CH][CH][CH]2)CC[C]1[O]_0_0",
            "Mesomer": "CCC1CC([C]([O])[C]2[CH][CH][CH][CH][CH]2)CC[C]1[O]_0",
            "MolFormula": "C15H18O2",
            "MurckoScaffold": "c1ccc(CC2CCCCC2)cc1",
            "NetCharge": "0",
            "RedoxPair": "CCC1CC([C]([O])[C]2[CH][CH][CH][CH][CH]2)CC[C]1[O]",
            "Regioisomer": "*C(*)=O.*CC.O=C1CCCCC1.c1ccccc1",
            "SmallWorldIndexBR": "B18R2",
            "SmallWorldIndexBRL": "B18R2L9",
            "cx_smiles": "CCC1CC(C(=O)c2ccccc2)CCC1=O",
            "inchi": "InChI=1S/C15H18O2/c1-2-11-10-13(8-9-14(11)16)15(17)12-6-4-3-5-7-12/h3-7,11,13H,2,8-10H2,1H3",
            "inchi_KET_15T": "InChI=1/C15H18O2/c1-2-11-10-13(8-9-14(11)16)15(17)12-6-4-3-5-7-12/h3-7H,2,8,10H2,1H3,(H,13,17)(H3,9,11,16)",
            "inchi_key": "NGWXTRJFWAYHFL-UHFFFAOYSA-N",
            "inchikey_KET_15T": "NEMNUJJNBNORAW-UHFFFAOYNA-N",
            "noiso_smiles": "CCC1CC(C(=O)c2ccccc2)CCC1=O",
            "smiles": "CCC1CC(C(=O)c2ccccc2)CCC1=O",
        },
        "ma3": {
            "AnonymousGraph": "**(*)**(*)*(*(*)*)*1*****1",
            "ArthorSubstructureOrder": "0010001001000c000400006f000000",
            "AtomBondCounts": "16,16",
            "CanonicalSmiles": "C=C(C)C(c1ccccc1)S(=O)CC(N)=O",
            "DegreeVector": "0,5,6,5",
            "ElementGraph": "CC(C)C(C1CCCCC1)[S](O)CC(N)O",
            "ExtendedMurcko": "*c1ccccc1",
            "HetAtomProtomer": "C=C(C)C([C]1[CH][CH][CH][CH][CH]1)[S]([O])C[C]([N])[O]_2",
            "HetAtomTautomer": "C=C(C)C([C]1[CH][CH][CH][CH][CH]1)[S]([O])C[C]([N])[O]_2_0",
            "Mesomer": "[CH2][C](C)C([C]1[CH][CH][CH][CH][CH]1)[S]([O])C[C](N)[O]_0",
            "MolFormula": "C12H15NO2S",
            "MurckoScaffold": "c1ccccc1",
            "NetCharge": "0",
            "RedoxPair": "[CH2][C](C)C([C]1[CH][CH][CH][CH][CH]1)[S]([O])C[C](N)[O]",
            "Regioisomer": "*CC(=C)C.*S(*)=O.CC(N)=O.c1ccccc1",
            "SmallWorldIndexBR": "B16R1",
            "SmallWorldIndexBRL": "B16R1L6",
            "cx_smiles": "C=C(C)C(c1ccccc1)S(=O)CC(N)=O",
            "inchi": "InChI=1S/C12H15NO2S/c1-9(2)12(16(15)8-11(13)14)10-6-4-3-5-7-10/h3-7,12H,1,8H2,2H3,(H2,13,14)",
            "inchi_KET_15T": "InChI=1/C12H15NO2S/c1-9(2)12(16(15)8-11(13)14)10-6-4-3-5-7-10/h3-7,12H,1H2,2H3,(H4,8,13,14)",
            "inchi_key": "ZFCHMUVIJDAZSM-UHFFFAOYSA-N",
            "inchikey_KET_15T": "DIAQTYFBHNUKSP-UHFFFAOYNA-N",
            "noiso_smiles": "C=C(C)C(c1ccccc1)S(=O)CC(N)=O",
            "smiles": "C=C(C)C(c1ccccc1)S(=O)CC(N)=O",
        },
        "ma4": {
            "AnonymousGraph": "**1***(*(*2****(*)*2)*(*)(*)*)**1",
            "ArthorSubstructureOrder": "0013001401000f000400007c000000",
            "AtomBondCounts": "19,20",
            "CanonicalSmiles": "Cc1cccc(C(C2CCC(N)CC2)C(F)(F)F)c1",
            "DegreeVector": "1,5,8,5",
            "ElementGraph": "CC1CCCC(C(C2CCC(N)CC2)C(F)(F)F)C1",
            "ExtendedMurcko": "*c1cccc(C(*)C2CCC(*)CC2)c1",
            "HetAtomProtomer": "C[C]1[CH][CH][CH][C](C(C2CCC([N])CC2)C(F)(F)F)[CH]1_2",
            "HetAtomTautomer": "C[C]1[CH][CH][CH][C](C(C2CCC([N])CC2)C(F)(F)F)[CH]1_2_0",
            "Mesomer": "C[C]1[CH][CH][CH][C](C(C2CCC(N)CC2)C(F)(F)F)[CH]1_0",
            "MolFormula": "C15H20F3N",
            "MurckoScaffold": "c1ccc(CC2CCCCC2)cc1",
            "NetCharge": "0",
            "RedoxPair": "C[C]1[CH][CH][CH][C](C(C2CCC(N)CC2)C(F)(F)F)[CH]1",
            "Regioisomer": "*C.*C(*)C.*F.*F.*F.*N.C1CCCCC1.c1ccccc1",
            "SmallWorldIndexBR": "B20R2",
            "SmallWorldIndexBRL": "B20R2L8",
            "cx_smiles": "Cc1cccc(C(C2CCC(N)CC2)C(F)(F)F)c1",
            "inchi": "InChI=1S/C15H20F3N/c1-10-3-2-4-12(9-10)14(15(16,17)18)11-5-7-13(19)8-6-11/h2-4,9,11,13-14H,5-8,19H2,1H3",
            "inchi_KET_15T": "InChI=1/C15H20F3N/c1-10-3-2-4-12(9-10)14(15(16,17)18)11-5-7-13(19)8-6-11/h2-4,9,11,13-14H,5-8,19H2,1H3",
            "inchi_key": "JHNUJAVMAYMRLC-UHFFFAOYSA-N",
            "inchikey_KET_15T": "JHNUJAVMAYMRLC-UHFFFAOYNA-N",
            "noiso_smiles": "Cc1cccc(C(C2CCC(N)CC2)C(F)(F)F)c1",
            "smiles": "Cc1cccc(C(C2CCC(N)CC2)C(F)(F)F)c1",
        },
        "ma5": {
            "AnonymousGraph": "***1***(*2***(*)*(*)*2)*2*****12",
            "ArthorSubstructureOrder": "00140016010011000300008f000000",
            "AtomBondCounts": "20,22",
            "CanonicalSmiles": "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21",
            "DegreeVector": "0,7,10,3",
            "ElementGraph": "CNC1CCC(C2CCC(Cl)C(Cl)C2)C2CCCCC12",
            "ExtendedMurcko": "*c1ccc(C2CCC(*)c3ccccc32)cc1*",
            "HetAtomProtomer": "C[N]C1CCC([C]2[CH][CH][C](Cl)[C](Cl)[CH]2)[C]2[CH][CH][CH][CH][C]21_1",
            "HetAtomTautomer": "C[N]C1CCC([C]2[CH][CH][C](Cl)[C](Cl)[CH]2)[C]2[CH][CH][CH][CH][C]21_1_0",
            "Mesomer": "CNC1CCC([C]2[CH][CH][C](Cl)[C](Cl)[CH]2)[C]2[CH][CH][CH][CH][C]21_0",
            "MolFormula": "C17H17Cl2N",
            "MurckoScaffold": "c1ccc(C2CCCc3ccccc32)cc1",
            "NetCharge": "0",
            "RedoxPair": "CNC1CCC([C]2[CH][CH][C](Cl)[C](Cl)[CH]2)[C]2[CH][CH][CH][CH][C]21",
            "Regioisomer": "*Cl.*Cl.*N*.C.c1ccc2c(c1)CCCC2.c1ccccc1",
            "SmallWorldIndexBR": "B22R3",
            "SmallWorldIndexBRL": "B22R3L10",
            "cx_smiles": "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21",
            "inchi": "InChI=1S/C17H17Cl2N/c1-20-17-9-7-12(13-4-2-3-5-14(13)17)11-6-8-15(18)16(19)10-11/h2-6,8,10,12,17,20H,7,9H2,1H3",
            "inchi_KET_15T": "InChI=1/C17H17Cl2N/c1-20-17-9-7-12(13-4-2-3-5-14(13)17)11-6-8-15(18)16(19)10-11/h2-6,8,10,12,17,20H,7,9H2,1H3",
            "inchi_key": "VGKDLMBJGBXTGI-UHFFFAOYSA-N",
            "inchikey_KET_15T": "VGKDLMBJGBXTGI-UHFFFAOYNA-N",
            "noiso_smiles": "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21",
            "smiles": "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21",
        },
        "ma6": {
            "AnonymousGraph": "*****(*1*****1)*1***(*)**1",
            "ArthorSubstructureOrder": "001200130100100002000079000000",
            "AtomBondCounts": "18,19",
            "CanonicalSmiles": "CCCOC(c1ccc(Cl)cc1)C1CCCCC1",
            "DegreeVector": "0,4,12,2",
            "ElementGraph": "CCCOC(C1CCCCC1)C1CCC(Cl)CC1",
            "ExtendedMurcko": "*c1ccc(C(*)C2CCCCC2)cc1",
            "HetAtomProtomer": "CCCOC([C]1[CH][CH][C](Cl)[CH][CH]1)C1CCCCC1_0",
            "HetAtomTautomer": "CCCOC([C]1[CH][CH][C](Cl)[CH][CH]1)C1CCCCC1_0_0",
            "Mesomer": "CCCOC([C]1[CH][CH][C](Cl)[CH][CH]1)C1CCCCC1_0",
            "MolFormula": "C16H23ClO",
            "MurckoScaffold": "c1ccc(CC2CCCCC2)cc1",
            "NetCharge": "0",
            "RedoxPair": "CCCOC([C]1[CH][CH][C](Cl)[CH][CH]1)C1CCCCC1",
            "Regioisomer": "*C*.*Cl.*O*.C1CCCCC1.CCC.c1ccccc1",
            "SmallWorldIndexBR": "B19R2",
            "SmallWorldIndexBRL": "B19R2L12",
            "cx_smiles": "CCCOC(c1ccc(Cl)cc1)C1CCCCC1",
            "inchi": "InChI=1S/C16H23ClO/c1-2-12-18-16(13-6-4-3-5-7-13)14-8-10-15(17)11-9-14/h8-11,13,16H,2-7,12H2,1H3",
            "inchi_KET_15T": "InChI=1/C16H23ClO/c1-2-12-18-16(13-6-4-3-5-7-13)14-8-10-15(17)11-9-14/h8-11,13,16H,2-7,12H2,1H3",
            "inchi_key": "IJTNMUAWUDRFON-UHFFFAOYSA-N",
            "inchikey_KET_15T": "IJTNMUAWUDRFON-UHFFFAOYNA-N",
            "noiso_smiles": "CCCOC(c1ccc(Cl)cc1)C1CCCCC1",
            "smiles": "CCCOC(c1ccc(Cl)cc1)C1CCCCC1",
        },
        "ra1": {
            "AnonymousGraph": "**(*1****2*****12)*1**(***2*****2)*2**(*)***12",
            "ArthorSubstructureOrder": "001e002201001a00040000ba000000",
            "AtomBondCounts": "30,34",
            "CanonicalSmiles": "Cc1ccc2c(C(=O)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1",
            "DegreeVector": "0,10,18,2",
            "ElementGraph": "CC1CCC2C(C(O)C3CCCC4CCCCC34)CN(CCN3CCOCC3)C2C1",
            "ExtendedMurcko": "*c1ccc2c(C(=*)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1",
            "HetAtomProtomer": "C[C]1[CH][CH][C]2[C]([C]([O])[C]3[CH][CH][CH][C]4[CH][CH][CH][CH][C]43)[CH]N(CCN3CCOCC3)[C]2[CH]1_0",
            "HetAtomTautomer": "C[C]1[CH][CH][C]2[C]([C]([O])[C]3[CH][CH][CH][C]4[CH][CH][CH][CH][C]43)[CH]N(CCN3CCOCC3)[C]2[CH]1_0_0",
            "Mesomer": "C[C]1[CH][CH][C]2[C]([C]([O])[C]3[CH][CH][CH][C]4[CH][CH][CH][CH][C]34)[CH]N(CCN3CCOCC3)[C]2[CH]1_0",
            "MolFormula": "C26H26N2O2",
            "MurckoScaffold": "c1ccc2c(Cc3cn(CCN4CCOCC4)c4ccccc34)cccc2c1",
            "NetCharge": "0",
            "RedoxPair": "C[C]1[CH][CH][C]2[C]([C]([O])[C]3[CH][CH][CH][C]4[CH][CH][CH][CH][C]34)[CH]N(CCN3CCOCC3)[C]2[CH]1",
            "Regioisomer": "*C.*C(*)=O.*CC*.C1COCCN1.c1ccc2[nH]ccc2c1.c1ccc2ccccc2c1",
            "SmallWorldIndexBR": "B34R5",
            "SmallWorldIndexBRL": "B34R5L18",
            "cx_smiles": "Cc1ccc2c(C(=O)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1",
            "inchi": "InChI=1S/C26H26N2O2/c1-19-9-10-22-24(26(29)23-8-4-6-20-5-2-3-7-21(20)23)18-28(25(22)17-19)12-11-27-13-15-30-16-14-27/h2-10,17-18H,11-16H2,1H3",
            "inchi_KET_15T": "InChI=1/C26H26N2O2/c1-19-9-10-22-24(26(29)23-8-4-6-20-5-2-3-7-21(20)23)18-28(25(22)17-19)12-11-27-13-15-30-16-14-27/h2-10,17-18H,11-16H2,1H3",
            "inchi_key": "DDVFEKLZEPNGMS-UHFFFAOYSA-N",
            "inchikey_KET_15T": "DDVFEKLZEPNGMS-UHFFFAOYNA-N",
            "noiso_smiles": "Cc1ccc2c(C(=O)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1",
            "smiles": "Cc1ccc2c(C(=O)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1",
        },
        "ra2": {
            "AnonymousGraph": "**(*1****2*****12)*1**(***2*****2)*2**(*)***12",
            "ArthorSubstructureOrder": "001e002201001a00040000ba000000",
            "AtomBondCounts": "30,34",
            "CanonicalSmiles": "Cc1ccc2c(C(=O)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1",
            "DegreeVector": "0,10,18,2",
            "ElementGraph": "CC1CCC2C(C(O)C3CCCC4CCCCC34)CN(CCN3CCOCC3)C2C1",
            "ExtendedMurcko": "*c1ccc2c(C(=*)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1",
            "HetAtomProtomer": "C[C]1[CH][CH][C]2[C]([C]([O])[C]3[CH][CH][CH][C]4[CH][CH][CH][CH][C]43)[CH]N(CCN3CCOCC3)[C]2[CH]1_0",
            "HetAtomTautomer": "C[C]1[CH][CH][C]2[C]([C]([O])[C]3[CH][CH][CH][C]4[CH][CH][CH][CH][C]43)[CH]N(CCN3CCOCC3)[C]2[CH]1_0_0",
            "Mesomer": "C[C]1[CH][CH][C]2[C]([C]([O])[C]3[CH][CH][CH][C]4[CH][CH][CH][CH][C]34)[CH]N(CCN3CCOCC3)[C]2[CH]1_0",
            "MolFormula": "C26H26N2O2",
            "MurckoScaffold": "c1ccc2c(Cc3cn(CCN4CCOCC4)c4ccccc34)cccc2c1",
            "NetCharge": "0",
            "RedoxPair": "C[C]1[CH][CH][C]2[C]([C]([O])[C]3[CH][CH][CH][C]4[CH][CH][CH][CH][C]34)[CH]N(CCN3CCOCC3)[C]2[CH]1",
            "Regioisomer": "*C.*C(*)=O.*CC*.C1COCCN1.c1ccc2[nH]ccc2c1.c1ccc2ccccc2c1",
            "SmallWorldIndexBR": "B34R5",
            "SmallWorldIndexBRL": "B34R5L18",
            "cx_smiles": "Cc1ccc2c(C(=O)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1",
            "inchi": "InChI=1S/C26H26N2O2/c1-19-9-10-22-24(26(29)23-8-4-6-20-5-2-3-7-21(20)23)18-28(25(22)17-19)12-11-27-13-15-30-16-14-27/h2-10,17-18H,11-16H2,1H3",
            "inchi_KET_15T": "InChI=1/C26H26N2O2/c1-19-9-10-22-24(26(29)23-8-4-6-20-5-2-3-7-21(20)23)18-28(25(22)17-19)12-11-27-13-15-30-16-14-27/h2-10,17-18H,11-16H2,1H3",
            "inchi_key": "DDVFEKLZEPNGMS-UHFFFAOYSA-N",
            "inchikey_KET_15T": "DDVFEKLZEPNGMS-UHFFFAOYNA-N",
            "noiso_smiles": "Cc1ccc2c(C(=O)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1",
            "smiles": "Cc1ccc2c(C(=O)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1",
        },
        "ra3": {
            "AnonymousGraph": "******1**(*(*)*2****3*****23)*2*****12",
            "ArthorSubstructureOrder": "001a001d010018000200009f000000",
            "AtomBondCounts": "26,29",
            "CanonicalSmiles": "CCCCCn1cc(C(=O)c2cccc3ccccc23)c2ccccc21",
            "DegreeVector": "0,8,16,2",
            "ElementGraph": "CCCCCN1CC(C(O)C2CCCC3CCCCC23)C2CCCCC21",
            "ExtendedMurcko": "*n1cc(C(=*)c2cccc3ccccc23)c2ccccc21",
            "HetAtomProtomer": "CCCCCN1[CH][C]([C]([O])[C]2[CH][CH][CH][C]3[CH][CH][CH][CH][C]32)[C]2[CH][CH][CH][CH][C]21_0",
            "HetAtomTautomer": "CCCCCN1[CH][C]([C]([O])[C]2[CH][CH][CH][C]3[CH][CH][CH][CH][C]32)[C]2[CH][CH][CH][CH][C]21_0_0",
            "Mesomer": "CCCCCN1[CH][C]([C]([O])[C]2[CH][CH][CH][C]3[CH][CH][CH][CH][C]23)[C]2[CH][CH][CH][CH][C]21_0",
            "MolFormula": "C24H23NO",
            "MurckoScaffold": "c1ccc2c(Cc3c[nH]c4ccccc34)cccc2c1",
            "NetCharge": "0",
            "RedoxPair": "CCCCCN1[CH][C]([C]([O])[C]2[CH][CH][CH][C]3[CH][CH][CH][CH][C]23)[C]2[CH][CH][CH][CH][C]21",
            "Regioisomer": "*C(*)=O.*CCCCC.c1ccc2[nH]ccc2c1.c1ccc2ccccc2c1",
            "SmallWorldIndexBR": "B29R4",
            "SmallWorldIndexBRL": "B29R4L16",
            "cx_smiles": "CCCCCn1cc(C(=O)c2cccc3ccccc23)c2ccccc21",
            "inchi": "InChI=1S/C24H23NO/c1-2-3-8-16-25-17-22(20-13-6-7-15-23(20)25)24(26)21-14-9-11-18-10-4-5-12-19(18)21/h4-7,9-15,17H,2-3,8,16H2,1H3",
            "inchi_KET_15T": "InChI=1/C24H23NO/c1-2-3-8-16-25-17-22(20-13-6-7-15-23(20)25)24(26)21-14-9-11-18-10-4-5-12-19(18)21/h4-7,9-15,17H,2-3,8,16H2,1H3",
            "inchi_key": "JDNLPKCAXICMBW-UHFFFAOYSA-N",
            "inchikey_KET_15T": "JDNLPKCAXICMBW-UHFFFAOYNA-N",
            "noiso_smiles": "CCCCCn1cc(C(=O)c2cccc3ccccc23)c2ccccc21",
            "smiles": "CCCCCn1cc(C(=O)c2cccc3ccccc23)c2ccccc21",
        },
        "ra4": {
            "AnonymousGraph": "**1*****1***1**(*(*)*2****3*****23)*2*****12",
            "ArthorSubstructureOrder": "001e002201001a00040000ba000000",
            "AtomBondCounts": "30,34",
            "CanonicalSmiles": "CC1COCCN1CCn1cc(C(=O)c2cccc3ccccc23)c2ccccc21",
            "DegreeVector": "0,10,18,2",
            "ElementGraph": "CC1COCCN1CCN1CC(C(O)C2CCCC3CCCCC23)C2CCCCC21",
            "ExtendedMurcko": "*C1COCCN1CCn1cc(C(=*)c2cccc3ccccc23)c2ccccc21",
            "HetAtomProtomer": "CC1COCCN1CCN1[CH][C]([C]([O])[C]2[CH][CH][CH][C]3[CH][CH][CH][CH][C]32)[C]2[CH][CH][CH][CH][C]21_0",
            "HetAtomTautomer": "CC1COCCN1CCN1[CH][C]([C]([O])[C]2[CH][CH][CH][C]3[CH][CH][CH][CH][C]32)[C]2[CH][CH][CH][CH][C]21_0_0",
            "Mesomer": "CC1COCCN1CCN1[CH][C]([C]([O])[C]2[CH][CH][CH][C]3[CH][CH][CH][CH][C]23)[C]2[CH][CH][CH][CH][C]21_0",
            "MolFormula": "C26H26N2O2",
            "MurckoScaffold": "c1ccc2c(Cc3cn(CCN4CCOCC4)c4ccccc34)cccc2c1",
            "NetCharge": "0",
            "RedoxPair": "CC1COCCN1CCN1[CH][C]([C]([O])[C]2[CH][CH][CH][C]3[CH][CH][CH][CH][C]23)[C]2[CH][CH][CH][CH][C]21",
            "Regioisomer": "*C.*C(*)=O.*CC*.C1COCCN1.c1ccc2[nH]ccc2c1.c1ccc2ccccc2c1",
            "SmallWorldIndexBR": "B34R5",
            "SmallWorldIndexBRL": "B34R5L18",
            "cx_smiles": "CC1COCCN1CCn1cc(C(=O)c2cccc3ccccc23)c2ccccc21",
            "inchi": "InChI=1S/C26H26N2O2/c1-19-18-30-16-15-27(19)13-14-28-17-24(22-10-4-5-12-25(22)28)26(29)23-11-6-8-20-7-2-3-9-21(20)23/h2-12,17,19H,13-16,18H2,1H3",
            "inchi_KET_15T": "InChI=1/C26H26N2O2/c1-19-18-30-16-15-27(19)13-14-28-17-24(22-10-4-5-12-25(22)28)26(29)23-11-6-8-20-7-2-3-9-21(20)23/h2-12,17,19H,13-16,18H2,1H3",
            "inchi_key": "AZPCJOZKMQFKLE-UHFFFAOYSA-N",
            "inchikey_KET_15T": "AZPCJOZKMQFKLE-UHFFFAOYNA-N",
            "noiso_smiles": "CC1COCCN1CCn1cc(C(=O)c2cccc3ccccc23)c2ccccc21",
            "smiles": "CC1COCCN1CCn1cc(C(=O)c2cccc3ccccc23)c2ccccc21",
        },
        "ra5": {
            "AnonymousGraph": "**(*1***(*)*2*****12)*1**(***2*****2)*2*****12",
            "ArthorSubstructureOrder": "001e002201001a00040000ba000000",
            "AtomBondCounts": "30,34",
            "CanonicalSmiles": "Cc1ccc(C(=O)c2cn(CCN3CCOCC3)c3ccccc23)c2ccccc12",
            "DegreeVector": "0,10,18,2",
            "ElementGraph": "CC1CCC(C(O)C2CN(CCN3CCOCC3)C3CCCCC23)C2CCCCC12",
            "ExtendedMurcko": "*c1ccc(C(=*)c2cn(CCN3CCOCC3)c3ccccc23)c2ccccc12",
            "HetAtomProtomer": "C[C]1[CH][CH][C]([C]([O])[C]2[CH]N(CCN3CCOCC3)[C]3[CH][CH][CH][CH][C]23)[C]2[CH][CH][CH][CH][C]12_0",
            "HetAtomTautomer": "C[C]1[CH][CH][C]([C]([O])[C]2[CH]N(CCN3CCOCC3)[C]3[CH][CH][CH][CH][C]23)[C]2[CH][CH][CH][CH][C]12_0_0",
            "Mesomer": "C[C]1[CH][CH][C]([C]([O])[C]2[CH]N(CCN3CCOCC3)[C]3[CH][CH][CH][CH][C]23)[C]2[CH][CH][CH][CH][C]12_0",
            "MolFormula": "C26H26N2O2",
            "MurckoScaffold": "c1ccc2c(Cc3cn(CCN4CCOCC4)c4ccccc34)cccc2c1",
            "NetCharge": "0",
            "RedoxPair": "C[C]1[CH][CH][C]([C]([O])[C]2[CH]N(CCN3CCOCC3)[C]3[CH][CH][CH][CH][C]23)[C]2[CH][CH][CH][CH][C]12",
            "Regioisomer": "*C.*C(*)=O.*CC*.C1COCCN1.c1ccc2[nH]ccc2c1.c1ccc2ccccc2c1",
            "SmallWorldIndexBR": "B34R5",
            "SmallWorldIndexBRL": "B34R5L18",
            "cx_smiles": "Cc1ccc(C(=O)c2cn(CCN3CCOCC3)c3ccccc23)c2ccccc12",
            "inchi": "InChI=1S/C26H26N2O2/c1-19-10-11-23(21-7-3-2-6-20(19)21)26(29)24-18-28(25-9-5-4-8-22(24)25)13-12-27-14-16-30-17-15-27/h2-11,18H,12-17H2,1H3",
            "inchi_KET_15T": "InChI=1/C26H26N2O2/c1-19-10-11-23(21-7-3-2-6-20(19)21)26(29)24-18-28(25-9-5-4-8-22(24)25)13-12-27-14-16-30-17-15-27/h2-11,18H,12-17H2,1H3",
            "inchi_key": "ICKWPPYMDARCKJ-UHFFFAOYSA-N",
            "inchikey_KET_15T": "ICKWPPYMDARCKJ-UHFFFAOYNA-N",
            "noiso_smiles": "Cc1ccc(C(=O)c2cn(CCN3CCOCC3)c3ccccc23)c2ccccc12",
            "smiles": "Cc1ccc(C(=O)c2cn(CCN3CCOCC3)c3ccccc23)c2ccccc12",
        },
        "ra6": {
            "AnonymousGraph": "**(*1****2*****12)*1*(*)*(***2*****2)*2*****12",
            "ArthorSubstructureOrder": "001e002201001a00040000ba000000",
            "AtomBondCounts": "30,34",
            "CanonicalSmiles": "Cc1c(CCN2CCOCC2)c2ccccc2n1C(=O)c1cccc2ccccc12",
            "DegreeVector": "0,10,18,2",
            "ElementGraph": "CC1C(CCN2CCOCC2)C2CCCCC2N1C(O)C1CCCC2CCCCC12",
            "ExtendedMurcko": "*c1c(CCN2CCOCC2)c2ccccc2n1C(=*)c1cccc2ccccc12",
            "HetAtomProtomer": "C[C]1[C](CCN2CCOCC2)[C]2[CH][CH][CH][CH][C]2N1[C]([O])[C]1[CH][CH][CH][C]2[CH][CH][CH][CH][C]21_0",
            "HetAtomTautomer": "C[C]1[C](CCN2CCOCC2)[C]2[CH][CH][CH][CH][C]2N1[C]([O])[C]1[CH][CH][CH][C]2[CH][CH][CH][CH][C]21_0_0",
            "Mesomer": "C[C]1[C](CCN2CCOCC2)[C]2[CH][CH][CH][CH][C]2N1[C]([O])[C]1[CH][CH][CH][C]2[CH][CH][CH][CH][C]12_0",
            "MolFormula": "C26H26N2O2",
            "MurckoScaffold": "c1ccc2c(Cn3cc(CCN4CCOCC4)c4ccccc43)cccc2c1",
            "NetCharge": "0",
            "RedoxPair": "C[C]1[C](CCN2CCOCC2)[C]2[CH][CH][CH][CH][C]2N1[C]([O])[C]1[CH][CH][CH][C]2[CH][CH][CH][CH][C]12",
            "Regioisomer": "*C.*C(*)=O.*CC*.C1COCCN1.c1ccc2[nH]ccc2c1.c1ccc2ccccc2c1",
            "SmallWorldIndexBR": "B34R5",
            "SmallWorldIndexBRL": "B34R5L18",
            "cx_smiles": "Cc1c(CCN2CCOCC2)c2ccccc2n1C(=O)c1cccc2ccccc12",
            "inchi": "InChI=1S/C26H26N2O2/c1-19-21(13-14-27-15-17-30-18-16-27)23-10-4-5-12-25(23)28(19)26(29)24-11-6-8-20-7-2-3-9-22(20)24/h2-12H,13-18H2,1H3",
            "inchi_KET_15T": "InChI=1/C26H26N2O2/c1-19-21(13-14-27-15-17-30-18-16-27)23-10-4-5-12-25(23)28(19)26(29)24-11-6-8-20-7-2-3-9-22(20)24/h2-12H,13-18H2,1H3",
            "inchi_key": "FKFDKPAJXHDYTK-UHFFFAOYSA-N",
            "inchikey_KET_15T": "FKFDKPAJXHDYTK-UHFFFAOYNA-N",
            "noiso_smiles": "Cc1c(CCN2CCOCC2)c2ccccc2n1C(=O)c1cccc2ccccc12",
            "smiles": "Cc1c(CCN2CCOCC2)c2ccccc2n1C(=O)c1cccc2ccccc12",
        },
        "ra7": {
            "AnonymousGraph": "**1***(*)*(**2**(*(*)*3****4*****34)*3*****23)*1",
            "ArthorSubstructureOrder": "001e002201001a00040000b9000000",
            "AtomBondCounts": "30,34",
            "CanonicalSmiles": "CN1CCN(C)C(Cn2cc(C(=O)c3cccc4ccccc34)c3ccccc32)C1",
            "DegreeVector": "0,11,16,3",
            "ElementGraph": "CN1CCN(C)C(CN2CC(C(O)C3CCCC4CCCCC34)C3CCCCC32)C1",
            "ExtendedMurcko": "*N1CCN(*)C(Cn2cc(C(=*)c3cccc4ccccc34)c3ccccc32)C1",
            "HetAtomProtomer": "CN1CCN(C)C(CN2[CH][C]([C]([O])[C]3[CH][CH][CH][C]4[CH][CH][CH][CH][C]43)[C]3[CH][CH][CH][CH][C]32)C1_0",
            "HetAtomTautomer": "CN1CCN(C)C(CN2[CH][C]([C]([O])[C]3[CH][CH][CH][C]4[CH][CH][CH][CH][C]43)[C]3[CH][CH][CH][CH][C]32)C1_0_0",
            "Mesomer": "CN1CCN(C)C(CN2[CH][C]([C]([O])[C]3[CH][CH][CH][C]4[CH][CH][CH][CH][C]34)[C]3[CH][CH][CH][CH][C]32)C1_0",
            "MolFormula": "C26H27N3O",
            "MurckoScaffold": "c1ccc2c(Cc3cn(CC4CNCCN4)c4ccccc34)cccc2c1",
            "NetCharge": "0",
            "RedoxPair": "CN1CCN(C)C(CN2[CH][C]([C]([O])[C]3[CH][CH][CH][C]4[CH][CH][CH][CH][C]34)[C]3[CH][CH][CH][CH][C]32)C1",
            "Regioisomer": "*C.*C.*C(*)=O.*C*.C1CNCCN1.c1ccc2[nH]ccc2c1.c1ccc2ccccc2c1",
            "SmallWorldIndexBR": "B34R5",
            "SmallWorldIndexBRL": "B34R5L16",
            "cx_smiles": "CN1CCN(C)C(Cn2cc(C(=O)c3cccc4ccccc34)c3ccccc32)C1",
            "inchi": "InChI=1S/C26H27N3O/c1-27-14-15-28(2)20(16-27)17-29-18-24(22-11-5-6-13-25(22)29)26(30)23-12-7-9-19-8-3-4-10-21(19)23/h3-13,18,20H,14-17H2,1-2H3",
            "inchi_KET_15T": "InChI=1/C26H27N3O/c1-27-14-15-28(2)20(16-27)17-29-18-24(22-11-5-6-13-25(22)29)26(30)23-12-7-9-19-8-3-4-10-21(19)23/h3-13,18,20H,14-17H2,1-2H3",
            "inchi_key": "CPHORMQWXVSQFN-UHFFFAOYSA-N",
            "inchikey_KET_15T": "CPHORMQWXVSQFN-UHFFFAOYNA-N",
            "noiso_smiles": "CN1CCN(C)C(Cn2cc(C(=O)c3cccc4ccccc34)c3ccccc32)C1",
            "smiles": "CN1CCN(C)C(Cn2cc(C(=O)c3cccc4ccccc34)c3ccccc32)C1",
        },
        "sa1": {
            "AnonymousGraph": "***(*)*(*)*",
            "ArthorSubstructureOrder": "000700060100050002000037000000",
            "AtomBondCounts": "7,6",
            "CanonicalSmiles": "CC[C@@H](C)[C@H](O)Cl",
            "DegreeVector": "0,2,1,4",
            "ElementGraph": "CC[C@@H](C)[C@H](O)Cl",
            "ExtendedMurcko": "",
            "HetAtomProtomer": "CC[C@@H](C)[C@H]([O])Cl_1",
            "HetAtomTautomer": "CC[C@@H](C)[C@H]([O])Cl_1_0",
            "Mesomer": "CC[C@@H](C)[C@H](O)Cl_0",
            "MolFormula": "C5H11ClO",
            "MurckoScaffold": "",
            "NetCharge": "0",
            "RedoxPair": "CC[C@@H](C)[C@H](O)Cl",
            "Regioisomer": "*Cl.*O.CCC(C)C",
            "SmallWorldIndexBR": "B6R0",
            "SmallWorldIndexBRL": "B6R0L1",
            "cx_smiles": "CC[C@@H](C)[C@H](O)Cl",
            "inchi": "InChI=1S/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4-,5+/m1/s1",
            "inchi_KET_15T": "InChI=1/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4-,5+/m1/s1",
            "inchi_key": "ZXUZVOQOEXOEFU-UHNVWZDZSA-N",
            "inchikey_KET_15T": "ZXUZVOQOEXOEFU-UHNVWZDZNA-N",
            "noiso_smiles": "CCC(C)C(O)Cl",
            "smiles": "CC[C@@H](C)[C@H](O)Cl",
        },
        "sa2": {
            "AnonymousGraph": "***(*)*(*)*",
            "ArthorSubstructureOrder": "000700060100050002000037000000",
            "AtomBondCounts": "7,6",
            "CanonicalSmiles": "CC[C@@H](C)[C@@H](O)Cl",
            "DegreeVector": "0,2,1,4",
            "ElementGraph": "CC[C@@H](C)[C@@H](O)Cl",
            "ExtendedMurcko": "",
            "HetAtomProtomer": "CC[C@@H](C)[C@@H]([O])Cl_1",
            "HetAtomTautomer": "CC[C@@H](C)[C@@H]([O])Cl_1_0",
            "Mesomer": "CC[C@@H](C)[C@@H](O)Cl_0",
            "MolFormula": "C5H11ClO",
            "MurckoScaffold": "",
            "NetCharge": "0",
            "RedoxPair": "CC[C@@H](C)[C@@H](O)Cl",
            "Regioisomer": "*Cl.*O.CCC(C)C",
            "SmallWorldIndexBR": "B6R0",
            "SmallWorldIndexBRL": "B6R0L1",
            "cx_smiles": "CC[C@@H](C)[C@@H](O)Cl",
            "inchi": "InChI=1S/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4-,5-/m1/s1",
            "inchi_KET_15T": "InChI=1/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4-,5-/m1/s1",
            "inchi_key": "ZXUZVOQOEXOEFU-RFZPGFLSSA-N",
            "inchikey_KET_15T": "ZXUZVOQOEXOEFU-RFZPGFLSNA-N",
            "noiso_smiles": "CCC(C)C(O)Cl",
            "smiles": "CC[C@@H](C)[C@@H](O)Cl",
        },
        "sa3": {
            "AnonymousGraph": "***(*)*(*)*",
            "ArthorSubstructureOrder": "000700060100050002000037000000",
            "AtomBondCounts": "7,6",
            "CanonicalSmiles": "CC[C@@H](C)C(O)Cl",
            "DegreeVector": "0,2,1,4",
            "ElementGraph": "CC[C@@H](C)C(O)Cl",
            "ExtendedMurcko": "",
            "HetAtomProtomer": "CC[C@@H](C)C([O])Cl_1",
            "HetAtomTautomer": "CC[C@@H](C)C([O])Cl_1_0",
            "Mesomer": "CC[C@@H](C)C(O)Cl_0",
            "MolFormula": "C5H11ClO",
            "MurckoScaffold": "",
            "NetCharge": "0",
            "RedoxPair": "CC[C@@H](C)C(O)Cl",
            "Regioisomer": "*Cl.*O.CCC(C)C",
            "SmallWorldIndexBR": "B6R0",
            "SmallWorldIndexBRL": "B6R0L1",
            "cx_smiles": "CC[C@@H](C)C(O)Cl",
            "inchi": "InChI=1S/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4-,5?/m1/s1",
            "inchi_KET_15T": "InChI=1/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4-,5?/m1/s1",
            "inchi_key": "ZXUZVOQOEXOEFU-CNZKWPKMSA-N",
            "inchikey_KET_15T": "ZXUZVOQOEXOEFU-CNZKWPKMNA-N",
            "noiso_smiles": "CCC(C)C(O)Cl",
            "smiles": "CC[C@@H](C)C(O)Cl",
        },
        "sa4": {
            "AnonymousGraph": "***(*)*(*)*",
            "ArthorSubstructureOrder": "000700060100050002000037000000",
            "AtomBondCounts": "7,6",
            "CanonicalSmiles": "CC[C@H](C)[C@H](O)Cl",
            "DegreeVector": "0,2,1,4",
            "ElementGraph": "CC[C@H](C)[C@H](O)Cl",
            "ExtendedMurcko": "",
            "HetAtomProtomer": "CC[C@H](C)[C@H]([O])Cl_1",
            "HetAtomTautomer": "CC[C@H](C)[C@H]([O])Cl_1_0",
            "Mesomer": "CC[C@H](C)[C@H](O)Cl_0",
            "MolFormula": "C5H11ClO",
            "MurckoScaffold": "",
            "NetCharge": "0",
            "RedoxPair": "CC[C@H](C)[C@H](O)Cl",
            "Regioisomer": "*Cl.*O.CCC(C)C",
            "SmallWorldIndexBR": "B6R0",
            "SmallWorldIndexBRL": "B6R0L1",
            "cx_smiles": "CC[C@H](C)[C@H](O)Cl",
            "inchi": "InChI=1S/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4-,5-/m0/s1",
            "inchi_KET_15T": "InChI=1/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4-,5-/m0/s1",
            "inchi_key": "ZXUZVOQOEXOEFU-WHFBIAKZSA-N",
            "inchikey_KET_15T": "ZXUZVOQOEXOEFU-WHFBIAKZNA-N",
            "noiso_smiles": "CCC(C)C(O)Cl",
            "smiles": "CC[C@H](C)[C@H](O)Cl",
        },
        "sa5": {
            "AnonymousGraph": "***(*)*(*)*",
            "ArthorSubstructureOrder": "000700060100050002000037000000",
            "AtomBondCounts": "7,6",
            "CanonicalSmiles": "CC[C@H](C)[C@@H](O)Cl",
            "DegreeVector": "0,2,1,4",
            "ElementGraph": "CC[C@H](C)[C@@H](O)Cl",
            "ExtendedMurcko": "",
            "HetAtomProtomer": "CC[C@H](C)[C@@H]([O])Cl_1",
            "HetAtomTautomer": "CC[C@H](C)[C@@H]([O])Cl_1_0",
            "Mesomer": "CC[C@H](C)[C@@H](O)Cl_0",
            "MolFormula": "C5H11ClO",
            "MurckoScaffold": "",
            "NetCharge": "0",
            "RedoxPair": "CC[C@H](C)[C@@H](O)Cl",
            "Regioisomer": "*Cl.*O.CCC(C)C",
            "SmallWorldIndexBR": "B6R0",
            "SmallWorldIndexBRL": "B6R0L1",
            "cx_smiles": "CC[C@H](C)[C@@H](O)Cl",
            "inchi": "InChI=1S/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4-,5+/m0/s1",
            "inchi_KET_15T": "InChI=1/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4-,5+/m0/s1",
            "inchi_key": "ZXUZVOQOEXOEFU-CRCLSJGQSA-N",
            "inchikey_KET_15T": "ZXUZVOQOEXOEFU-CRCLSJGQNA-N",
            "noiso_smiles": "CCC(C)C(O)Cl",
            "smiles": "CC[C@H](C)[C@@H](O)Cl",
        },
        "sa6": {
            "AnonymousGraph": "***(*)*(*)*",
            "ArthorSubstructureOrder": "000700060100050002000037000000",
            "AtomBondCounts": "7,6",
            "CanonicalSmiles": "CC[C@H](C)C(O)Cl",
            "DegreeVector": "0,2,1,4",
            "ElementGraph": "CC[C@H](C)C(O)Cl",
            "ExtendedMurcko": "",
            "HetAtomProtomer": "CC[C@H](C)C([O])Cl_1",
            "HetAtomTautomer": "CC[C@H](C)C([O])Cl_1_0",
            "Mesomer": "CC[C@H](C)C(O)Cl_0",
            "MolFormula": "C5H11ClO",
            "MurckoScaffold": "",
            "NetCharge": "0",
            "RedoxPair": "CC[C@H](C)C(O)Cl",
            "Regioisomer": "*Cl.*O.CCC(C)C",
            "SmallWorldIndexBR": "B6R0",
            "SmallWorldIndexBRL": "B6R0L1",
            "cx_smiles": "CC[C@H](C)C(O)Cl",
            "inchi": "InChI=1S/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4-,5?/m0/s1",
            "inchi_KET_15T": "InChI=1/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4-,5?/m0/s1",
            "inchi_key": "ZXUZVOQOEXOEFU-ROLXFIACSA-N",
            "inchikey_KET_15T": "ZXUZVOQOEXOEFU-ROLXFIACNA-N",
            "noiso_smiles": "CCC(C)C(O)Cl",
            "smiles": "CC[C@H](C)C(O)Cl",
        },
        "sa7": {
            "AnonymousGraph": "***(*)*(*)*",
            "ArthorSubstructureOrder": "000700060100050002000037000000",
            "AtomBondCounts": "7,6",
            "CanonicalSmiles": "CCC(C)[C@H](O)Cl",
            "DegreeVector": "0,2,1,4",
            "ElementGraph": "CCC(C)[C@H](O)Cl",
            "ExtendedMurcko": "",
            "HetAtomProtomer": "CCC(C)[C@H]([O])Cl_1",
            "HetAtomTautomer": "CCC(C)[C@H]([O])Cl_1_0",
            "Mesomer": "CCC(C)[C@H](O)Cl_0",
            "MolFormula": "C5H11ClO",
            "MurckoScaffold": "",
            "NetCharge": "0",
            "RedoxPair": "CCC(C)[C@H](O)Cl",
            "Regioisomer": "*Cl.*O.CCC(C)C",
            "SmallWorldIndexBR": "B6R0",
            "SmallWorldIndexBRL": "B6R0L1",
            "cx_smiles": "CCC(C)[C@H](O)Cl",
            "inchi": "InChI=1S/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4?,5-/m0/s1",
            "inchi_KET_15T": "InChI=1/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4?,5-/m0/s1",
            "inchi_key": "ZXUZVOQOEXOEFU-AKGZTFGVSA-N",
            "inchikey_KET_15T": "ZXUZVOQOEXOEFU-AKGZTFGVNA-N",
            "noiso_smiles": "CCC(C)C(O)Cl",
            "smiles": "CCC(C)[C@H](O)Cl",
        },
        "sa8": {
            "AnonymousGraph": "***(*)*(*)*",
            "ArthorSubstructureOrder": "000700060100050002000037000000",
            "AtomBondCounts": "7,6",
            "CanonicalSmiles": "CCC(C)[C@@H](O)Cl",
            "DegreeVector": "0,2,1,4",
            "ElementGraph": "CCC(C)[C@@H](O)Cl",
            "ExtendedMurcko": "",
            "HetAtomProtomer": "CCC(C)[C@@H]([O])Cl_1",
            "HetAtomTautomer": "CCC(C)[C@@H]([O])Cl_1_0",
            "Mesomer": "CCC(C)[C@@H](O)Cl_0",
            "MolFormula": "C5H11ClO",
            "MurckoScaffold": "",
            "NetCharge": "0",
            "RedoxPair": "CCC(C)[C@@H](O)Cl",
            "Regioisomer": "*Cl.*O.CCC(C)C",
            "SmallWorldIndexBR": "B6R0",
            "SmallWorldIndexBRL": "B6R0L1",
            "cx_smiles": "CCC(C)[C@@H](O)Cl",
            "inchi": "InChI=1S/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4?,5-/m1/s1",
            "inchi_KET_15T": "InChI=1/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3/t4?,5-/m1/s1",
            "inchi_key": "ZXUZVOQOEXOEFU-BRJRFNKRSA-N",
            "inchikey_KET_15T": "ZXUZVOQOEXOEFU-BRJRFNKRNA-N",
            "noiso_smiles": "CCC(C)C(O)Cl",
            "smiles": "CCC(C)[C@@H](O)Cl",
        },
        "sa9": {
            "AnonymousGraph": "***(*)*(*)*",
            "ArthorSubstructureOrder": "000700060100050002000037000000",
            "AtomBondCounts": "7,6",
            "CanonicalSmiles": "CCC(C)C(O)Cl",
            "DegreeVector": "0,2,1,4",
            "ElementGraph": "CCC(C)C(O)Cl",
            "ExtendedMurcko": "",
            "HetAtomProtomer": "CCC(C)C([O])Cl_1",
            "HetAtomTautomer": "CCC(C)C([O])Cl_1_0",
            "Mesomer": "CCC(C)C(O)Cl_0",
            "MolFormula": "C5H11ClO",
            "MurckoScaffold": "",
            "NetCharge": "0",
            "RedoxPair": "CCC(C)C(O)Cl",
            "Regioisomer": "*Cl.*O.CCC(C)C",
            "SmallWorldIndexBR": "B6R0",
            "SmallWorldIndexBRL": "B6R0L1",
            "cx_smiles": "CCC(C)C(O)Cl",
            "inchi": "InChI=1S/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3",
            "inchi_KET_15T": "InChI=1/C5H11ClO/c1-3-4(2)5(6)7/h4-5,7H,3H2,1-2H3",
            "inchi_key": "ZXUZVOQOEXOEFU-UHFFFAOYSA-N",
            "inchikey_KET_15T": "ZXUZVOQOEXOEFU-UHFFFAOYNA-N",
            "noiso_smiles": "CCC(C)C(O)Cl",
            "smiles": "CCC(C)C(O)Cl",
        },
        "ta1": {
            "AnonymousGraph": "**1****1",
            "ArthorSubstructureOrder": "000600060100040002000027000000",
            "AtomBondCounts": "6,6",
            "CanonicalSmiles": "OC1=NCCC1",
            "DegreeVector": "0,1,4,1",
            "ElementGraph": "OC1CCCN1",
            "ExtendedMurcko": "*C1=NCCC1",
            "HetAtomProtomer": "[O][C]1CCC[N]1_1",
            "HetAtomTautomer": "[O][C]1CCC[N]1_1_0",
            "Mesomer": "O[C]1CCC[N]1_0",
            "MolFormula": "C4H7NO",
            "MurckoScaffold": "C1=NCCC1",
            "NetCharge": "0",
            "RedoxPair": "O[C]1CCC[N]1",
            "Regioisomer": "*O.C1=NCCC1",
            "SmallWorldIndexBR": "B6R1",
            "SmallWorldIndexBRL": "B6R1L4",
            "cx_smiles": "OC1=NCCC1",
            "inchi": "InChI=1S/C4H7NO/c6-4-2-1-3-5-4/h1-3H2,(H,5,6)",
            "inchi_KET_15T": "InChI=1/C4H7NO/c6-4-2-1-3-5-4/h1,3H2,(H3,2,5,6)",
            "inchi_key": "HNJBEVLQSNELDL-UHFFFAOYSA-N",
            "inchikey_KET_15T": "DXJWFBGIRPOVOQ-UHFFFAOYNA-N",
            "noiso_smiles": "OC1=NCCC1",
            "smiles": "OC1=NCCC1",
        },
        "ta2": {
            "AnonymousGraph": "**1****1",
            "ArthorSubstructureOrder": "000600060100040002000027000000",
            "AtomBondCounts": "6,6",
            "CanonicalSmiles": "O=C1CCCN1",
            "DegreeVector": "0,1,4,1",
            "ElementGraph": "OC1CCCN1",
            "ExtendedMurcko": "*=C1CCCN1",
            "HetAtomProtomer": "[O][C]1CCC[N]1_1",
            "HetAtomTautomer": "[O][C]1CCC[N]1_1_0",
            "Mesomer": "[O][C]1CCCN1_0",
            "MolFormula": "C4H7NO",
            "MurckoScaffold": "C1CCNC1",
            "NetCharge": "0",
            "RedoxPair": "[O][C]1CCCN1",
            "Regioisomer": "O=C1CCCN1",
            "SmallWorldIndexBR": "B6R1",
            "SmallWorldIndexBRL": "B6R1L4",
            "cx_smiles": "O=C1CCCN1",
            "inchi": "InChI=1S/C4H7NO/c6-4-2-1-3-5-4/h1-3H2,(H,5,6)",
            "inchi_KET_15T": "InChI=1/C4H7NO/c6-4-2-1-3-5-4/h1,3H2,(H3,2,5,6)",
            "inchi_key": "HNJBEVLQSNELDL-UHFFFAOYSA-N",
            "inchikey_KET_15T": "DXJWFBGIRPOVOQ-UHFFFAOYNA-N",
            "noiso_smiles": "O=C1CCCN1",
            "smiles": "O=C1CCCN1",
        },
        "tb1": {
            "AnonymousGraph": "***(*)*1*(*)***(*)*1*",
            "ArthorSubstructureOrder": "000d000d01000a0003000053000000",
            "AtomBondCounts": "13,13",
            "CanonicalSmiles": "C/N=C(\\C)C1C(=O)CCC(C)C1=O",
            "DegreeVector": "0,5,3,5",
            "ElementGraph": "CNC(C)C1C(O)CCC(C)C1O",
            "ExtendedMurcko": "*C1CCC(=*)C(*)C1=*",
            "HetAtomProtomer": "C[N][C](C)C1[C]([O])CCC(C)[C]1[O]_0",
            "HetAtomTautomer": "C[N][C](C)C1[C]([O])CCC(C)[C]1[O]_0_0",
            "Mesomer": "C[N][C](C)C1[C]([O])CCC(C)[C]1[O]_0",
            "MolFormula": "C10H15NO2",
            "MurckoScaffold": "C1CCCCC1",
            "NetCharge": "0",
            "RedoxPair": "C[N][C](C)C1[C]([O])CCC(C)[C]1[O]",
            "Regioisomer": "*C.*N=C(*)C.C.O=C1CCCC(=O)C1",
            "SmallWorldIndexBR": "B13R1",
            "SmallWorldIndexBRL": "B13R1L3",
            "cx_smiles": "C/N=C(\\C)C1C(=O)CCC(C)C1=O",
            "inchi": "InChI=1S/C10H15NO2/c1-6-4-5-8(12)9(10(6)13)7(2)11-3/h6,9H,4-5H2,1-3H3/b11-7+",
            "inchi_KET_15T": "InChI=1/C10H15NO2/c1-6-4-5-8(12)9(10(6)13)7(2)11-3/h4H2,1-3H3,(H4,5,6,9,11,12,13)",
            "inchi_key": "LFFJNZFRBPZXBS-YRNVUSSQSA-N",
            "inchikey_KET_15T": "SJQNTMZZZVTREP-UHFFFAOYNA-N",
            "noiso_smiles": "CN=C(C)C1C(=O)CCC(C)C1=O",
            "smiles": "C/N=C(\\C)C1C(=O)CCC(C)C1=O",
        },
        "tb2": {
            "AnonymousGraph": "***(*)*1*(*)***(*)*1*",
            "ArthorSubstructureOrder": "000d000d01000a0003000053000000",
            "AtomBondCounts": "13,13",
            "CanonicalSmiles": "C/N=C(\\C)C1=C(O)CCC(C)C1=O",
            "DegreeVector": "0,5,3,5",
            "ElementGraph": "CNC(C)C1C(O)CCC(C)C1O",
            "ExtendedMurcko": "*C1=C(*)C(=*)C(*)CC1",
            "HetAtomProtomer": "C[N][C](C)[C]1[C]([O])CCC(C)[C]1[O]_1",
            "HetAtomTautomer": "C[N][C](C)[C]1[C]([O])CCC(C)[C]1[O]_1_0",
            "Mesomer": "C[N][C](C)[C]1[C](O)CCC(C)[C]1[O]_0",
            "MolFormula": "C10H15NO2",
            "MurckoScaffold": "C1=CCCCC1",
            "NetCharge": "0",
            "RedoxPair": "C[N][C](C)[C]1[C](O)CCC(C)[C]1[O]",
            "Regioisomer": "*C.*N=C(*)C.*O.C.O=C1C=CCCC1",
            "SmallWorldIndexBR": "B13R1",
            "SmallWorldIndexBRL": "B13R1L3",
            "cx_smiles": "C/N=C(\\C)C1=C(O)CCC(C)C1=O",
            "inchi": "InChI=1S/C10H15NO2/c1-6-4-5-8(12)9(10(6)13)7(2)11-3/h6,12H,4-5H2,1-3H3/b11-7+",
            "inchi_KET_15T": "InChI=1/C10H15NO2/c1-6-4-5-8(12)9(10(6)13)7(2)11-3/h4H2,1-3H3,(H4,5,6,9,11,12,13)",
            "inchi_key": "MCKSLMLPGJYCNP-YRNVUSSQSA-N",
            "inchikey_KET_15T": "SJQNTMZZZVTREP-UHFFFAOYNA-N",
            "noiso_smiles": "CN=C(C)C1=C(O)CCC(C)C1=O",
            "smiles": "C/N=C(\\C)C1=C(O)CCC(C)C1=O",
        },
        "tb3": {
            "AnonymousGraph": "***(*)*1*(*)***(*)*1*",
            "ArthorSubstructureOrder": "000d000d01000a0003000053000000",
            "AtomBondCounts": "13,13",
            "CanonicalSmiles": "C/N=C(\\C)C1=C(O)C(C)CCC1=O",
            "DegreeVector": "0,5,3,5",
            "ElementGraph": "CNC(C)C1C(O)CCC(C)C1O",
            "ExtendedMurcko": "*C1=C(*)C(*)CCC1=*",
            "HetAtomProtomer": "C[N][C](C)[C]1[C]([O])CCC(C)[C]1[O]_1",
            "HetAtomTautomer": "C[N][C](C)[C]1[C]([O])CCC(C)[C]1[O]_1_0",
            "Mesomer": "C[N][C](C)[C]1[C]([O])CCC(C)[C]1O_0",
            "MolFormula": "C10H15NO2",
            "MurckoScaffold": "C1=CCCCC1",
            "NetCharge": "0",
            "RedoxPair": "C[N][C](C)[C]1[C]([O])CCC(C)[C]1O",
            "Regioisomer": "*C.*N=C(*)C.*O.C.O=C1C=CCCC1",
            "SmallWorldIndexBR": "B13R1",
            "SmallWorldIndexBRL": "B13R1L3",
            "cx_smiles": "C/N=C(\\C)C1=C(O)C(C)CCC1=O",
            "inchi": "InChI=1S/C10H15NO2/c1-6-4-5-8(12)9(10(6)13)7(2)11-3/h6,13H,4-5H2,1-3H3/b11-7+",
            "inchi_KET_15T": "InChI=1/C10H15NO2/c1-6-4-5-8(12)9(10(6)13)7(2)11-3/h4H2,1-3H3,(H4,5,6,9,11,12,13)",
            "inchi_key": "LQHZFYQSMISUAH-YRNVUSSQSA-N",
            "inchikey_KET_15T": "SJQNTMZZZVTREP-UHFFFAOYNA-N",
            "noiso_smiles": "CN=C(C)C1=C(O)C(C)CCC1=O",
            "smiles": "C/N=C(\\C)C1=C(O)C(C)CCC1=O",
        },
        "tc1": {
            "AnonymousGraph": "**(*)*1*(*)***(*)*1*",
            "ArthorSubstructureOrder": "000c000c010009000300004e000000",
            "AtomBondCounts": "12,12",
            "CanonicalSmiles": "CC(=O)C1=C(O)C(C)CCC1=O",
            "DegreeVector": "0,5,2,5",
            "ElementGraph": "CC(O)C1C(O)CCC(C)C1O",
            "ExtendedMurcko": "*C1=C(*)C(*)CCC1=*",
            "HetAtomProtomer": "C[C]([O])[C]1[C]([O])CCC(C)[C]1[O]_1",
            "HetAtomTautomer": "C[C]([O])[C]1[C]([O])CCC(C)[C]1[O]_1_0",
            "Mesomer": "C[C]([O])[C]1[C]([O])CCC(C)[C]1O_0",
            "MolFormula": "C9H12O3",
            "MurckoScaffold": "C1=CCCCC1",
            "NetCharge": "0",
            "RedoxPair": "C[C]([O])[C]1[C]([O])CCC(C)[C]1O",
            "Regioisomer": "*C.*C(C)=O.*O.O=C1C=CCCC1",
            "SmallWorldIndexBR": "B12R1",
            "SmallWorldIndexBRL": "B12R1L2",
            "cx_smiles": "CC(=O)C1=C(O)C(C)CCC1=O",
            "inchi": "InChI=1S/C9H12O3/c1-5-3-4-7(11)8(6(2)10)9(5)12/h5,12H,3-4H2,1-2H3",
            "inchi_KET_15T": "InChI=1/C9H12O3/c1-5-3-4-7(11)8(6(2)10)9(5)12/h3H2,1-2H3,(H4,4,5,8,10,11,12)",
            "inchi_key": "JKVFZUSRZCBLOO-UHFFFAOYSA-N",
            "inchikey_KET_15T": "XCMTZZNIZOTRIL-UHFFFAOYNA-N",
            "noiso_smiles": "CC(=O)C1=C(O)C(C)CCC1=O",
            "smiles": "CC(=O)C1=C(O)C(C)CCC1=O",
        },
        "tc2": {
            "AnonymousGraph": "**(*)*1*(*)***(*)*1*",
            "ArthorSubstructureOrder": "000c000c010009000300004e000000",
            "AtomBondCounts": "12,12",
            "CanonicalSmiles": "CC(=O)C1C(=O)CCC(C)C1=O",
            "DegreeVector": "0,5,2,5",
            "ElementGraph": "CC(O)C1C(O)CCC(C)C1O",
            "ExtendedMurcko": "*C1CCC(=*)C(*)C1=*",
            "HetAtomProtomer": "C[C]([O])C1[C]([O])CCC(C)[C]1[O]_0",
            "HetAtomTautomer": "C[C]([O])C1[C]([O])CCC(C)[C]1[O]_0_0",
            "Mesomer": "C[C]([O])C1[C]([O])CCC(C)[C]1[O]_0",
            "MolFormula": "C9H12O3",
            "MurckoScaffold": "C1CCCCC1",
            "NetCharge": "0",
            "RedoxPair": "C[C]([O])C1[C]([O])CCC(C)[C]1[O]",
            "Regioisomer": "*C.*C(C)=O.O=C1CCCC(=O)C1",
            "SmallWorldIndexBR": "B12R1",
            "SmallWorldIndexBRL": "B12R1L2",
            "cx_smiles": "CC(=O)C1C(=O)CCC(C)C1=O",
            "inchi": "InChI=1S/C9H12O3/c1-5-3-4-7(11)8(6(2)10)9(5)12/h5,8H,3-4H2,1-2H3",
            "inchi_KET_15T": "InChI=1/C9H12O3/c1-5-3-4-7(11)8(6(2)10)9(5)12/h3H2,1-2H3,(H4,4,5,8,10,11,12)",
            "inchi_key": "MMPTYALDYQTEML-UHFFFAOYSA-N",
            "inchikey_KET_15T": "XCMTZZNIZOTRIL-UHFFFAOYNA-N",
            "noiso_smiles": "CC(=O)C1C(=O)CCC(C)C1=O",
            "smiles": "CC(=O)C1C(=O)CCC(C)C1=O",
        },
        "tc3": {
            "AnonymousGraph": "**(*)*1*(*)***(*)*1*",
            "ArthorSubstructureOrder": "000c000c010009000300004e000000",
            "AtomBondCounts": "12,12",
            "CanonicalSmiles": "CC(=O)C1=C(O)CCC(C)C1=O",
            "DegreeVector": "0,5,2,5",
            "ElementGraph": "CC(O)C1C(O)CCC(C)C1O",
            "ExtendedMurcko": "*C1=C(*)C(=*)C(*)CC1",
            "HetAtomProtomer": "C[C]([O])[C]1[C]([O])CCC(C)[C]1[O]_1",
            "HetAtomTautomer": "C[C]([O])[C]1[C]([O])CCC(C)[C]1[O]_1_0",
            "Mesomer": "C[C]([O])[C]1[C](O)CCC(C)[C]1[O]_0",
            "MolFormula": "C9H12O3",
            "MurckoScaffold": "C1=CCCCC1",
            "NetCharge": "0",
            "RedoxPair": "C[C]([O])[C]1[C](O)CCC(C)[C]1[O]",
            "Regioisomer": "*C.*C(C)=O.*O.O=C1C=CCCC1",
            "SmallWorldIndexBR": "B12R1",
            "SmallWorldIndexBRL": "B12R1L2",
            "cx_smiles": "CC(=O)C1=C(O)CCC(C)C1=O",
            "inchi": "InChI=1S/C9H12O3/c1-5-3-4-7(11)8(6(2)10)9(5)12/h5,11H,3-4H2,1-2H3",
            "inchi_KET_15T": "InChI=1/C9H12O3/c1-5-3-4-7(11)8(6(2)10)9(5)12/h3H2,1-2H3,(H4,4,5,8,10,11,12)",
            "inchi_key": "UYIJIQGWGXARDH-UHFFFAOYSA-N",
            "inchikey_KET_15T": "XCMTZZNIZOTRIL-UHFFFAOYNA-N",
            "noiso_smiles": "CC(=O)C1=C(O)CCC(C)C1=O",
            "smiles": "CC(=O)C1=C(O)CCC(C)C1=O",
        },
        "tc4": {
            "AnonymousGraph": "**1***(*)*(*(*)*)*1*",
            "ArthorSubstructureOrder": "000c000c010009000300004e000000",
            "AtomBondCounts": "12,12",
            "CanonicalSmiles": "C/C(O)=C1\\C(=O)CCC(C)C1=O",
            "DegreeVector": "0,5,2,5",
            "ElementGraph": "CC1CCC(O)C(C(C)O)C1O",
            "ExtendedMurcko": "*C1CCC(=*)C(=*)C1=*",
            "HetAtomProtomer": "C[C]([O])[C]1[C]([O])CCC(C)[C]1[O]_1",
            "HetAtomTautomer": "C[C]([O])[C]1[C]([O])CCC(C)[C]1[O]_1_0",
            "Mesomer": "C[C](O)[C]1[C]([O])CCC(C)[C]1[O]_0",
            "MolFormula": "C9H12O3",
            "MurckoScaffold": "C1CCCCC1",
            "NetCharge": "0",
            "RedoxPair": "C[C](O)[C]1[C]([O])CCC(C)[C]1[O]",
            "Regioisomer": "*C.CC(O)=C1C(=O)CCCC1=O",
            "SmallWorldIndexBR": "B12R1",
            "SmallWorldIndexBRL": "B12R1L2",
            "cx_smiles": "C/C(O)=C1\\C(=O)CCC(C)C1=O",
            "inchi": "InChI=1S/C9H12O3/c1-5-3-4-7(11)8(6(2)10)9(5)12/h5,10H,3-4H2,1-2H3/b8-6-",
            "inchi_KET_15T": "InChI=1/C9H12O3/c1-5-3-4-7(11)8(6(2)10)9(5)12/h3H2,1-2H3,(H4,4,5,8,10,11,12)",
            "inchi_key": "LTGVOMHYBZTAIZ-VURMDHGXSA-N",
            "inchikey_KET_15T": "XCMTZZNIZOTRIL-UHFFFAOYNA-N",
            "noiso_smiles": "CC(O)=C1C(=O)CCC(C)C1=O",
            "smiles": "C/C(O)=C1\\C(=O)CCC(C)C1=O",
        },
    }
    calculated = {}
    for x in examples:
        rdmol = cif.Chem.MolFromSmiles(x.get("smiles"))
        hash_map = calculate_molecular_hash_values(rdmol=rdmol)
        name = x.get("name")
        calculated[name] = hash_map
        # print()
        # import pprint
        # pprint.pprint(calculated)
    # checking only few hashed descriptors that are valid across multiple RDKit versions
    hashed_descriptors = [
        "AnonymousGraph",
        "CanonicalSmiles",
        "inchi",
        "inchi_KET_15T",
        "inchi_key",
        "inchikey_KET_15T",
    ]
    for hd in hashed_descriptors:
        for k, v in calculated.items():
            assert v[hd] == reference.get(k).get(hd)
    # clever testing


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
    )  # same reaction, one reactant has a different smiles: test mol canonicalization
    assert ces1.get(0) == ces1.get(
        2
    )  # same reaction, two reactant have a different smiles: test mol canonicalization
    assert ces1.get(0) == ces1.get(
        3
    )  # same reaction, two reactant have a different smiles: test mol canonicalization
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
    smiles = "O[C:2](=[O:1])[c:3]1[cH:12][cH:7][cH:6][cH:5][cH:4]1.CN>ClCCl.ClC(=O)C(Cl)=O>C[NH:13][C:2](=[O:1])[c:3]1[cH:12][cH:7][cH:6][cH:5][cH:4]1"
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
            "smiles": "Cl",
            "role": "reagent_quench",
            "stoichiometry": 1,
        },
    ]
    ce = chemical_equation_constructor.build_from_db(db_list2)
    assert ce.role_map["reagents"] != []
    # print(cif.rdrxn_to_string(ce.rdrxn, "smiles"))
    # ce_from_string = chemical_equation_constructor.build_from_reaction_string(
    #     "O=C(O)CCl.Oc1cc(Cl)c(Cl)cc1Cl>([Na+].[OH-]).([Na+].[OH-]).Cl>([Cl-].[Na+]).([Cl-].[Na+]).O.O.O=C(O)COc1cc(Cl)c(Cl)cc1Cl",
    #     "smiles",
    # )


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
            cem = ratam_constructor.create_ratam(
                reaction_mols, reaction_mols["products"][0]
            )
            assert cem
            assert cem.atom_transformations
            map_numbers = set()
            for d in cem.full_map_info.values():
                for k, v in d.items():
                    map_numbers.update(
                        m for d in v for m in d.values() if m not in [0, -1]
                    )
            # check if an AtomTransformation exists for each map number
            assert len(map_numbers) == len(cem.atom_transformations)

            for role, map_info in cem.full_map_info.items():
                smiles_list = [
                    m.smiles for uid, m in catalog.items() if uid in map_info
                ]
                assert item["expected"][role] == smiles_list


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

    # the reactant hash is insensitive to the input order of reactants (reaction canonicalization OK)
    assert results.get(0).get("reactants") == results.get(1).get("reactants")
    # the product hash is insensitive to the input order of products (reaction canonicalization OK)
    assert results.get(7).get("products") == results.get(8).get("products")
    # the machinery does break when the reactants are missing: Template is None
    assert results.get(2) is None
    # the machinery does not break when the agents are missing
    assert results.get(3).get("reagents")
    # the machinery does break when the products are missing: Template is None
    assert results.get(4) is None
    # reagents are happily ignored
    # there is a special hash for missing roles (it is the hash of an empty string)
    assert results.get(3).get("reagents") == create_hash("")
    # the reactant and products hashes are conserved even when the reagents are missing
    assert results.get(0).get("reactants") == results.get(5).get("reactants")
    assert results.get(0).get("products") == results.get(5).get("products")
    # the base r>p hash is conserved if the agents are missing in one reaction
    assert results.get(0).get("r_p") == results.get(5).get("r_p")
    # the full r>a>p hash is conserved  if the reagents are missing in one reaction (reagents are ignored!!)
    assert results.get(0).get("r_r_p") == results.get(5).get("r_r_p")
    # the base r>>p hash is not conserved if the reaction is reversed
    assert results.get(0).get("r_p") != results.get(6).get("r_p")
    # the full r>a>p hash is not conserved if the reaction is reversed
    assert results.get(0).get("r_r_p") != results.get(6).get("r_r_p")
    # the reversible base r<>p hash is not conserved if the reaction is reversed (this comes from rdchiral teplate extraction)
    # in some special cases it might be true, but it not necessarily is
    assert results.get(0).get("u_r_p") != results.get(9).get("u_r_p")
    assert results.get(3).get("u_r_p") != results.get(6).get("u_r_p")
    # the reversible full r<a>p hash is not conserved if the reaction is reversed (this comes from rdchiral teplate extraction)
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
            f"The disconnection identifier is the same for some "
            f"reactions that have the same products but different reactants: \n"
            + f"{a}: {disconnection_a.uid} \n"
            f"{b}: {disconnection_b.uid} \n"
        )
        assert disconnection_a.hash_map.get(
            "disconnection_summary"
        ) != disconnection_b.hash_map.get("disconnection_summary"), (
            f"The disconnection summary is the same for some reactions that have the same "
            f"products but different reactants: \n"
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
            f"The disconnection identifier is the same for some "
            f"reactions that have different products but same reactants: \n"
            + f"{a}: {disconnection_a.uid} \n"
            f"{b}: {disconnection_b.uid} \n"
        )
        assert disconnection_a.hash_map.get(
            "disconnection_summary"
        ) != disconnection_b.hash_map.get("disconnection_summary"), (
            f"The disconnection summary is the same for some reactions that have the same "
            f"products but different reactants: \n"
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
            f"The disconnection identifier is the same for some "
            f"reactions that have different products but same reactants: \n"
            + f"{a}: {disconnection_a.uid} \n"
            f"{b}: {disconnection_b.uid} \n"
        )
        assert disconnection_a.hash_map.get(
            "disconnection_summary"
        ) != disconnection_b.hash_map.get("disconnection_summary"), (
            f"The disconnection summary is the same for some reactions that have the same "
            f"products but different reactants: \n"
            + f'{a}: {disconnection_a.hash_map.get("disconnection_summary")} \n'
            f'{b}: {disconnection_b.hash_map.get("disconnection_summary")} \n'
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
        # lio.write_rdkit_depict(data=depiction_data, file_path=f"{item.get('name')}_reaction.png")
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
