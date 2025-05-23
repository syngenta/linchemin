import unittest

import pytest

import linchemin.cheminfo.functions as cif
from linchemin.cheminfo.chemical_hashes import (
    CanonicalSmiles,
    CxSmiles,
    Inchi,
    InchiKET15,
    InchiKey,
    InchiKeyKET15,
    ReactantProduct,
    ReactantReagentProduct,
    ReactionIdentifiersFactory,
    UnavailableMolIdentifier,
    UnavailableReactionIdentifier,
    UnorderedReactantProduct,
    UnorderedReactantReagentProduct,
    calculate_molecular_hash_map,
    get_all_molecular_identifiers,
    get_all_reaction_identifiers,
    is_valid_molecular_identifier,
    is_valid_reaction_identifier,
    validate_molecular_identifier,
    validate_reaction_identifier,
)


@pytest.fixture
def molecules_map():
    return {"reactants": "CC(=O)O.CN", "reagents": "O", "products": "CNC(C)=O"}


@pytest.fixture
def example_smiles():
    """To get a list of example smiles"""
    return [
        {
            "name": "ra1",
            "smiles": "Cc1ccc2c(C(=O)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1",
        },
        {
            "name": "ra2",
            "smiles": "Cc1ccc2c(C(=O)c3cccc4ccccc34)cn(CCN3CCOCC3)c2c1",
        },
        {"name": "ra3", "smiles": "CCCCCn1cc(C(=O)c2cccc3ccccc23)c2ccccc21"},
        {"name": "ra4", "smiles": "CC1COCCN1CCn1cc(C(=O)c2cccc3ccccc23)c2ccccc21"},
        {
            "name": "ra5",
            "smiles": "Cc1ccc(C(=O)c2cn(CCN3CCOCC3)c3ccccc23)c2ccccc12",
        },
        {"name": "ra6", "smiles": "Cc1c(CCN2CCOCC2)c2ccccc2n1C(=O)c1cccc2ccccc12"},
        {
            "name": "ra7",
            "smiles": "CN1CCN(C)C(Cn2cc(C(=O)c3cccc4ccccc34)c3ccccc32)C1",
        },
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
        {"name": "es1", "smiles": "C[C@H](Cl)[C@H](F)C |&1:2,3|"},
    ]


@pytest.fixture
def reference_hashes():
    return {
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
        "es1": {
            "AnonymousGraph": "**(*)*(*)*",
            "ElementGraph": "C[C@H](Cl)[C@H](C)F",
            "CanonicalSmiles": "C[C@H](Cl)[C@@H](C)F",
            "MolFormula": "C4H8ClF",
            "AtomBondCounts": "6,5",
            "DegreeVector": "0,2,0,4",
            "Mesomer": "C[C@H](Cl)[C@H](C)F_0",
            "HetAtomTautomer": "C[C@H](Cl)[C@H](C)F_0_0",
            "HetAtomProtomer": "C[C@H](Cl)[C@H](C)F_0",
            "RedoxPair": "C[C@H](Cl)[C@H](C)F",
            "Regioisomer": "*Cl.*F.CCCC",
            "NetCharge": "0",
            "SmallWorldIndexBR": "B5R0",
            "SmallWorldIndexBRL": "B5R0L0",
            "ArthorSubstructureOrder": "000600050100040002000032000000",
            "HetAtomTautomerv2": "[CH3]-[C@H](-[Cl])-[C@H](-[CH3])-[F]_0_0",
            "cx_smiles": "C[C@H](Cl)[C@H](C)F |&1:3|",
            "noiso_smiles": "CC(F)C(C)Cl",
            "inchi": "InChI=1S/C4H8ClF/c1-3(5)4(2)6/h3-4H,1-2H3/t3-,4+/m0/s1",
            "inchi_KET_15T": "InChI=1/C4H8ClF/c1-3(5)4(2)6/h3-4H,1-2H3/t3-,4+/m0/s1",
            "inchi_key": "UZCSZAFBHDEEHP-IUYQGCFVSA-N",
            "inchikey_KET_15T": "UZCSZAFBHDEEHP-IUYQGCFVNA-N",
            "smiles": "C[C@H](Cl)[C@@H](C)F",
        },
    }


def test_inchi(example_smiles, reference_hashes):
    identifier = Inchi()
    calculated = {}
    for x in example_smiles:
        rdmol = cif.Chem.MolFromSmiles(x.get("smiles"))
        hash_key = identifier.get_identifier(rdmol=rdmol)
        name = x.get("name")
        calculated[name] = hash_key
    for name, v in calculated.items():
        assert v == reference_hashes.get(name).get("inchi")


def test_inchi_key(example_smiles, reference_hashes):
    identifier = InchiKey()
    calculated = {}
    for x in example_smiles:
        rdmol = cif.Chem.MolFromSmiles(x.get("smiles"))
        hash_key = identifier.get_identifier(rdmol=rdmol)
        name = x.get("name")
        calculated[name] = hash_key

    for name, v in calculated.items():
        assert v == reference_hashes.get(name).get("inchi_key")


def test_inchiket15(example_smiles, reference_hashes):
    identifier = InchiKET15()
    calculated = {}
    for x in example_smiles:
        rdmol = cif.Chem.MolFromSmiles(x.get("smiles"))
        hash_key = identifier.get_identifier(rdmol=rdmol)
        name = x.get("name")
        calculated[name] = hash_key

    for name, v in calculated.items():
        assert v == reference_hashes.get(name).get("inchi_KET_15T")


def test_inchi_key_ket15(example_smiles, reference_hashes):
    identifier = InchiKeyKET15()
    calculated = {}
    for x in example_smiles:
        rdmol = cif.Chem.MolFromSmiles(x.get("smiles"))
        hash_key = identifier.get_identifier(rdmol=rdmol)
        name = x.get("name")
        calculated[name] = hash_key

    for name, v in calculated.items():
        assert v == reference_hashes.get(name).get("inchikey_KET_15T")


def test_cx_smiles(example_smiles, reference_hashes):
    identifier = CxSmiles()
    calculated = {}
    for x in example_smiles:
        rdmol = cif.Chem.MolFromSmiles(x.get("smiles"))
        hash_key = identifier.get_identifier(rdmol=rdmol)
        name = x.get("name")
        calculated[name] = hash_key

    for name, v in calculated.items():
        assert v == reference_hashes.get(name).get("cx_smiles")
        if name == "es1":
            assert v != reference_hashes.get(name).get("smiles")


def test_canonical_smiles(example_smiles, reference_hashes):
    identifier = CanonicalSmiles()
    calculated = {}
    for x in example_smiles:
        rdmol = cif.Chem.MolFromSmiles(x.get("smiles"))
        if rdmol is None:
            print("something wrong")
            print(x)
        hash_key = identifier.get_identifier(rdmol=rdmol)
        name = x.get("name")
        calculated[name] = hash_key
    for name, v in calculated.items():
        assert v == reference_hashes.get(name).get("smiles")
        assert v == reference_hashes.get(name).get("CanonicalSmiles")


def test_all_mol_identifiers_are_listed():
    identifiers = get_all_molecular_identifiers()
    assert "smiles" in identifiers["factory"]
    assert "inchi_key" in identifiers["factory"]
    assert "MurckoScaffold" in identifiers["rdkit"]


def test_mol_identifier_is_valid():
    assert is_valid_molecular_identifier("smiles") is True
    assert is_valid_molecular_identifier("r_p") is False


def test_mol_identifier_validation():
    validate_molecular_identifier("cx_smiles")
    with pytest.raises(UnavailableMolIdentifier):
        validate_molecular_identifier("something_wrong")


def test_reaction_representation_factory():
    representation_list = ReactionIdentifiersFactory.list_reaction_identifiers()
    assert "r_p" in representation_list
    assert "u_r_r_p" in representation_list


def test_reaction_presentations_r_p(molecules_map):
    r_p = ReactantProduct().get_reaction_identifier(molecules_map)
    assert r_p == "CC(=O)O.CN>>CNC(C)=O"


def test_reaction_presentations_r_r_p(molecules_map):
    r_r_p = ReactantReagentProduct().get_reaction_identifier(molecules_map)
    assert r_r_p == "CC(=O)O.CN>O>CNC(C)=O"


def test_reaction_presentations_u_r_p(molecules_map):
    u_r_p = UnorderedReactantProduct().get_reaction_identifier(molecules_map)
    assert u_r_p == "CC(=O)O.CN>>CNC(C)=O"


def test_reaction_presentations_u_r_r_p(molecules_map):
    u_r_r_p = UnorderedReactantReagentProduct().get_reaction_identifier(molecules_map)
    assert u_r_r_p == "CC(=O)O.CN>>CNC(C)=O>>O"


def test_all_reaction_identifiers_are_listed():
    identifiers = get_all_reaction_identifiers()
    assert "r_p" in identifiers
    assert "u_r_p" in identifiers


def test_reaction_identifier_is_valid():
    assert is_valid_reaction_identifier("r_r_p") is True
    assert is_valid_reaction_identifier("smiles") is False


def test_reaction_identifier_validation():
    validate_reaction_identifier("u_r_r_p")
    with pytest.raises(UnavailableReactionIdentifier):
        validate_reaction_identifier("something_wrong")


def test_calculate_molecular_hash_map(example_smiles):
    hash_list = {
        "ExtendedMurcko",
        "Regioisomer",
        "cx_smiles",
        "inchi",
        "inchi_key",
        "inchikey_KET_15T",
        "smiles",
    }
    rdmol = cif.Chem.MolFromSmiles(example_smiles[0]["smiles"])
    hash_map = calculate_molecular_hash_map(rdmol=rdmol, hash_list=hash_list)
    assert hash_list == set(hash_map.keys())

    hash_list.add("wrong_identifier")
    with unittest.TestCase().assertLogs(
        "linchemin.cheminfo.chemical_hashes", level="WARNING"
    ):
        hash_map = calculate_molecular_hash_map(rdmol=rdmol, hash_list=hash_list)
    assert "wrong_identifier" not in hash_map
