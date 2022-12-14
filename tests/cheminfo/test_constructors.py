from linchemin.cheminfo.constructors import (MoleculeConstructor, ChemicalEquationConstructor, RatamConstructor,
                                             BadMapping, PatternConstructor, TemplateConstructor,
                                             DisconnectionConstructor)
from linchemin.cheminfo.models import Template
from linchemin.utilities import create_hash
import linchemin.cheminfo.functions as cif

import pytest


def test_molecule_equality():
    mols = {
        0: {'smiles': 'CN'},  # M1
        1: {'smiles': 'CN'},  # M1
        2: {'smiles': 'NC'},  # M1
        3: {'smiles': 'CC'},  # M2
        4: {'smiles': 'CC(C)=O'},  # M3_T1
        5: {'smiles': 'CC(O)=C'},  # M3_T2
        6: {'smiles': 'CC(O)=N'},  # M4_T1
        7: {'smiles': 'CC(N)=O'},  # M4_T2
        8: {'smiles': 'CCC(C)=O'},  # M5_T1
        9: {'smiles': r'C\C=C(\C)O'},  # M5_T2
        10: {'smiles': 'Cl[C:2]([CH3:1])=[O:3]'},  # M6_atom_mapping_1
        11: {'smiles': 'Cl[C:1]([CH3:2])=[O:5]'},  # M6_atom_mapping_2

    }

    # initialize the constructor to use smiles as identity property
    molecule_constructor = MoleculeConstructor(identity_property_name='smiles')

    # using smiles
    ms1 = {k: molecule_constructor.build_from_molecule_string(molecule_string=v.get('smiles'), inp_fmt='smiles')
           for k, v in mols.items()}

    assert ms1.get(0) == ms1.get(1)  # identical molecule, identical input string
    assert ms1.get(0) == ms1.get(2)  # identical molecule, different input string; assess the canonicalization mechanism
    assert ms1.get(0) != ms1.get(3)  # different molecules
    assert ms1.get(4) != ms1.get(5)  # same molecule, but different tautomers: smiles fails to capture identity
    assert ms1.get(6) != ms1.get(7)  # same molecule, but different tautomers: smiles fails to capture identity
    assert ms1.get(8) != ms1.get(9)  # same molecule, but different tautomers: smiles fails to capture identity
    assert ms1.get(10) == ms1.get(11)  # same molecule, but different atom mapping

    # initialize the constructor to use inchi_key as identity property
    molecule_constructor = MoleculeConstructor(identity_property_name='inchi_key')
    ms2 = {k: molecule_constructor.build_from_molecule_string(molecule_string=v.get('smiles'), inp_fmt='smiles')
           for k, v in mols.items()}
    assert ms2.get(0) == ms2.get(1)  # identical molecule, identical input string
    assert ms2.get(0) == ms2.get(2)  # identical molecule, different input string; assess the canonicalization mechanism
    assert ms2.get(0) != ms2.get(3)  # different molecules
    # assert ms2.get(4) == ms2.get(5)  # same molecule, but different tautomers: inchi_key succeeds to capture identity # TODO: it does not work inchi are different!!!!!
    assert ms2.get(6) == ms2.get(7)  # same molecule, but different tautomers: inchi_key succeeds to capture identity
    assert ms2.get(10) == ms2.get(11)  # same molecule, but different atom mapping


def test_chemical_equation_hashing():
    reactions = {0: {'smiles': 'CN.CC(O)=O>O>CNC(C)=O'},
                 1: {'smiles': 'CC(O)=O.CN>O>CNC(C)=O'},
                 2: {'smiles': '>>CNC(C)=O'},
                 3: {'smiles': 'CC(O)=O.CN>>CNC(C)=O'},
                 4: {'smiles': 'CC(O)=O.CN>>'},
                 5: {'smiles': 'CN.CC(O)=O>>CNC(C)=O'},
                 6: {'smiles': 'CNC(C)=O>>CN.CC(O)=O'},
                 7: {'smiles': 'CN.CC(O)=O>>CNC(C)=O.O'},
                 8: {'smiles': 'CN.CC(O)=O>>O.CNC(C)=O'},
                 9: {'smiles': 'CNC(C)=O>O>CN.CC(O)=O'}
                 }
    # initialize the constructor
    chemical_equation_constructor = ChemicalEquationConstructor(identity_property_name='smiles')
    results = {}
    for k, v in reactions.items():
        chemical_equation = chemical_equation_constructor.build_from_reaction_string(
            reaction_string=v.get('smiles'),
            inp_fmt='smiles')

        h = chemical_equation.hash_map
        results[k] = h
        # print(k, h)

    # the hashes are calculated and have a non-null (None) value
    assert results.get(0).get('reactants')
    assert results.get(0).get('reagents')
    assert results.get(0).get('products')
    assert results.get(0).get('r_p')
    assert results.get(0).get('r_r_p')
    assert results.get(0).get('u_r_p')
    assert results.get(0).get('u_r_r_p')

    # the reactant hash is insensitive to the input order of reactants (reaction canonicalization OK)
    assert results.get(0).get('reactants') == results.get(1).get('reactants')

    # the product hash is insensitive to the input order of products (reaction canonicalization OK)
    assert results.get(7).get('products') == results.get(8).get('products')

    # the machinery does not break when the reactants are missing
    assert results.get(2).get('reactants')

    # the machinery does not break when the agents are missing
    assert results.get(3).get('reagents')

    # the machinery does not break when the products are missing
    assert results.get(4).get('products')

    # there is a special hash for missing roles (it is the hash of an empty string)
    assert results.get(2).get('reactants') == results.get(3).get('reagents') == results.get(4).get(
        'products') == create_hash('')

    # the reactant and products hashes are conserved even when the reagents are missing
    assert results.get(0).get('reactants') == results.get(5).get('reactants')
    assert results.get(0).get('products') == results.get(5).get('products')

    # the agent hash is different if the agents are missing
    assert results.get(0).get('reagents') != results.get(5).get('reagents')

    # the base r>p hash is conserved if the agents are missing in one reaction
    assert results.get(0).get('r_p') == results.get(5).get('r_p')

    # the full r>a>p hash is not conserved if the reagents are missing in one reaction
    assert results.get(0).get('r_r_p') != results.get(5).get('r_r_p')

    # the base r>>p hash is not conserved if the reaction is reversed
    assert results.get(0).get('r_p') != results.get(6).get('r_p')

    # the full r>a>p hash is not conserved if the reaction is reversed
    assert results.get(0).get('r_r_p') != results.get(6).get('r_r_p')

    # the reversible base r<>p hash is  conserved if the reaction is reversed
    assert results.get(0).get('u_r_p') == results.get(9).get('u_r_p')
    assert results.get(3).get('u_r_p') == results.get(6).get('u_r_p')

    # the reversible full r<a>p hash is  conserved if the reaction is reversed
    assert results.get(0).get('u_r_r_p') == results.get(9).get('u_r_r_p')
    assert results.get(3).get('u_r_r_p') == results.get(6).get('u_r_r_p')


def test_instantiate_chemical_equation():
    reaction_smiles_input = 'NC.CC(O)=O>O>CNC(C)=O'
    chemical_equation_constructor = ChemicalEquationConstructor(identity_property_name='smiles')
    chemical_equation = chemical_equation_constructor.build_from_reaction_string(reaction_string=reaction_smiles_input,
                                                                                 inp_fmt='smiles')
    assert chemical_equation
    # assert molecules are canonicalized
    # assert reaction is canonicalized


def test_create_reaction_smiles_from_chemical_equation():
    reaction_smiles_input = 'CN.CC(O)=O>O>CNC(C)=O'
    chemical_equation_constructor = ChemicalEquationConstructor(identity_property_name='smiles')
    chemical_equation = chemical_equation_constructor.build_from_reaction_string(reaction_string=reaction_smiles_input,
                                                                                 inp_fmt='smiles')
    reaction_smiles = chemical_equation.build_reaction_smiles()
    assert reaction_smiles


def test_reaction_canonicalization_from_molecules():
    reactions = {0: {'smiles': 'CN.CC(O)=O>O>CNC(C)=O'},
                 1: {'smiles': 'CC(O)=O.CN>O>CNC(C)=O'},
                 2: {'smiles': '>>CNC(C)=O'},
                 3: {'smiles': 'CC(O)=O.CN>>CNC(C)=O'},
                 4: {'smiles': 'CC(O)=O.CN>>'},
                 5: {'smiles': 'CN.CC(O)=O>>CNC(C)=O'},
                 6: {'smiles': 'CNC(C)=O>>CN.CC(O)=O'},
                 7: {'smiles': 'CN.CC(O)=O>>CNC(C)=O.O'},
                 8: {'smiles': 'CN.CC(O)=O>>O.CNC(C)=O'},
                 9: {'smiles': 'CNC(C)=O>O>CN.CC(O)=O'}
                 }
    # initialize the constructor
    chemical_equation_constructor = ChemicalEquationConstructor(identity_property_name='smiles')
    results = {}
    for k, v in reactions.items():
        chemical_equation = chemical_equation_constructor.build_from_reaction_string(reaction_string=v.get('smiles'),
                                                                                     inp_fmt='smiles')
        results[k] = chemical_equation
        # print(k, h)
    # the reaction smiles is insensitive to the input order of reactants (reaction canonicalization OK)

    assert results.get(0).smiles == results.get(1).smiles


def test_chemical_equation_equality():
    reactions = {0: {'smiles': 'CN.CC(O)=O>O>CNC(C)=O'},  # R1
                 1: {'smiles': 'CN.CC(O)=O>O>CNC(C)=O'},  # R1
                 2: {'smiles': 'NC.CC(O)=O>O>CNC(C)=O'},  # R1
                 3: {'smiles': 'NC.CC(=O)O>O>CNC(C)=O'},  # R1
                 4: {'smiles': 'CC(O)=O.CN>O>CNC(C)=O'},  # R1
                 5: {'smiles': 'CC#N.Cl[C:2]([CH3:1])=[O:3].[CH3:4][NH2:5]>>O.[CH3:1][C:2](=[O:3])[NH:5][CH3:4]'},  # R2
                 6: {'smiles': 'CC#N.Cl[C:20]([CH3:1])=[O:3].[CH3:4][NH2:50]>>O.[CH3:1][C:20](=[O:3])[NH:50][CH3:4]'},
                 # R2
                 7: {'smiles': 'Cl[C:20]([CH3:1])=[O:3].CC#N.[CH3:4][NH2:50]>>O.[CH3:1][C:20](=[O:3])[NH:50][CH3:4]'},
                 # R2
                 }

    ces1 = {}

    # initialize the constructor
    chemical_equation_constructor = ChemicalEquationConstructor(identity_property_name='smiles')

    for k, v in reactions.items():
        chemical_equation = chemical_equation_constructor.build_from_reaction_string(reaction_string=v.get('smiles'),
                                                                                     inp_fmt='smiles')
        ces1[k] = chemical_equation

    assert ces1.get(0) == ces1.get(1)  # same reaction, one reactant has a different smiles: test mol canonicalization
    assert ces1.get(0) == ces1.get(2)  # same reaction, two reactant have a different smiles: test mol canonicalization
    assert ces1.get(0) == ces1.get(3)  # same reaction, two reactant have a different smiles: test mol canonicalization
    assert ces1.get(0) == ces1.get(4)  # same reaction, different reactant ordering: test reaction canonicalization
    assert ces1.get(5) == ces1.get(6)  # same reaction, different atom mapping
    assert ces1.get(5) == ces1.get(7)  # same reaction, different atom mapping,
    # different reactant ordering: test reaction canonicalization


def test_chemical_equation_builder():
    reaction_string_reference = 'CC(=O)O.CN.CN>O>CNC(C)=O'

    # initialize the constructor
    cec = ChemicalEquationConstructor(identity_property_name='smiles')

    for reaction_string_test in [
        'CC(=O)O.CN.CN>O>CNC(C)=O',  # expected smiles
        'CC(=O)O.NC.CN>O>CNC(C)=O',  # test molecule canonicalization: change order of atoms in reactant molecules
        'CN.CC(=O)O.CN>O>CNC(C)=O',  # test reaction canonicalization: change order of molecules in reactants
    ]:
        chemical_equation = cec.build_from_reaction_string(reaction_string=reaction_string_test, inp_fmt='smiles')
        reaction_string_calculated = chemical_equation.smiles
        assert reaction_string_calculated == reaction_string_reference


def test_chemical_equation_attributes_are_not_available():
    smiles = 'CN.CC(O)=O>O>CNC(C)=O'
    chemical_equation_constructor = ChemicalEquationConstructor(identity_property_name='smiles')
    chemical_equation = chemical_equation_constructor.build_from_reaction_string(reaction_string=smiles,
                                                                                 inp_fmt='smiles')
    disconnection = chemical_equation.disconnection
    assert not disconnection
    template = chemical_equation.template
    assert not template


def test_ratam_constructor():
    test_set = [
        {'name': 'rnx_1',
         'smiles': '[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]',
         'expected': {'reactants': ['CC(=O)O', 'CN'],
                      'reagents': [],
                      'products': ['CNC(C)=O', 'O']}},
        {'name': 'rnx_2',
         'smiles': '[CH3:1][C:2]([OH:3])=[O:4]>[CH3:6][NH2:5]>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]',
         'expected': {'reactants': ['CC(=O)O', 'CN'],
                      'reagents': [],
                      'products': ['CNC(C)=O', 'O']}},
        {'name': 'rnx_3',
         'smiles': '[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>CN>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]',
         'expected': {'reactants': ['CC(=O)O', 'CN'],
                      'reagents': ['CN'],
                      'products': ['CNC(C)=O', 'O']}},
        {'name': 'rnx_4',
         'smiles': '[CH3:1][C:2]([OH:1])=[O:4].[CH3:6][NH2:5]>CN>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:1]',
         'expected': {'reactants': ['CC(=O)O', 'CN'],
                      'reagents': ['CN'],
                      'products': ['CNC(C)=O', 'O']}},
    ]
    mol_constructor = MoleculeConstructor(identity_property_name='smiles')
    for item in test_set:
        rdrxn = cif.rdrxn_from_string(input_string=item.get('smiles'), inp_fmt='smiles')
        rdmol_catalog = {'reactants': list(rdrxn.GetReactants()),
                         'reagents': list(rdrxn.GetAgents()),
                         'products': list(rdrxn.GetProducts())}
        reactants_reagents = [mol_constructor.build_from_rdmol(rdmol=rdmol) for role, rdmol_list in
                              rdmol_catalog.items()
                              if role in ['reactants', 'reagents'] for rdmol in rdmol_list]
        products = [mol_constructor.build_from_rdmol(rdmol=rdmol) for role, rdmol_list in rdmol_catalog.items()
                    if role == 'products' for rdmol in rdmol_list]
        reaction_mols = {'reactants_reagents': reactants_reagents, 'products': products}
        catalog = {}
        for role, rdmol_list in rdmol_catalog.items():
            list_tmp = [mol_constructor.build_from_rdmol(rdmol=rdmol) for rdmol in rdmol_list]
            set_tmp = set(list_tmp)
            _tmp = {m.uid: m for m in set_tmp}
            catalog = {**catalog, **_tmp}
        if item['name'] == 'rnx_4':
            with pytest.raises(BadMapping) as ke:
                ratam_constructor = RatamConstructor()
                ratam_constructor.create_ratam(reaction_mols)
            assert "BadMapping" in str(ke.type)
        else:
            ratam_constructor = RatamConstructor()
            cem = ratam_constructor.create_ratam(reaction_mols)
            assert cem
            assert cem.atom_transformations
            map_numbers = set()
            for k, v in cem.full_map_info.items():
                map_numbers.update(m for d in v for m in d.values() if m not in [0, -1])

            # check if an AtomTransformation exists for each map number
            assert len(map_numbers) == len(cem.atom_transformations)

            new_roles = cif.new_role_reassignment(reaction_mols, cem, desired_product=products[0])
            for role, mols in new_roles.items():
                smiles_list = [m.smiles for uid, m in catalog.items() if uid in mols]
                assert item['expected'][role] == smiles_list


def test_pattern_creation():
    test_set = [
        {'name': 'pattern_1',
         'smarts': '[NH2;D1;+0:4]-[C;H0;D3;+0:1](=[O;D1;H0:2])-[c:3]',
         'expected': {},
         },
        {'name': 'pattern_2',
         'smarts': '[CH3:6][NH:5][C:2]([CH3:1])=[O:4]',
         'expected': {},
         },
        {'name': 'pattern_3',
         'smarts': '[CH3:1][c:2]1[cH:3][cH:4][cH:5][cH:6][n:7]1',
         'expected': {},
         },
    ]

    pc = PatternConstructor()
    for item in test_set:
        pattern = pc.build_from_molecule_string(molecule_string=item.get('smarts'), inp_fmt='smarts')
        print(f"\n{item.get('name')} {pattern.to_dict()}")
        assert pattern


def test_template_creation():
    print("IMPLEMENT: test_template_creation")
    test_set = [
        {'name': 'rnx_1',
         'smiles': '[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]',
         'expected': {},
         },
        {'name': 'rnx_2',
         'smiles': '[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]>>[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]',
         'expected': {},
         }
    ]

    for item in test_set:
        tc = TemplateConstructor()
        template = tc.build_from_reaction_string(reaction_string=item.get('smiles'), inp_fmt='smiles')
        assert isinstance(template, Template)


def test_template_hashing():
    reactions = {0: {'smiles': '[CH3:1][NH2:2].O[C:3]([CH3:4])=[O:5]>O>[CH3:1][NH:2][C:3]([CH3:4])=[O:5]'},
                 1: {'smiles': 'O[C:3]([CH3:4])=[O:5].[CH3:1][NH2:2]>O>[CH3:1][NH:2][C:3]([CH3:4])=[O:5]'},
                 2: {'smiles': '>>[CH3:1][NH:2][C:3]([CH3:4])=[O:5]'},
                 3: {'smiles': '[CH3:1][NH2:2].O[C:3]([CH3:4])=[O:5]>>[CH3:1][NH:2][C:3]([CH3:4])=[O:5]'},
                 4: {'smiles': '[CH3:1][NH2:2].O[C:3]([CH3:4])=[O:5]>>'},
                 5: {'smiles': '[CH3:1][NH2:2].O[C:3]([CH3:4])=[O:5]>>[CH3:1][NH:2][C:3]([CH3:4])=[O:5]'},
                 6: {'smiles': '[CH3:1][NH:2][C:3]([CH3:4])=[O:5]>>[CH3:1][NH2:2].O[C:3]([CH3:4])=[O:5]'},
                 7: {
                     'smiles': '[CH3:1][NH2:2].[C:3]([CH3:4])(=[O:5])[OH:6]>>[CH3:1][NH:2][C:3]([CH3:4])=[O:5].[OH2:6]'},
                 8: {
                     'smiles': '[C:3]([CH3:4])(=[O:5])[OH:6].[CH3:1][NH2:2]>>[OH2:6].[CH3:1][NH:2][C:3]([CH3:4])=[O:5]'},
                 9: {'smiles': '[CH3:1][NH:2][C:3]([CH3:4])=[O:5]>O>[CH3:1][NH2:2].O[C:3]([CH3:4])=[O:5]'},
                 }
    # initialize the constructor
    template_constructor = TemplateConstructor(identity_property_name='smarts')
    results = {}
    for k, v in reactions.items():
        if template := template_constructor.build_from_reaction_string(reaction_string=v.get('smiles'),
                                                                       inp_fmt='smiles'):
            h = template.hash_map
        else:
            h = None
        results[k] = h
        # print(k, h)

    # the hashes are calculated and have a non-null (None) value
    assert results.get(0).get('reactants')
    assert results.get(0).get('reagents')
    assert results.get(0).get('products')
    assert results.get(0).get('r_p')
    assert results.get(0).get('r_r_p')
    assert results.get(0).get('u_r_p')
    assert results.get(0).get('u_r_r_p')

    # the reactant hash is insensitive to the input order of reactants (reaction canonicalization OK)
    assert results.get(0).get('reactants') == results.get(1).get('reactants')

    # the product hash is insensitive to the input order of products (reaction canonicalization OK)
    assert results.get(7).get('products') == results.get(8).get('products')

    # the machinery does break when the reactants are missing: Template is None
    assert results.get(2) is None

    # the machinery does not break when the agents are missing
    assert results.get(3).get('reagents')

    # the machinery does break when the products are missing: Template is None
    assert results.get(4) is None

    # reagents are happily ignored

    # there is a special hash for missing roles (it is the hash of an empty string)
    assert results.get(3).get('reagents') == create_hash('')

    # the reactant and products hashes are conserved even when the reagents are missing
    assert results.get(0).get('reactants') == results.get(5).get('reactants')
    assert results.get(0).get('products') == results.get(5).get('products')

    # the base r>p hash is conserved if the agents are missing in one reaction
    assert results.get(0).get('r_p') == results.get(5).get('r_p')

    # the full r>a>p hash is conserved  if the reagents are missing in one reaction (reagents are ignored!!)
    assert results.get(0).get('r_r_p') == results.get(5).get('r_r_p')

    # the base r>>p hash is not conserved if the reaction is reversed
    assert results.get(0).get('r_p') != results.get(6).get('r_p')

    # the full r>a>p hash is not conserved if the reaction is reversed
    assert results.get(0).get('r_r_p') != results.get(6).get('r_r_p')

    # the reversible base r<>p hash is not conserved if the reaction is reversed (this comes from rdchiral teplate extraction)
    # in some special cases it might be true, but it not necessarily is
    assert results.get(0).get('u_r_p') != results.get(9).get('u_r_p')
    assert results.get(3).get('u_r_p') != results.get(6).get('u_r_p')

    # the reversible full r<a>p hash is not conserved if the reaction is reversed (this comes from rdchiral teplate extraction)
    # in some special cases it might be true, but it not necessarily is
    assert results.get(0).get('u_r_r_p') != results.get(9).get('u_r_r_p')
    assert results.get(3).get('u_r_r_p') != results.get(6).get('u_r_r_p')


def test_disconnection_equality():
    test_set = [
        {'name': 'rnx_1',  # fully balanced amide formation from carboxylic acid and amine
         'smiles': '[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]',
         'expected': {},
         },
        {'name': 'rnx_2',  # fully balanced amide hydrolysis
         'smiles': '[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]>>[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]',
         'expected': {},
         },
        {'name': 'rnx_3',
         # fully balanced intramolecular michael addition ring forming, one new bond and one changend bond
         'smiles': '[CH3:1][CH2:2][C:3](=[O:4])[c:5]1[cH:6][cH:7][cH:8][n:9]1[CH2:10]/[CH:11]=[CH:12]\[C:13](=[O:14])[O:15][CH3:16]>>[CH3:1][CH:2]1[CH:11]([CH2:10][n:9]2[cH:8][cH:7][cH:6][c:5]2[C:3]1=[O:4])[CH2:12][C:13](=[O:14])[O:15][CH3:16]',
         'expected': {},
         },
        {'name': 'rnx_4',  # fully balanced diels-alder product regioisomer 1
         'smiles': '[CH3:6][CH:7]=[CH2:8].[CH3:1][C:2](=[CH2:3])[CH:4]=[CH2:5]>>[CH3:6][CH:7]1[CH2:8][CH2:3][C:2](=[CH:4][CH2:5]1)[CH3:1] ',
         'expected': {},
         },
        {'name': 'rnx_5',  # fully balanced diels-alder product regioisomer 2
         'smiles': '[CH3:6][CH:7]=[CH2:8].[CH3:1][C:2](=[CH2:3])[CH:4]=[CH2:5]>>[CH3:6][CH:7]1[CH2:8][CH2:5][CH:4]=[C:2]([CH2:3]1)[CH3:1]',
         'expected': {},
         },
        {'name': 'rnx_6',  # fully balanced amide formation from acyl chloride and amine (same disconnection as rnx_1)
         'smiles': '[CH3:1][C:2]([Cl:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[Cl:3][H]',
         'expected': {},
         },
        {'name': 'rnx_7',  # not fully balanced reaction
         'smiles': '[cH:5]1[cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]2[c:3]([cH:4]1)[C:2](=[O:1])O.[N-:13]=[N+:14]=[N-:15]>C(Cl)Cl.C(=O)(C(=O)Cl)Cl>[cH:5]1[cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]2[c:3]([cH:4]1)[C:2](=[O:1])[N:13]=[N+:14]=[N-:15]',
         'expected': {},
         },

    ]

    dc = DisconnectionConstructor(identity_property_name='smiles')

    results = {item.get('name'): dc.build_from_reaction_string(reaction_string=item.get('smiles'), inp_fmt='smiles')
               for item in test_set}

    # regioisomer products from the same reactants: disconnection is different (fragments might be the same)
    assert results.get('rnx_4') != results.get('rnx_5')

    # same product from two sets of equivalent reactants (at synthol level)
    assert results.get('rnx_1') == results.get('rnx_6')
