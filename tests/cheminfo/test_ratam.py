from linchemin.cheminfo.functions import rdrxn_from_string
from linchemin.cheminfo.ratam import ChemicalEquationMapping, new_role_reassignment
from linchemin.cheminfo.molecule import MoleculeConstructor


def test_chemical_equation_mapping():
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
         'smiles': '[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>CO>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]',
         'expected': {'reactants': ['CC(=O)O', 'CN'],
                      'reagents': ['CO'],
                      'products': ['CNC(C)=O', 'O']}},
    ]
    mol_constructor = MoleculeConstructor(identity_property_name='smiles')
    for item in test_set:
        rdrxn = rdrxn_from_string(input_string=item.get('smiles'), inp_fmt='smiles')
        rdmol_catalog = {'reactants': list(rdrxn.GetReactants()),
                         'reagents': list(rdrxn.GetAgents()),
                         'products': list(rdrxn.GetProducts())}
        reactants_reagents = [mol_constructor.build_from_rdmol(rdmol=rdmol) for role, rdmol_list in rdmol_catalog.items()
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
        cem = ChemicalEquationMapping(reaction_mols)
        assert cem
        assert cem.atom_transformations
        map_numbers = {m for k, v in cem.full_map_info.items() for m in v.values() if m not in [0, -1]}
        # check if an AtomTransformation exists for each map number
        assert len(map_numbers) == len(cem.atom_transformations)

        new_roles = new_role_reassignment(reaction_mols, cem, desired_product=products[0])
        for role, mols in new_roles.items():
            smiles_list = [m.smiles for uid, m in catalog.items() if uid in mols]
            assert item['expected'][role] == smiles_list

