from rdkit import Chem
from rdkit.Chem import Draw
# from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdChemReactions
import rdkit
from collections import namedtuple
import pprint

import linchemin.cheminfo.functions as cif
import linchemin.utilities as utilities
from linchemin.cheminfo.molecule import Molecule

print(rdkit.__version__)
logger = utilities.console_logger(__name__)

rxn1 = rdChemReactions.ReactionFromRxnBlock('''$RXN

      Mrv2102  111820212128

  2  1
$MOL

  Mrv2102 11182121282D          

 13 13  0  0  0  0            999 V2000
   -7.5723    2.6505    0.0000 C   0  0  0  0  0  0  0  0  0  1  0  0
   -6.8579    2.2380    0.0000 O   0  0  0  0  0  0  0  0  0  2  0  0
   -6.8580    1.4130    0.0000 C   0  0  0  0  0  0  0  0  0  3  0  0
   -6.1435    1.0004    0.0000 O   0  0  0  0  0  0  0  0  0  4  0  0
   -7.5725    1.0005    0.0000 C   0  0  0  0  0  0  0  0  0  5  0  0
   -7.5725    0.1755    0.0000 N   0  0  0  0  0  0  0  0  0  6  0  0
   -8.2869   -0.2369    0.0000 C   0  0  0  0  0  0  0  0  0  7  0  0
   -8.2870   -1.0620    0.0000 C   0  0  0  0  0  0  0  0  0  8  0  0
   -9.0015   -1.4745    0.0000 C   0  0  0  0  0  0  0  0  0  9  0  0
   -9.0015   -2.2995    0.0000 C   0  0  0  0  0  0  0  0  0 10  0  0
   -8.2870   -2.7120    0.0000 C   0  0  0  0  0  0  0  0  0 11  0  0
   -7.5726   -2.2995    0.0000 C   0  0  0  0  0  0  0  0  0 12  0  0
   -7.5726   -1.4745    0.0000 C   0  0  0  0  0  0  0  0  0 13  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
  3  4  2  0  0  0  0
  3  5  1  0  0  0  0
  5  6  1  0  0  0  0
  6  7  2  0  0  0  0
  7  8  1  0  0  0  0
  8  9  1  0  0  0  0
  8 13  2  0  0  0  0
  9 10  2  0  0  0  0
 10 11  1  0  0  0  0
 11 12  2  0  0  0  0
 12 13  1  0  0  0  0
M  END
$MOL

  Mrv2102 11182121282D          

 12 11  0  0  0  0            999 V2000
   -3.7934    0.7703    0.0000 C   0  0  0  0  0  0  0  0  0 14  0  0
   -3.0790    1.1828    0.0000 C   0  0  0  0  0  0  0  0  0 15  0  0
   -2.3645    0.7703    0.0000 C   0  0  0  0  0  0  0  0  0 16  0  0
   -3.7934   -0.0547    0.0000 C   0  0  0  0  0  0  0  0  0 17  0  0
   -4.5078   -0.4672    0.0000 O   0  0  0  0  0  0  0  0  0 18  0  0
   -3.0789   -0.4671    0.0000 O   0  0  0  0  0  0  0  0  0 19  0  0
   -1.6500    1.1828    0.0000 O   0  0  0  0  0  0  0  0  0 20  0  0
   -2.3645   -0.0547    0.0000 O   0  0  0  0  0  0  0  0  0 21  0  0
   -3.0788   -1.2922    0.0000 C   0  0  0  0  0  0  0  0  0 22  0  0
   -1.6500   -0.4672    0.0000 C   0  0  0  0  0  0  0  0  0 23  0  0
   -2.3644   -1.7046    0.0000 C   0  0  0  0  0  0  0  0  0 24  0  0
   -1.6500   -1.2922    0.0000 C   0  0  0  0  0  0  0  0  0 25  0  0
  1  2  2  0  0  0  0
  1  4  1  0  0  0  0
  2  3  1  0  0  0  0
  3  7  2  0  0  0  0
  3  8  1  0  0  0  0
  4  5  2  0  0  0  0
  4  6  1  0  0  0  0
  6  9  1  0  0  0  0
  8 10  1  0  0  0  0
  9 11  1  0  0  0  0
 10 12  1  0  0  0  0
M  END
$MOL

  Mrv2102 11182121282D          

 25 26  0  0  0  0            999 V2000
    5.1328    0.9532    0.0000 C   0  0  0  0  0  0  0  0  0  5  0  0
    5.8002    0.4683    0.0000 N   0  0  0  0  0  0  0  0  0  6  0  0
    5.5453   -0.3163    0.0000 C   0  0  0  0  0  0  0  0  0  7  0  0
    4.7203   -0.3163    0.0000 C   0  0  0  0  0  0  0  0  0 14  0  0
    4.4654    0.4683    0.0000 C   0  0  0  0  0  0  0  0  0 15  0  0
    5.1328    1.7782    0.0000 C   0  0  0  0  0  0  0  0  0  3  0  0
    3.6807    0.7232    0.0000 C   0  0  0  0  0  0  0  0  0 16  0  0
    4.2354   -0.9838    0.0000 C   0  0  0  0  0  0  0  0  0 17  0  0
    6.0302   -0.9838    0.0000 C   0  0  0  0  0  0  0  0  0  8  0  0
    6.8507   -0.8975    0.0000 C   0  0  0  0  0  0  0  0  0  9  0  0
    7.3356   -1.5650    0.0000 C   0  0  0  0  0  0  0  0  0 10  0  0
    7.0001   -2.3187    0.0000 C   0  0  0  0  0  0  0  0  0 11  0  0
    6.1796   -2.4049    0.0000 C   0  0  0  0  0  0  0  0  0 12  0  0
    5.6947   -1.7375    0.0000 C   0  0  0  0  0  0  0  0  0 13  0  0
    3.4149   -0.8975    0.0000 O   0  0  0  0  0  0  0  0  0 18  0  0
    4.5709   -1.7375    0.0000 O   0  0  0  0  0  0  0  0  0 19  0  0
    4.0860   -2.4049    0.0000 C   0  0  0  0  0  0  0  0  0 22  0  0
    3.2655   -2.3187    0.0000 C   0  0  0  0  0  0  0  0  0 24  0  0
    3.5092    1.5302    0.0000 O   0  0  0  0  0  0  0  0  0 20  0  0
    3.0676    0.1712    0.0000 O   0  0  0  0  0  0  0  0  0 21  0  0
    2.2830    0.4261    0.0000 C   0  0  0  0  0  0  0  0  0 23  0  0
    1.6699   -0.1259    0.0000 C   0  0  0  0  0  0  0  0  0 25  0  0
    5.8473    2.1907    0.0000 O   0  0  0  0  0  0  0  0  0  4  0  0
    4.4183    2.1907    0.0000 O   0  0  0  0  0  0  0  0  0  2  0  0
    4.4183    3.0157    0.0000 C   0  0  0  0  0  0  0  0  0  1  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
  3  4  1  0  0  0  0
  4  5  1  0  0  0  0
  1  5  1  0  0  0  0
  1  6  1  0  0  0  0
  5  7  1  0  0  0  0
  4  8  1  0  0  0  0
  3  9  1  0  0  0  0
 10 11  2  0  0  0  0
 11 12  1  0  0  0  0
 12 13  2  0  0  0  0
 13 14  1  0  0  0  0
  9 10  1  0  0  0  0
  9 14  2  0  0  0  0
  8 15  2  0  0  0  0
  8 16  1  0  0  0  0
 16 17  1  0  0  0  0
 17 18  1  0  0  0  0
  7 19  2  0  0  0  0
  7 20  1  0  0  0  0
 20 21  1  0  0  0  0
 21 22  1  0  0  0  0
  6 23  2  0  0  0  0
  6 24  1  0  0  0  0
 24 25  1  0  0  0  0
M  END
''')

AtomTransformation = namedtuple('AtomTransformation', ['product_uid', 'reactant_uid', 'prod_atom_id', 'react_atom_id',
                                                       'map_num'])


class ChemicalEquationMapping:
    """ Class to store ChemicalEquation atom mapping information.

        Attributes:
            full_map_info: a dictionary in the form {mol_uid: {atom_idx: map_num'}} containing the full
                           information about the ChemicalEquation atom mapping

            atom_transformations: a list of AtomTransformation namedtuples
    """

    def __init__(self, reaction_mols: dict):
        self.atom_transformations = None
        self.full_map_info = {}
        for m in reaction_mols['reactants_reagents'] + reaction_mols['products']:
            self.full_map_info[m.uid] = {}
            for a in m.rdmol_mapped.GetAtoms():
                if isinstance(a.GetAtomMapNum(), int):
                    self.full_map_info[m.uid][a.GetIdx()] = a.GetAtomMapNum()
                else:
                    self.full_map_info[m.uid][a.GetIdx()] = -1

        self.get_atom_transformations(reaction_mols)

    def get_atom_transformations(self, reaction_mols: dict):
        """ To create the list of AtomTransformations from a catalog of mapped Molecule objects """
        atom_transformations = []
        for product in reaction_mols['products']:
            prod_map = self.full_map_info[product.uid]
            for reactant in reaction_mols['reactants_reagents']:
                reactant_map = self.full_map_info[reactant.uid]
                if matching_map_num := [map_num for map_num in reactant_map.values() if map_num in prod_map.values()
                                                                                        and map_num not in [0, -1]]:
                    atom_transformations.extend(build_atom_transformations(matching_map_num, prod_map,
                                                                           product.uid, reactant_map,
                                                                           reactant.uid))
        self.atom_transformations = atom_transformations


def build_atom_transformations(matching_map_num, prod_map, product_uid, reactant_map, reactant_uid):
    """ To build the list of AtomTransformation objects for each pair of product-reactant with matching map number"""
    ats = []
    for map_num in matching_map_num:
        p_aids = [aid for aid, map in prod_map.items() if map == map_num]
        r_aids = [aid for aid, map in reactant_map.items() if map == map_num]
        ats.extend([AtomTransformation(product_uid, reactant_uid, p_aid, r_aid, map_num)
                    for p_aid in p_aids
                    for r_aid in r_aids])
    return ats


def new_role_reassignment(reaction_mols: dict, cem: ChemicalEquationMapping, desired_product: Molecule):
    """ To reassign the roles of reactants and reagents based on the mapping on the desired product """
    if desired_product not in reaction_mols['products']:
        logger.error('The selected product is not among the reaction products.')
        return None
    desired_product_transformations = [at for at in cem.atom_transformations if at.product_uid == desired_product.uid]
    true_reactants_uid = {at.reactant_uid for at in desired_product_transformations}
    true_reagents = {r.uid for r in reaction_mols['reactants_reagents'] if r.uid not in true_reactants_uid}
    true_reactants = {r.uid for r in reaction_mols['reactants_reagents'] if r.uid in true_reactants_uid}
    products = [m.uid for m in reaction_mols['products']]
    return {'reactants': sorted(list(true_reactants)),
            'reagents': sorted(list(true_reagents)),
            'products': sorted(products)}


def map_atoms(rdmol_reaction_catalog):
    atom_map_d = []
    atom_tuple = namedtuple('atom_touple', ('role', 'role_molecule_idx', 'atom_idx', 'mapnum'))
    atom_map_t = []
    for role, rdmols in rdmol_reaction_catalog.items():
        for role_molecule_idx, reactant in enumerate(rdmols):
            for atom in reactant.GetAtoms():
                atom_idx = atom.GetIdx()
                mapnum = atom.GetAtomMapNum()
                print(role, role_molecule_idx, atom_idx, mapnum)
                data_point = {'role': role, 'role_molecule_idx': role_molecule_idx, 'atom_idx': atom_idx,
                              'mapnum': mapnum}
                atom_map_d.append(data_point)
                atom_map_t.append(atom_tuple(**data_point))
    return atom_map_d, atom_map_t


def role_reassignment(rdrxn: rdChemReactions.ChemicalReaction):
    """function to identify reactant molecules that should be reagent and address the problem"""

    rdmol_reaction_catalog = cif.rdrxn_to_reaction_rdmols(rdrxn)
    atom_map_d, atom_map_t = map_atoms(rdmol_reaction_catalog=rdmol_reaction_catalog)
    print(atom_map_d)
    print(atom_map_t)
    desired_product_idx = 0

    #  diagnosis
    atom_data_not_mapped = [item for item in atom_map_d if item.get('mapnum') == 0]
    atom_data_mapped = [item for item in atom_map_d if item.get('mapnum') != 0]

    output = utilities.list_of_dict_groupby(data_input=atom_data_mapped, keys=['role', 'role_molecule_idx'])
    # pprint.pprint(output)

    # for each molecule get the set of mapnos
    # group the mapno by role.molecule
    u = {k: set(utilities.list_of_dict_groupby(data_input=v, keys=['mapnum']).keys()) for k, v in output.items()}
    print(f'\nu\n {u}')

    # reactant and reagents that map into desired product (excluding unmapped atoms that have 0 mapno)
    ref = set(u.get(('products', desired_product_idx)))

    p = {(role, role_molecule_idx): (ref - v) for (role, role_molecule_idx), v in u.items() if
         role in ['reactants', 'reagents']}
    print(f'\np\n {p}')

    return


if __name__ == '__main__':
    rxn2 = rdChemReactions.ReactionFromSmarts(
        # '[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]',
        '[CH3:1][C:2]([OH:3])=[O:4]>C>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]',
        # '[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O].[OH2:3]',
        useSmiles=True)
    # dofun(rdrxn=rxn2)

    # diagnosis(rdrxn=rxn2)
