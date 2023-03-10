import abc

from linchemin.cheminfo.models import ChemicalEquation, Molecule
import linchemin.cheminfo.functions as cif
from linchemin.utilities import console_logger

"""
Module containing functions and classes for computing score and metrics of single nodes of a route.
"""

logger = console_logger(__name__)


class NoMapping(Exception):
    """ Raised if the atom mapping is absent but needed to compute a descriptor """
    pass


class ChemicalEquationDescriptor(metaclass=abc.ABCMeta):
    """ Definition of the abstract class for NodeScore. """

    @abc.abstractmethod
    def compute_descriptor(self, reaction: ChemicalEquation):
        pass


class CEHypsicity(ChemicalEquationDescriptor):
    """ Subclass to compute the hypsicity of a ChemicalEquation """

    def compute_descriptor(self, reaction: ChemicalEquation) -> float:
        """ Takes a ChemicalEquation instance and returns its hypsicity, i.e., the change in the oxidation state of
            the atoms """

        if reaction.mapping is None:
            logger.error("No atom mapping: the descriptor cannot be computed")
            raise NoMapping

        # select the desired product and add the oxidation number property to its atoms
        desired_product = next((prod for h, prod in reaction.catalog.items() if h in reaction.role_map['products']),
                               None)
        cif.compute_oxidation_numbers(desired_product.rdmol_mapped)
        desired_product_at = [at for at in reaction.mapping.atom_transformations if
                              at.product_uid == desired_product.uid]
        # identify the involved reactants and add the oxidation number property to their atoms
        reactants_uid = [at.reactant_uid for at in desired_product_at]
        reactants = [mol for h, mol in reaction.catalog.items() if mol.uid in reactants_uid]
        for r in reactants:
            cif.compute_oxidation_numbers(r.rdmol_mapped)
        return self.compute_oxidation_state_change(desired_product_at, desired_product, reactants)

    @staticmethod
    def compute_oxidation_state_change(atom_transformations: list, desired_product: Molecule, reactants: list) -> float:
        """ Computes the change in oxidation state for each mapped atom in the ChemicalEquation """
        delta = 0.0
        # ox_nrs = []
        for at in atom_transformations:
            p_atom = next((atom for atom in desired_product.rdmol_mapped.GetAtoms()
                           if atom.GetIdx() == at.prod_atom_id), None)

            reactant = next((reac for reac in reactants if reac.uid == at.reactant_uid), None)

            r_atom = next((atom for atom in reactant.rdmol_mapped.GetAtoms()
                           if atom.GetIdx() == at.react_atom_id), None)
            if p_atom.GetSymbol() != r_atom.GetSymbol():
                print('problem with mapping')
                raise Exception
            # ox_nrs.append((prod_ox, react_ox))
            delta += p_atom.GetIntProp('_OxidationNumber') - r_atom.GetIntProp('_OxidationNumber')

        # print(ox_nrs)
        return delta


class CEAtomEfficiency(ChemicalEquationDescriptor):
    """ Subclass of atom efficiency at ChemicalEquation level """

    def compute_descriptor(self, reaction: ChemicalEquation) -> float:
        """ Takes a ChemicalEquation instance and compute the atom efficiency """
        if reaction.mapping is None:
            logger.error("No atom mapping: the descriptor cannot be computed")
            raise NoMapping

        # compute the number of mapped atoms in the desired product
        desired_product = next((prod.rdmol_mapped for h, prod in reaction.catalog.items()
                                if h in reaction.role_map['products']), None)
        n_atoms_prod = len([a for a in desired_product.GetAtoms() if a.GetAtomMapNum() not in [0, -1]])
        # compute the number of atoms in all the reactants
        reactants = [reac.rdmol for h, reac in reaction.catalog.items() if h in reaction.role_map['reactants']]
        n_atoms_reactants = sum(r.GetNumAtoms() for r in reactants)
        return n_atoms_prod / n_atoms_reactants


class CDNodeScore(ChemicalEquationDescriptor):
    """ Subclass of NodeScore representing the Convergent Disconnection Score.
        https://pubs.acs.org/doi/10.1021/acs.jcim.1c01074 """

    def compute_descriptor(self, reaction: ChemicalEquation) -> float:
        """ Takes a ChemicalEquation instance and compute the Convergent Disconnection Score [0, 1].
            The closer the score is to 1, the more balanced is the reaction.
        """

        # Retrieve list of products and reactants of the input reaction
        products = [prod.rdmol for h, prod in reaction.catalog.items() if h in reaction.role_map['products']]
        reactants = [reac.rdmol for h, reac in reaction.catalog.items() if h in reaction.role_map['reactants']]

        if len(reactants) == 1:
            return 1

        prod_n_atoms = [p.GetNumAtoms() for p in products]
        reacs_n_atoms = [r.GetNumAtoms() for r in reactants]
        scale_factor = prod_n_atoms[0] / len(reactants)
        abs_error = [abs(r - scale_factor) for r in reacs_n_atoms]
        return 1 / (1 + sum(abs_error) / len(abs_error))


class ChemicalEquationDescriptorCalculator:
    """ Definition of the ChemicalEquationDescriptorCalculator factory. """

    ce_descriptors = {

        'cdscore': CDNodeScore,
        'ce_efficiency': CEAtomEfficiency,
        'ce_hypsicity': CEHypsicity,
    }

    def select_ce_descriptor(self, reaction: ChemicalEquation, score: str):
        """ Takes a string indicating a metrics and a SynGraph and returns the value of the metrics """
        if score not in self.ce_descriptors:
            raise KeyError(f"Invalid score. Available node scores are: {self.ce_descriptors.keys()}")

        calculator = self.ce_descriptors.get(score)
        return calculator().compute_descriptor(reaction)


def node_descriptor_calculator(reaction: ChemicalEquation, score: str):
    """ Gives access to the NodeScoreCalculator factory.
            :param:
                node: a ChemicalEquation instance
                score: a string indicating which score should be computed

            :return:
                score: a float
    """
    if type(reaction) != ChemicalEquation:
        raise TypeError("Step descriptors can be computed only on ChemicalEquation instances.")

    score_selector = ChemicalEquationDescriptorCalculator()
    return score_selector.select_ce_descriptor(reaction, score)


def reaction_mapping(reactant_map: dict, product_map: dict, ids_transferred_atoms: list = None):
    """ Takes the dictionaries mapping the atom ids and their atom-2-atom mapping index for a reactant and a product
        of a ChemicalEquation and returns the list of atom ids transferred from the reactant to the product.

        :params:
            reactant_map: a dictionary {atom_id : mapping_number}

            product_map: a dictionary {atom_id : mapping_number}

            ids_transferred_atoms: a list with the ids of atoms transferred from a previous ChemicalEquation

        :return:
            ids_transferred_atoms: a list with the ids of atoms transferred from the reactant to the product
    """
    if ids_transferred_atoms:
        map_num_transferred_atoms = [map_num for atom_id, map_num in reactant_map.items() if atom_id
                                     in ids_transferred_atoms and map_num != 0]

        ids_transferred_atoms = [atom_id for atom_id, map_num in product_map.items() if map_num
                                 in map_num_transferred_atoms]
    else:
        ids_transferred_atoms = [atom_id for atom_id, map_num in product_map.items() if map_num
                                 in reactant_map.values() and map_num != 0]
    return ids_transferred_atoms
