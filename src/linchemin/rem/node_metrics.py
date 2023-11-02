import abc

from linchemin.cheminfo.models import ChemicalEquation

"""
Module containing functions and classes for computing score and metrics of single nodes of a route.
"""


class NodeScore(metaclass=abc.ABCMeta):
    """Definition of the abstract class for NodeScore."""

    @abc.abstractmethod
    def compute_score(self, node):
        pass


class CDNodeScore(NodeScore):
    """Subclass of NodeScore representing the Convergent Disconnection Score.
    https://pubs.acs.org/doi/10.1021/acs.jcim.1c01074"""

    def compute_score(self, reaction: ChemicalEquation) -> float:
        """Takes a ChemicalEquation instance and compute the Convergent Disconnection Score [0, 1].
        The closer the score is to 1, the more balanced is the reaction.
        """
        if type(reaction) != ChemicalEquation:
            raise TypeError(
                "CDscore can be computed only on ChemicalEquation instances."
            )

        # Retrieve list of products and reactants of the input reaction
        products = [
            prod.rdmol
            for h, prod in reaction.catalog.items()
            if h in reaction.role_map["products"]
        ]
        reactants = [
            reac.rdmol
            for h, reac in reaction.catalog.items()
            if h in reaction.role_map["reactants"]
        ]

        if len(reactants) == 1:
            return 1

        prod_n_atoms = [p.GetNumAtoms() for p in products]
        reacs_n_atoms = [r.GetNumAtoms() for r in reactants]
        scale_factor = prod_n_atoms[0] / len(reactants)
        abs_error = [abs(r - scale_factor) for r in reacs_n_atoms]
        return 1 / (1 + sum(abs_error) / len(abs_error))


class NodeScoreCalculator:
    """Definition of the NodeScore factory."""

    node_scores = {
        "cdscore": CDNodeScore,
    }

    def select_node_score(self, node, score: str):
        """Takes a string indicating a metrics and a SynGraph and returns the value of the metrics"""
        if score not in self.node_scores:
            raise KeyError(
                f"Invalid score. Available node scores are: {self.node_scores.keys()}"
            )

        calculator = self.node_scores.get(score)
        return calculator().compute_score(node)


def node_score_calculator(node, score: str):
    """Gives access to the NodeScoreCalculator factory.
    :param:
        node: a Molecule or ChemicalEquation instance
        score: a string indicating which score should be computed

    :return:
        score: a float
    """
    score_selector = NodeScoreCalculator()
    return score_selector.select_node_score(node, score)


def reaction_mapping(
    reactant_map: dict, product_map: dict, ids_transferred_atoms: list = None
):
    """Takes the dictionaries mapping the atom ids and their atom-2-atom mapping index for a reactant and a product
    of a ChemicalEquation and returns the list of atom ids transferred from the reactant to the product.

    :params:
        reactant_map: a dictionary {atom_id : mapping_number}

        product_map: a dictionary {atom_id : mapping_number}

        ids_transferred_atoms: a list with the ids of atoms transferred from a previous ChemicalEquation

    :return:
        ids_transferred_atoms: a list with the ids of atoms transferred from the reactant to the product
    """
    if ids_transferred_atoms:
        map_num_transferred_atoms = [
            map_num
            for atom_id, map_num in reactant_map.items()
            if atom_id in ids_transferred_atoms and map_num != 0
        ]

        ids_transferred_atoms = [
            atom_id
            for atom_id, map_num in product_map.items()
            if map_num in map_num_transferred_atoms
        ]
    else:
        ids_transferred_atoms = [
            atom_id
            for atom_id, map_num in product_map.items()
            if map_num in reactant_map.values() and map_num != 0
        ]
    return ids_transferred_atoms
