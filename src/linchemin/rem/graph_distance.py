import abc
import multiprocessing as mp
from dataclasses import dataclass
from functools import partial
from typing import List, Union

import networkx as nx
import numpy as np
import pandas as pd

import linchemin.cheminfo.functions as cif
from linchemin import settings
from linchemin.cgu.syngraph import BipartiteSynGraph, MonopartiteReacSynGraph
from linchemin.cgu.translate import translator
from linchemin.cheminfo.chemical_similarity import (
    compute_mol_fingerprint,
    compute_reaction_fingerprint,
    compute_similarity,
)
from linchemin.cheminfo.models import ChemicalEquation, Molecule
from linchemin.configuration.defaults import DEFAULT_GED
from linchemin.utilities import console_logger

"""
Module containing classes and functions to compute the similarity between pairs of routes.
"""

logger = console_logger(__name__)


class GraphDistanceError(Exception):
    """Base class for exceptions leading to unsuccessful distance calculation."""

    pass


class UnavailableGED(GraphDistanceError):
    """Raised if the selected method to compute the graph distance is not among the available ones."""

    pass


class MismatchingGraph(GraphDistanceError):
    """Raised if the input graphs are of different types"""

    pass


class TooFewRoutes(GraphDistanceError):
    """Raised if fewer than 2 routes are passed when computing the distance matrix"""

    pass


@dataclass
class ChemicalSimilarityParameters:
    reaction_fingerprint: str = settings.GED.reaction_fp
    reaction_fp_params: Union[dict, None] = settings.GED.reaction_fp_params
    reaction_similarity: str = settings.GED.reaction_similarity_name
    molecular_fingerprint: str = settings.GED.molecular_fp
    molecular_fp_params: Union[dict, None] = settings.GED.molecular_fp_params
    molecular_fp_count_vect: bool = settings.GED.molecular_fp_count_vect
    molecular_similarity_name: str = settings.GED.molecular_similarity_name


class Ged(metaclass=abc.ABCMeta):
    """Abstract class for Ged calculators."""

    @abc.abstractmethod
    def compute_ged(
        self,
        syngraph1: Union[MonopartiteReacSynGraph, BipartiteSynGraph],
        syngraph2: Union[MonopartiteReacSynGraph, BipartiteSynGraph],
        ged_params: ChemicalSimilarityParameters,
    ) -> float:
        """
        To calculate the Graph Edit Distance for a pair of graphs.

        Parameters:
        ------------
        syngraph1: Union[MonopartiteReacSynGraph, BipartiteSynGraph]
            The first graph
        syngraph2: Union[MonopartiteReacSynGraph, BipartiteSynGraph]
            The second graph
        ged_params: ChemicalSimilarityParameters
            It contains the parameters to be used in the chemical similarity calculation

        Returns:
        ---------
        ged: float
            The value of the GED
        """
        pass


class GedOptNx(Ged):
    """Subclass for the calculation of the optimized GED algorithm as implemented in NetworkX."""

    def compute_ged(
        self,
        syngraph1: Union[MonopartiteReacSynGraph, BipartiteSynGraph],
        syngraph2: Union[MonopartiteReacSynGraph, BipartiteSynGraph],
        ged_params: ChemicalSimilarityParameters,
    ) -> float:
        """Takes two SynGraph instances, fingerprints and similarity methods for both molecules
        and reactions and returns the GED between the two graphs as computed by the optimized GED algorithm in
        NetworkX."""
        if isinstance(syngraph1, MonopartiteReacSynGraph) and isinstance(
            syngraph2, MonopartiteReacSynGraph
        ):
            out_data_model = "monopartite_reactions"

        elif isinstance(syngraph1, BipartiteSynGraph) and isinstance(
            syngraph2, BipartiteSynGraph
        ):
            out_data_model = "bipartite"

        else:
            logger.error(
                f"Graph1 has type = {type(syngraph1)} \nGraph2 has type = {type(syngraph2)}. "
                f"The GED cannot be computed between graph of different types."
            )
            raise MismatchingGraph
        nx_graphs = [
            translator("syngraph", s, "networkx", out_data_model=out_data_model)
            for s in [syngraph1, syngraph2]
        ]
        # The cost function uses the selected reaction and molecular fingerprints and the selected similarity type.
        node_subst_cost_partial = partial(
            node_subst_cost,
            reaction_fingerprints=ged_params.reaction_fingerprint,
            reaction_fp_params=ged_params.reaction_fp_params,
            reaction_similarity_name=ged_params.reaction_similarity,
            molecular_fingerprint=ged_params.molecular_fingerprint,
            molecular_fp_params=ged_params.molecular_fp_params,
            molecular_fp_count_vect=ged_params.molecular_fp_count_vect,
            molecular_similarity_name=ged_params.molecular_similarity_name,
        )

        opt_ged = nx.optimize_graph_edit_distance(
            nx_graphs[0], nx_graphs[1], node_subst_cost=node_subst_cost_partial
        )

        for g in opt_ged:
            min_g = g
        return min_g


class GedNxPrecomputedMatrix(Ged):
    """Subclass for the calculation of the GED algorithm as implemented in NetworkX; the chemical similarity between
    nodes is precomputed."""

    def compute_ged(
        self,
        syngraph1: Union[MonopartiteReacSynGraph, BipartiteSynGraph],
        syngraph2: Union[MonopartiteReacSynGraph, BipartiteSynGraph],
        ged_params: ChemicalSimilarityParameters,
    ) -> float:
        """Takes two SynGraph instances, fingerprints and similarity methods for both molecules
        and reactions and returns the GED between the two graphs as computed by the GED algorithm in NetworkX.
        The similarity matrix between nodes in the involved graphs is precomputed.
        """
        if isinstance(syngraph1, MonopartiteReacSynGraph) and isinstance(
            syngraph2, MonopartiteReacSynGraph
        ):
            reaction_similarity_matrix = self.precompute_reaction_similarity_matrix(
                syngraph1, syngraph2, ged_params
            )
            out_data_model = "monopartite_reactions"

            # The cost function uses the selected reaction fingerprints.
            node_subst_cost_partial = partial(
                node_subst_cost_matrix,
                reaction_similarity_matrix=reaction_similarity_matrix,
                molecule_similarity_matrix=None,
            )

        elif isinstance(syngraph1, BipartiteSynGraph) and isinstance(
            syngraph2, BipartiteSynGraph
        ):
            reaction_similarity_matrix = self.precompute_reaction_similarity_matrix(
                syngraph1,
                syngraph2,
                ged_params,
            )
            molecule_similarity_matrix = self.precompute_molecule_similarity_matrix(
                syngraph1, syngraph2, ged_params
            )
            out_data_model = "bipartite"

            # The cost function uses the selected reaction and molecular fingerprints and the selected similarity type.
            node_subst_cost_partial = partial(
                node_subst_cost_matrix,
                reaction_similarity_matrix=reaction_similarity_matrix,
                molecule_similarity_matrix=molecule_similarity_matrix,
            )
        else:
            logger.error(
                f"Graph1 has type = {type(syngraph1)} \nGraph2 has type = {type(syngraph2)}. "
                f"The GED cannot be computed between graph of different types."
            )
            raise MismatchingGraph
        nx_graphs = [
            translator("syngraph", s, "networkx", out_data_model=out_data_model)
            for s in [syngraph1, syngraph2]
        ]
        # Retrieve the roots of the routes
        root_g1 = next(n for n, d in nx_graphs[0].out_degree() if d == 0)
        root_g2 = next(n for n, d in nx_graphs[1].out_degree() if d == 0)

        return nx.graph_edit_distance(
            nx_graphs[0],
            nx_graphs[1],
            node_subst_cost=node_subst_cost_partial,
            roots=(root_g1, root_g2),
        )

    @staticmethod
    def precompute_reaction_similarity_matrix(
        syngraph1: Union[MonopartiteReacSynGraph, BipartiteSynGraph],
        syngraph2: Union[MonopartiteReacSynGraph, BipartiteSynGraph],
        ged_params: ChemicalSimilarityParameters,
    ) -> pd.DataFrame:
        """To precompute the similarity matrix for ChemicalEquation nodes only"""
        d_reactions1 = get_reactions_fp_dict(
            syngraph1, ged_params.reaction_fingerprint, ged_params.reaction_fp_params
        )

        d_reactions2 = get_reactions_fp_dict(
            syngraph2, ged_params.reaction_fingerprint, ged_params.reaction_fp_params
        )

        return build_similarity_matrix(
            d_reactions1, d_reactions2, ged_params.reaction_similarity
        )

    @staticmethod
    def precompute_molecule_similarity_matrix(
        syngraph1: Union[MonopartiteReacSynGraph, BipartiteSynGraph],
        syngraph2: Union[MonopartiteReacSynGraph, BipartiteSynGraph],
        ged_params: ChemicalSimilarityParameters,
    ) -> pd.DataFrame:
        """To precompute the similarity matrix for ChemicalEquation nodes only"""
        d_mol1 = get_mol_fp_dict(
            syngraph1,
            ged_params.molecular_fingerprint,
            ged_params.molecular_fp_params,
            ged_params.molecular_fp_count_vect,
        )
        d_mol2 = get_mol_fp_dict(
            syngraph2,
            ged_params.molecular_fingerprint,
            ged_params.molecular_fp_params,
            ged_params.molecular_fp_count_vect,
        )

        return build_similarity_matrix(d_mol1, d_mol2, ged_params.reaction_similarity)


class GedNx(Ged):
    """Subclass for the calculation of the GED algorithm as implemented in NetworkX."""

    def compute_ged(
        self,
        syngraph1: Union[MonopartiteReacSynGraph, BipartiteSynGraph],
        syngraph2: Union[MonopartiteReacSynGraph, BipartiteSynGraph],
        ged_params: ChemicalSimilarityParameters,
    ) -> float:
        """Takes two SynGraph instances, fingerprints and similarity methods for both molecules
        and reactions and returns the GED between the two graphs as computed by the GED algorithm in NetworkX.
        """
        if isinstance(syngraph1, MonopartiteReacSynGraph) and isinstance(
            syngraph2, MonopartiteReacSynGraph
        ):
            out_data_model = "monopartite_reactions"

        elif isinstance(syngraph1, BipartiteSynGraph) and isinstance(
            syngraph2, BipartiteSynGraph
        ):
            out_data_model = "bipartite"

        else:
            logger.error(
                f"Graph1 has type = {type(syngraph1)} \nGraph2 has type = {type(syngraph2)}. "
                f"The GED cannot be computed between graph of different types."
            )
            raise MismatchingGraph
        nx_graphs = [
            translator("syngraph", s, "networkx", out_data_model=out_data_model)
            for s in [syngraph1, syngraph2]
        ]
        # The cost function uses the selected reaction and molecular fingerprints and the selected similarity type.
        node_subst_cost_partial = partial(
            node_subst_cost,
            reaction_fingerprints=ged_params.reaction_fingerprint,
            reaction_fp_params=ged_params.reaction_fp_params,
            reaction_similarity_name=ged_params.reaction_similarity,
            molecular_fingerprint=ged_params.molecular_fingerprint,
            molecular_fp_params=ged_params.molecular_fp_params,
            molecular_fp_count_vect=ged_params.molecular_fp_count_vect,
            molecular_similarity_name=ged_params.molecular_similarity_name,
        )
        # Retrieve the roots of the routes
        root_g1 = next(n for n, d in nx_graphs[0].out_degree if d == 0)
        root_g2 = next(n for n, d in nx_graphs[1].out_degree if d == 0)

        return nx.graph_edit_distance(
            nx_graphs[0],
            nx_graphs[1],
            node_subst_cost=node_subst_cost_partial,
            roots=(root_g1, root_g2),
        )


class GedFactory:
    """GED Factory to give access to the GED calculators.

    Attributes:
    -----------
    available_ged: a dictionary
        It maps the strings representing the 'name' of a GED algorithm to the correct Ged subclass
    """

    available_ged = {
        "nx_ged": {
            "value": GedNx,
            "info": 'Standard NetworkX GED algorithm. The "root" argument is used',
        },
        "nx_ged_matrix": {
            "value": GedNxPrecomputedMatrix,
            "info": "Standard NetworkX GED algorithm. The distance matrix is computed in advance"
            'and the "root" algorithm is used',
        },
        "nx_optimized_ged": {
            "value": GedOptNx,
            "info": "Optimized NetworkX GED algorithm",
        },
    }

    def select_ged(
        self,
        syngraph1,
        syngraph2,
        ged_method,
        ged_params: ChemicalSimilarityParameters,
        # reaction_fp,
        # reaction_fp_params,
        # reaction_similarity_name,
        # molecular_fp,
        # molecular_fp_params,
        # molecular_fp_count_vect,
        # molecular_similarity_name,
    ):
        if ged_method not in self.available_ged:
            logger.error(
                f"'{ged_method}' is invalid. Available algorithms are: {self.available_ged.keys()}"
            )
            raise UnavailableGED

        selector = self.available_ged[ged_method]["value"]
        return selector().compute_ged(syngraph1, syngraph2, ged_params)


def graph_distance_factory(
    syngraph1: Union[MonopartiteReacSynGraph, BipartiteSynGraph],
    syngraph2: Union[MonopartiteReacSynGraph, BipartiteSynGraph],
    ged_method: str,
    ged_params: Union[dict, None] = None,
) -> float:
    """
    To compute the graph edit distance between 2 SynGraph objects

    Parameters:
    -----------
    syngraph1: Union[MonopartiteReacSynGraph, BipartiteSynGraph]
        One of the input graphs
    syngraph2: Union[MonopartiteReacSynGraph, BipartiteSynGraph]
        Another input graph
    ged_method: str
        The graph edit distance algorithm to be used
    ged_params: Union[dict, None]
        It contains the optional parameters for chemical similarity calculations, which are:
        (i) reaction_fp: a string corresponding to the type of fingerprints to be used for reactions
        (ii) reaction_fp_params: a dictionary with the optional parameters for computing reaction fingerprints
        (iii) reaction_similarity_name: a string corresponding to the similarity type to be used for reactions
        (iv) molecular_fp: a string corresponding to the type of fingerprints to be used for molecules
        (v) molecular_fp_params: a dictionary with the optional parameters for computing molecular fingerprints
        (vi) molecular_fp_count_vect: a boolean indicating whether 'GetCountFingerprint' should be used
        (vii) molecular_similarity_name: a string corresponding to the similarity type to be used for molecules
        If it is not provided, the default parameters are used (default None)

    Returns:
    ------
    ged: float
        The ged between the two input graphs

    Example:
    -------
    >>> graphs = json.loads(open('ibm_file.json').read())
    >>> syngraphs = [translator('ibm_retro', g, 'syngraph', out_data_model='bipartite') for g in graphs]
    >>> ged = graph_distance_factory(syngraphs[0], syngraphs[3], ged_method='nx_ged')
    """
    params = set_chemical_similarity_parameters(ged_params)
    ged_calculator = GedFactory()
    return ged_calculator.select_ged(syngraph1, syngraph2, ged_method, params)


def set_chemical_similarity_parameters(
    ged_params: dict,
) -> ChemicalSimilarityParameters:
    """To set the instance of ChemicalSimilarityParameters with the desired parameters"""
    params = ChemicalSimilarityParameters()
    if ged_params is None:
        return params
    params.reaction_fingerprint = ged_params.get(
        "reaction_fp", settings.GED.reaction_fp
    )
    params.reaction_fp_params = ged_params.get(
        "reaction_fp_params", settings.GED.reaction_fp_params
    )
    params.reaction_similarity_name = ged_params.get(
        "reaction_similarity_name", settings.GED.reaction_similarity_name
    )
    params.molecular_fingerprint = ged_params.get(
        "molecular_fp", settings.GED.molecular_fp
    )
    params.molecular_fp_params = ged_params.get(
        "molecular_fp_params", settings.GED.molecular_fp_params
    )
    params.molecular_fp_count_vect = ged_params.get(
        "molecular_fp_count_vect", settings.GED.molecular_fp_count_vect
    )
    params.molecular_similarity_name = ged_params.get(
        "molecular_similarity_name", settings.GED.molecular_similarity_name
    )
    return params


# COST FUNCTIONS
def node_subst_cost_matrix(
    node1,
    node2,
    reaction_similarity_matrix: pd.DataFrame,
    molecule_similarity_matrix: Union[pd.DataFrame, None],
):
    """To compute the cost of substituting ona node with another, based on the pre-computed similarity matrices.
    The more different the nodes, the higher the cost.
    """
    # The correct similarity matrix is used based on the node types
    if isinstance(node1["properties"]["node_type"], ChemicalEquation) and isinstance(
        node2["properties"]["node_type"], ChemicalEquation
    ):
        similarity = reaction_similarity_matrix.loc[
            node2["properties"]["node_type"].uid, node1["properties"]["node_type"].uid
        ]
        return 1.0 - similarity

    elif isinstance(node1["properties"]["node_type"], Molecule) and isinstance(
        node2["properties"]["node_type"], Molecule
    ):
        similarity = molecule_similarity_matrix.loc[
            node2["properties"]["node_type"].uid, node1["properties"]["node_type"].uid
        ]
        return 1.0 - similarity

    else:
        return 1.0


def node_subst_cost(
    node1,
    node2,
    reaction_fingerprints,
    reaction_fp_params,
    reaction_similarity_name,
    molecular_fingerprint,
    molecular_fp_params,
    molecular_fp_count_vect,
    molecular_similarity_name,
) -> float:
    """To compute the cost of substituting one node with another, based on the selected fingerprints/similarity.
    The more different the nodes, the higher the cost.

    Returns:
    ---------
    cost: float
        The cost of the substitution (between 0 and 1)
    """
    # If both nodes are ChemicalEquation, their similarity is computed
    if isinstance(node1["properties"]["node_type"], ChemicalEquation) and isinstance(
        node2["properties"]["node_type"], ChemicalEquation
    ):
        return get_reaction_similarity(
            node1["properties"]["node_type"].rdrxn,
            node2["properties"]["node_type"].rdrxn,
            reaction_fingerprints,
            reaction_fp_params,
            reaction_similarity_name,
        )
    # if both nodes are MoleculeEquation, their similarity is computed
    elif isinstance(node1["properties"]["node_type"], Molecule) and isinstance(
        node2["properties"]["node_type"], Molecule
    ):
        return get_molecular_similarity(
            node1["properties"]["node_type"].rdmol,
            node2["properties"]["node_type"].rdmol,
            molecular_fingerprint,
            molecular_fp_params,
            molecular_similarity_name,
            molecular_fp_count_vect,
        )
    # if the two nodes are of different types, the maximum diversity is returned
    else:
        return 1.0


def get_reaction_similarity(
    rdrxn1: cif.rdChemReactions,
    rdrxn2: cif.rdChemReactions,
    reaction_fingerprint,
    reaction_fp_params,
    reaction_similarity,
) -> float:
    """To compute the similarity between two reactions"""
    fp1 = compute_reaction_fingerprint(
        rdrxn1,
        fp_name=reaction_fingerprint,
        params=reaction_fp_params,
    )
    fp2 = compute_reaction_fingerprint(
        rdrxn2,
        fp_name=reaction_fingerprint,
        params=reaction_fp_params,
    )
    tanimoto = compute_similarity(fp1, fp2, similarity_name=reaction_similarity)
    return 1.0 - tanimoto


def get_molecular_similarity(
    rdmol1: cif.rdChemReactions,
    rdmol2: cif.rdChemReactions,
    molecular_fingerprint,
    molecular_fp_params,
    molecular_similarity_name,
    molecular_fp_count_vect,
) -> float:
    """To compute the similarity between two molecules"""
    fp1 = compute_mol_fingerprint(
        rdmol1,
        fp_name=molecular_fingerprint,
        parameters=molecular_fp_params,
        count_fp_vector=molecular_fp_count_vect,
    )
    fp2 = compute_mol_fingerprint(
        rdmol2,
        fp_name=molecular_fingerprint,
        parameters=molecular_fp_params,
        count_fp_vector=molecular_fp_count_vect,
    )
    tanimoto = compute_similarity(fp1, fp2, similarity_name=molecular_similarity_name)
    return 1.0 - tanimoto


def get_mol_fp_dict(
    syngraph: BipartiteSynGraph,
    molecular_fingerprint: str,
    molecular_fp_params=DEFAULT_GED["molecular_fp_params"]["value"],
    molecular_fp_count_vect=DEFAULT_GED["molecular_fp_count_vect"]["value"],
) -> dict:
    """
    To build a dictionary, whose keys are the hashes of the Molecule nodes in a SynGraph and the values their fingerprints.

    Parameters:
    ------------
    syngraph: BipartiteSynGraph
        The graph object for whose nodes fingerprints should be computed
    molecular_fingerprint: str
        The selected type of molecular fingerprint
    molecular_fp_params: dict
        The optional parameters for computing molecular fingerprints
    molecular_fp_count_vect: bool
        Whether 'GetCountFingerprint' should be used

    Returns:
    ----------
    molecule_node_fingerprints: dict
        The fingerprints of the Molecule nodes
    """

    molecules = get_molecule_nodes(syngraph)

    return {
        mol.uid: compute_mol_fingerprint(
            mol.rdmol,
            molecular_fingerprint,
            molecular_fp_params,
            molecular_fp_count_vect,
        )
        for mol in molecules
    }


def get_reactions_fp_dict(
    syngraph: Union[MonopartiteReacSynGraph, BipartiteSynGraph],
    reaction_fingerprints: str,
    reaction_fp_params=DEFAULT_GED["reaction_fp_params"]["value"],
) -> dict:
    """
    To build a dictionary, whose keys are the hashes of the ChemicalEquation nodes in a SynGraph and the values their fingerprints.

    Parameters:
    ------------
    syngraph: Union[MonopartiteReacSynGraph, BipartiteSynGraph]
        The graph object for whose nodes fingerprints should be computed
    reaction_fingerprints: str
        The selected type of reaction fingerprint
    reaction_fp_params: dict
        The optional parameters for computing reaction fingerprints


    Returns:
    ----------
    reaction_node_fingerprints: dict
        The fingerprints of the ChemicalEquation nodes
    """
    reactions = get_chemical_equation_nodes(syngraph)
    return {
        ce.uid: compute_reaction_fingerprint(
            ce.rdrxn, reaction_fingerprints, reaction_fp_params
        )
        for ce in reactions
    }


def get_chemical_equation_nodes(
    syngraph: Union[MonopartiteReacSynGraph, BipartiteSynGraph]
) -> set:
    """To extract all ChemicalEquation nodes from a SynGraph"""
    reactions = set()
    for parent, children in syngraph.graph.items():
        if isinstance(parent, ChemicalEquation):
            reactions.add(parent)
            [
                reactions.add(child)
                for child in children
                if isinstance(child, ChemicalEquation)
            ]
    return reactions


def get_molecule_nodes(syngraph: BipartiteSynGraph) -> set:
    """To extract all Molecule nodes from a SynGraph"""
    molecules = set()
    for parent, children in syngraph.graph.items():
        if isinstance(parent, Molecule):
            molecules.add(parent)
            [molecules.add(child) for child in children if isinstance(child, Molecule)]
    return molecules


def build_similarity_matrix(
    d_fingerprints1: dict,
    d_fingerprints2: dict,
    similarity_name: str = settings.GED.molecular_similarity_name,
) -> pd.DataFrame:
    """
    To build the similarity matrix between two routes with the selected method.

    Parameters:
    ------------
    d_fingerprints1: dict
        The fingerprint of the first graph to be considered in the form {hash: fingerprints}
    d_fingerprints2: dict
        The fingerprint of the second graph to be considered in the form {hash: fingerprints}
    similarity_name: str
        The similarity method to be used

    Returns:
    ---------
    matrix: pd.DataFrame
        a pandas dataframe (n nodes in graph1) x (n nodes in graph2) containing the similarity values
    """
    columns = list(d_fingerprints1.keys())
    rows = list(d_fingerprints2.keys())
    matrix = pd.DataFrame(
        np.zeros((len(rows), len(columns))), columns=columns, index=rows
    )

    for h1, fp1 in d_fingerprints1.items():
        for h2, fp2 in d_fingerprints2.items():
            if matrix.loc[h2, h1] == 0:
                sim = compute_similarity(fp1, fp2, similarity_name=similarity_name)
                matrix.loc[h2, h1] = sim
    return matrix


def compute_distance_matrix(
    syngraphs: List[Union[MonopartiteReacSynGraph, BipartiteSynGraph]],
    ged_method: str,
    ged_params: Union[dict, None] = None,
    parallelization: bool = False,
    n_cpu=settings.GED.n_cpu,
) -> pd.DataFrame:
    """
    To compute the distance matrix of a set of routes.

    Parameters:
    -----------
    syngraphs: List[Union[MonopartiteReacSynGraph, BipartiteSynGraph]]
        The routes for which the distance matrix must be computed
    ged_method: str
        The graph edit distance method to be used
    ged_params: Optional[Union[dict, None]]
        The dictionary containing the parameters for fingerprints and similarity calculations; if it is not provided,
        the default values are used (default None)
    parallelization: Optional[bool]
        Whether parallelization should be used (default False)
    n_cpu: Optional[int]
        If parallelization is activated, it indicates the number of CPUs to be used (default 8)

    Returns:
    --------
    matrix: a pandas DataFrame
        The distance matrix, with dimensions (n routes x n routes), with the graph distances

    Example:
    --------
    >>> graph = json.loads(open('az_file.json').read())
    >>> mp_syngraphs = [translator('az_retro', g, 'syngraph', out_data_model='monopartite_reactions') for g in graph]
    >>> m = compute_distance_matrix(mp_syngraphs, ged_method='nx_ged')
    """
    if len(syngraphs) < 2:
        logger.error(
            "Less than 2 routes were found: it is not possible to compute the distance matrix"
        )
        raise TooFewRoutes

    # Calculation with parallelization
    if parallelization:
        matrix = setup_parallel_calculation(syngraphs, n_cpu, ged_method, ged_params)

    else:
        n_routes = len(syngraphs)
        matrix = pd.DataFrame(columns=range(n_routes), index=range(n_routes))
        for i in range(n_routes):
            for j in range(i, n_routes):
                sim = graph_distance_factory(
                    syngraphs[i],
                    syngraphs[j],
                    ged_method=ged_method,
                    ged_params=ged_params,
                )
                matrix.loc[j, i] = sim
                matrix.loc[i, j] = sim
    return matrix


def setup_parallel_calculation(
    syngraphs: list, n_cpu: int, ged_method: str, ged_params: dict
) -> pd.DataFrame:
    """To setup the matrix calculation with parallelization"""
    routes_range = range(len(syngraphs))

    results = []
    pool = mp.Pool(n_cpu)
    for i in routes_range:
        in_routes = [(i, j, syngraphs[i], syngraphs[j]) for j in routes_range if j >= i]
        results.append(
            pool.starmap(
                parallel_matrix_calculations,
                [(tup, ged_method, ged_params) for tup in in_routes],
            )
        )
    pool.close()
    return build_ged_matrix(results, routes_range)


def parallel_matrix_calculations(
    data: tuple, ged_method: str, ged_params: dict
) -> tuple:
    """
    To compute the distance matrix elements in a fashion suitable for parallel computation.

    Parameters:
    -----------
    data: tuple
        It contains the two indices and the two routes for computing an element of the distance matrix
        (i, j, route1, route2)
    ged_method: str
        The graph edit distance method to be used
    ged_params: dict
        The optional parameters for the ged calculation

    Returns:
    --------
    i, j, sim: tuple
            Two indices of the matrix and the relative distance value
    """
    (i, j, r1, r2) = data
    sim = graph_distance_factory(r1, r2, ged_method=ged_method, ged_params=ged_params)
    return i, j, sim


def build_ged_matrix(results: list, routes_range: range) -> pd.DataFrame:
    """To build the distance matrix from a list of lists"""
    matrix = pd.DataFrame(columns=routes_range, index=routes_range)
    for result in results:
        for t in result:
            matrix.loc[t[1], t[0]] = t[2]
            matrix.loc[t[0], t[1]] = t[2]
    return matrix


# Helper functions
def get_available_ged_algorithms() -> dict:
    """Returns a dictionary with the available GED algorithms and some info"""
    return {
        f: additional_info["info"]
        for f, additional_info in GedFactory.available_ged.items()
    }


def get_ged_default_parameters() -> dict:
    """Returns a dictionary with the default parameters used in GED calculation"""
    return {f: additional_info["info"] for f, additional_info in DEFAULT_GED.items()}


def get_ged_parameters() -> dict:
    """Returns a dictionary with the default parameters used in GED calculations"""
    return {
        f: additional_info["general_info"] for f, additional_info in DEFAULT_GED.items()
    }
