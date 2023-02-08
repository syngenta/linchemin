import abc
import multiprocessing as mp
from functools import partial

import networkx as nx
import numpy as np
import pandas as pd

import linchemin.cheminfo.functions as cif
from linchemin.cgu.syngraph import BipartiteSynGraph, MonopartiteReacSynGraph, SynGraph
from linchemin.cgu.translate import translator
from linchemin.cheminfo.reaction import ChemicalEquation, Molecule

"""
Module containing classes and functions to compute the similarity between pairs of routes.

    AbstractClasses:
        Ged
        
    Classes:
        GedFactory
         
        GedNx(Ged)
        GedNxPrecomputedMatrix(Ged)
        GedOptNx(Ged)

    Functions:
        graph_distance_factory(syngraph1, syngraph2, ged_method: str, reaction_fp,
                               reaction_fp_params, reaction_similarity_name, molecular_fp,
                               molecular_fp_params, molecular_fp_count_vect, 
                               molecular_similarity_name)
                       
        # Functions to compute the cost of node substitution
        node_subst_cost_matrix(node1, node2, reaction_similarity_matrix, molecule_similarity_matrix)
                   
        node_subst_cost(node1, node2, reaction_fingerprints, reaction_fp_params, reaction_similarity_name,
                    molecular_fingerprint, molecular_fp_params, molecular_fp_count_vect, molecular_similarity_name)
        
        # Supporting functions
        compute_nodes_fingerprints(syngraph, reaction_fingerprints, molecular_fingerprint, reaction_fp_params=None,
                               molecular_fp_params=None, molecular_fp_count_vect=False)
                               
        build_similarity_matrix(d_fingerprints1, d_fingerprints2, similarity_name='tanimoto')
        
        compute_distance_matrix(syngraphs: list, ged_method: str, ged_params=None)
        
        get_available_ged_algorithms():

    Global variables:
        DEFAULT
"""
DEFAULT_GED = {
    "reaction_fp": {
        "value": "structure_fp",
        "info": "Structural reaction fingerprints as defined in RDKit",
        "general_info": "Chemical reaction fingerprints to be used",
    },
    "reaction_fp_params": {
        "value": None,
        "info": "The default parameters provided by RDKit are used for the specified"
        " reaction fingerprints",
        "general_info": "Chemical reaction fingerprints parameters",
    },
    "reaction_similarity_name": {
        "value": "tanimoto",
        "info": "The Tanimoto similarity as defined in RDKit is used for"
        " computing the reaction similarity",
        "general_info": "Chemical similarity method for reactions",
    },
    "molecular_fp": {
        "value": "rdkit",
        "info": "The molecular fingerprints rdFingerprintGenerator.GetRDKitFPGenerator"
        " are used",
        "general_info": "Molecular fingerprints to be used",
    },
    "molecular_fp_params": {
        "value": None,
        "info": "The default parameters provided by RDKit are used for the specified"
        " molecular fingerprints",
        "general_info": "Molecular fingerprints parameters",
    },
    "molecular_fp_count_vect": {
        "value": False,
        "info": "False. If set to True, GetCountFingerprint is used for the"
        " molecular fingerprints",
        "general_info": "If set to True, GetCountFingerprint is used for the molecular "
        "fingerprints",
    },
    "molecular_similarity_name": {
        "value": "tanimoto",
        "info": "The Tanimoto similarity as defined in RDKit is used for"
        " computing the molecular similarity",
        "general_info": "Chemical similarity method for molecules",
    },
}


class SingleRouteClustering(Exception):
    pass


class Ged(metaclass=abc.ABCMeta):
    """Abstract class for Ged calculators."""

    @abc.abstractmethod
    def compute_ged(
        self,
        syngraph1,
        syngraph2,
        reaction_fp,
        reaction_fp_params,
        reaction_similarity_name,
        molecular_fp,
        molecular_fp_params,
        molecular_fp_count_vect,
        molecular_similarity_name,
    ):
        """Calculates the Graph Edit Distance for a pair of graphs.

        Parameters:
            syngraph1, syngraph2: two graph objects
                They are graphs for which the GED should be computed

            reaction_fp: a string
                It indicates which reaction fingerprints should be used

            reaction_fp_params: a dictionary
                It contains the parameters for the reaction fingerprints available in RDKIT

            reaction_similarity_name: a string
                It indicates which method should be used to compute the similarity between reactions

            molecular_fp: a string
                It indicates which molecular fingerprints should be used

            molecular_fp_params: a dictionary
                It contains the parameters for the molecular fingerprints available in RDKIT

            molecular_fp_count_vect: a boolean
                If set to True, GetCountFingerprint is used for the molecular fingerprints

            molecular_similarity_name: a string
                It indicates which method should be used to compute the similarity between molecules


        Returns:
            the value of the GED
        """
        pass


class GedOptNx(Ged):
    """Subclass for the calculation of the optimized GED algorithm as implemented in NetworkX."""

    def compute_ged(
        self,
        syngraph1,
        syngraph2,
        reaction_fp,
        reaction_fp_params,
        reaction_similarity_name,
        molecular_fingerprint,
        molecular_fp_params,
        molecular_fp_count_vect,
        molecular_similarity_name,
    ):
        """Takes two SynGraph instances, fingerprints and similarity methods for both molecules
        and reactions and returns the GED between the two graphs as computed by the optmized GED algorithm in
        NetworkX."""
        if (
            type(syngraph1) == type(syngraph2)
            and type(syngraph1) == MonopartiteReacSynGraph
        ):
            nx_graphs = [
                translator(
                    "syngraph", s, "networkx", out_data_model="monopartite_reactions"
                )
                for s in [syngraph1, syngraph2]
            ]
            # The cost function uses the selected reaction fingerprints.
            node_subst_cost_partial = partial(
                node_subst_cost,
                reaction_fingerprints=reaction_fp,
                reaction_similarity_name=reaction_similarity_name,
                reaction_fp_params=reaction_fp_params,
                molecular_fingerprint=molecular_fingerprint,
                molecular_fp_params=molecular_fp_params,
                molecular_fp_count_vect=molecular_fp_count_vect,
                molecular_similarity_name=molecular_similarity_name,
            )
            # The NetworkX GED is called
            opt_ged = nx.optimize_graph_edit_distance(
                nx_graphs[0], nx_graphs[1], node_subst_cost=node_subst_cost_partial
            )

        elif (
            type(syngraph1) == type(syngraph2) and type(syngraph1) == BipartiteSynGraph
        ):
            nx_graphs = [
                translator("syngraph", s, "networkx", out_data_model="bipartite")
                for s in [syngraph1, syngraph2]
            ]
            # The cost function uses the selected reaction and molecular fingerprints and the selected similarity type.
            node_subst_cost_partial = partial(
                node_subst_cost,
                reaction_fingerprints=reaction_fp,
                reaction_fp_params=reaction_fp_params,
                reaction_similarity_name=reaction_similarity_name,
                molecular_fingerprint=molecular_fingerprint,
                molecular_fp_params=molecular_fp_params,
                molecular_fp_count_vect=molecular_fp_count_vect,
                molecular_similarity_name=molecular_similarity_name,
            )

            opt_ged = nx.optimize_graph_edit_distance(
                nx_graphs[0], nx_graphs[1], node_subst_cost=node_subst_cost_partial
            )

        else:
            raise TypeError(
                f"Graph1 has type = {type(syngraph1)} \nGraph2 has type = {type(syngraph2)}. "
                f"The GED cannot be computed between graph of different types."
            )

        for g in opt_ged:
            min_g = g
        return min_g


class GedNxPrecomputedMatrix(Ged):
    """Subclass for the calculation of the GED algorithm as implemented in NetworkX; the chemical similarity between
    nodes is precomputed."""

    def compute_ged(
        self,
        syngraph1,
        syngraph2,
        reaction_fingerprints,
        reaction_fp_params,
        reaction_similarity_name,
        molecular_fingerprint,
        molecular_fp_params,
        molecular_fp_count_vect,
        molecular_similarity_name,
    ):
        """Takes two SynGraph instances, fingerprints and similarity methods for both molecules
        and reactions and returns the GED between the two graphs as computed by the GED algorithm in NetworkX.
        The similarity matrix between nodes in the involved graphs is precomputed.
        """
        if (
            type(syngraph1) == type(syngraph2)
            and type(syngraph1) == MonopartiteReacSynGraph
        ):
            d_reactions1, d_mol = compute_nodes_fingerprints(
                syngraph1,
                reaction_fingerprints,
                molecular_fingerprint,
                reaction_fp_params=reaction_fp_params,
                molecular_fp_params=molecular_fp_params,
                molecular_fp_count_vect=molecular_fp_count_vect,
            )
            d_reactions2, d_mol = compute_nodes_fingerprints(
                syngraph2,
                reaction_fingerprints,
                molecular_fingerprint,
                reaction_fp_params=reaction_fp_params,
                molecular_fp_params=molecular_fp_params,
                molecular_fp_count_vect=molecular_fp_count_vect,
            )
            reaction_similarity_matrix = build_similarity_matrix(
                d_reactions1, d_reactions2, reaction_similarity_name
            )
            nx_graphs = [
                translator(
                    "syngraph", s, "networkx", out_data_model="monopartite_reactions"
                )
                for s in [syngraph1, syngraph2]
            ]
            # The cost function uses the selected reaction fingerprints.
            node_subst_cost_partial = partial(
                node_subst_cost_matrix,
                reaction_similarity_matrix=reaction_similarity_matrix,
                molecule_similarity_matrix=None,
            )
            root_g1 = [n for n, d in nx_graphs[0].out_degree() if d == 0]
            root_g2 = [n for n, d in nx_graphs[1].out_degree() if d == 0]
            # The NetworkX GED is called
            ged = nx.graph_edit_distance(
                nx_graphs[0],
                nx_graphs[1],
                node_subst_cost=node_subst_cost_partial,
                roots=(root_g1[0], root_g2[0]),
            )

        elif (
            type(syngraph1) == type(syngraph2) and type(syngraph1) == BipartiteSynGraph
        ):
            d_reactions1, d_mol1 = compute_nodes_fingerprints(
                syngraph1,
                reaction_fingerprints,
                molecular_fingerprint,
                reaction_fp_params=reaction_fp_params,
                molecular_fp_params=molecular_fp_params,
                molecular_fp_count_vect=molecular_fp_count_vect,
            )
            d_reactions2, d_mol2 = compute_nodes_fingerprints(
                syngraph2,
                reaction_fingerprints,
                molecular_fingerprint,
                reaction_fp_params=reaction_fp_params,
                molecular_fp_params=molecular_fp_params,
                molecular_fp_count_vect=molecular_fp_count_vect,
            )
            reaction_similarity_matrix = build_similarity_matrix(
                d_reactions1, d_reactions2, reaction_similarity_name
            )
            molecule_similarity_matrix = build_similarity_matrix(
                d_mol1, d_mol2, molecular_similarity_name
            )
            nx_graphs = [
                translator("syngraph", s, "networkx", out_data_model="bipartite")
                for s in [syngraph1, syngraph2]
            ]
            # The cost function uses the selected reaction and molecular fingerprints and the selected similarity type.
            node_subst_cost_partial = partial(
                node_subst_cost_matrix,
                reaction_similarity_matrix=reaction_similarity_matrix,
                molecule_similarity_matrix=molecule_similarity_matrix,
            )
            # Retrieve the roots of the routes
            root_g1 = [n for n, d in nx_graphs[0].out_degree() if d == 0]
            root_g2 = [n for n, d in nx_graphs[1].out_degree() if d == 0]

            ged = nx.graph_edit_distance(
                nx_graphs[0],
                nx_graphs[1],
                node_subst_cost=node_subst_cost_partial,
                roots=(root_g1[0], root_g2[0]),
            )

        else:
            raise TypeError(
                f"Graph1 has type = {type(syngraph1)} \nGraph2 has type = {type(syngraph2)}. "
                f"The GED cannot be computed between graph of different types."
            )

        return ged


class GedNx(Ged):
    """Subclass for the calculation of the GED algorithm as implemented in NetworkX."""

    def compute_ged(
        self,
        syngraph1,
        syngraph2,
        reaction_fp,
        reaction_fp_params,
        reaction_similarity_name,
        molecular_fingerprint,
        molecular_fp_params,
        molecular_fp_count_vect,
        molecular_similarity_name,
    ):
        """Takes two SynGraph instances, fingerprints and similarity methods for both molecules
        and reactions and returns the GED between the two graphs as computed by the GED algorithm in NetworkX.
        """
        if (
            type(syngraph1) == type(syngraph2)
            and type(syngraph1) == MonopartiteReacSynGraph
        ):
            nx_graphs = [
                translator(
                    "syngraph", s, "networkx", out_data_model="monopartite_reactions"
                )
                for s in [syngraph1, syngraph2]
            ]
            # The cost function uses the selected reaction fingerprints.
            node_subst_cost_partial = partial(
                node_subst_cost,
                reaction_fingerprints=reaction_fp,
                reaction_fp_params=reaction_fp_params,
                reaction_similarity_name=reaction_similarity_name,
                molecular_fingerprint=molecular_fingerprint,
                molecular_fp_params=molecular_fp_params,
                molecular_fp_count_vect=molecular_fp_count_vect,
                molecular_similarity_name=molecular_similarity_name,
            )
            root_g1 = [n for n, d in nx_graphs[0].out_degree() if d == 0]
            root_g2 = [n for n, d in nx_graphs[1].out_degree() if d == 0]

            ged = nx.graph_edit_distance(
                nx_graphs[0],
                nx_graphs[1],
                node_subst_cost=node_subst_cost_partial,
                roots=(root_g1[0], root_g2[0]),
            )

        elif (
            type(syngraph1) == type(syngraph2) and type(syngraph1) == BipartiteSynGraph
        ):
            nx_graphs = [
                translator("syngraph", s, "networkx", out_data_model="bipartite")
                for s in [syngraph1, syngraph2]
            ]
            # The cost function uses the selected reaction and molecular fingerprints and the selected similarity type.
            node_subst_cost_partial = partial(
                node_subst_cost,
                reaction_fingerprints=reaction_fp,
                reaction_fp_params=reaction_fp_params,
                reaction_similarity_name=reaction_similarity_name,
                molecular_fingerprint=molecular_fingerprint,
                molecular_fp_params=molecular_fp_params,
                molecular_fp_count_vect=molecular_fp_count_vect,
                molecular_similarity_name=molecular_similarity_name,
            )
            # Retrieve the roots of the routes
            root_g1 = [n for n, d in nx_graphs[0].out_degree() if d == 0]
            root_g2 = [n for n, d in nx_graphs[1].out_degree() if d == 0]

            ged = nx.graph_edit_distance(
                nx_graphs[0],
                nx_graphs[1],
                node_subst_cost=node_subst_cost_partial,
                roots=(root_g1[0], root_g2[0]),
            )

        else:
            raise TypeError(
                f"Graph1 has type = {type(syngraph1)} \nGraph2 has type = {type(syngraph2)}. "
                f"The GED cannot be computed between graph of different types."
            )

        return ged


class GedFactory:
    """GED Factory to give access to the GED calculators.

    Attributes:
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
        reaction_fp,
        reaction_fp_params,
        reaction_similarity_name,
        molecular_fp,
        molecular_fp_params,
        molecular_fp_count_vect,
        molecular_similarity_name,
    ):
        if ged_method not in self.available_ged:
            raise KeyError(
                f"'{ged_method}' is invalid. Available algorithms are: {self.available_ged.keys()}"
            )

        selector = self.available_ged[ged_method]["value"]
        return selector().compute_ged(
            syngraph1,
            syngraph2,
            reaction_fp,
            reaction_fp_params,
            reaction_similarity_name,
            molecular_fp,
            molecular_fp_params,
            molecular_fp_count_vect,
            molecular_similarity_name,
        )


def graph_distance_factory(syngraph1, syngraph2, ged_method: str, ged_params=None):
    """Gives access to the Ged factory. Computes the distance matrix between a pair of SynGraph instances.

    Parameters:
        syngraph1, syngraph2: two SynGraph objects

        ged_method: a string
            It corresponds to the graph edit distance algorithm to be used

        ged_params: a dictionary (optional; default: None -> default parameters are used)
            It contains the optional parameters for chemical similarity calculations, which are:
            (i) reaction_fp: a string corresponding to the type of fingerprints to be used for reactions
            (ii) reaction_fp_params: a dictionary with the optional parameters for computing reaction fingerprints
            (iii) reaction_similarity_name: a string corresponding to the similarity type to be used for reactions
            (iv) molecular_fp: a string corresponding to the type of fingerprints to be used for molecules
            (v) molecular_fp_params: a dictionary with the optional parameters for computing molecular fingerprints
            (vi) molecular_fp_count_vect: a boolean indicating whether 'GetCountFingerprint' should be used
            (vii) molecular_similarity_name: a string corresponding to the similarity type to be used for molecules

    Returns:
        The output of the selected similarity algorithm, representing the ged between the two input graphs
    """

    if ged_params is None:
        ged_params = {}

    reaction_fp = ged_params.get("reaction_fp", DEFAULT_GED["reaction_fp"]["value"])
    reaction_fp_params = ged_params.get(
        "reaction_fp_params", DEFAULT_GED["reaction_fp_params"]["value"]
    )
    reaction_similarity_name = ged_params.get(
        "reaction_similarity_name", DEFAULT_GED["reaction_similarity_name"]["value"]
    )
    molecular_fp = ged_params.get("molecular_fp", DEFAULT_GED["molecular_fp"]["value"])
    molecular_fp_params = ged_params.get(
        "molecular_fp_params", DEFAULT_GED["molecular_fp_params"]["value"]
    )
    molecular_fp_count_vect = ged_params.get(
        "molecular_fp_count_vect", DEFAULT_GED["molecular_fp_count_vect"]["value"]
    )
    molecular_similarity_name = ged_params.get(
        "molecular_similarity_name", DEFAULT_GED["molecular_similarity_name"]["value"]
    )

    ged_calculator = GedFactory()
    return ged_calculator.select_ged(
        syngraph1,
        syngraph2,
        ged_method,
        reaction_fp,
        reaction_fp_params,
        reaction_similarity_name,
        molecular_fp,
        molecular_fp_params,
        molecular_fp_count_vect,
        molecular_similarity_name,
    )


# COST FUNCTIONS
def node_subst_cost_matrix(
    node1, node2, reaction_similarity_matrix, molecule_similarity_matrix
):
    """To compute the cost of substituting ona node with another, based on the pre-computed similarity matrices.
    The more different the nodes, the higher the cost.


    """
    # The correct similarity matrix is used based on the node types
    if (
        type(node1["attributes"]["properties"]["node_class"]) == ChemicalEquation
        and type(node2["attributes"]["properties"]["node_class"]) == ChemicalEquation
    ):
        similarity = reaction_similarity_matrix.loc[
            node2["attributes"]["properties"]["node_class"].uid,
            node1["attributes"]["properties"]["node_class"].uid,
        ]
        return 1.0 - similarity

    elif (
        type(node1["attributes"]["properties"]["node_class"]) == Molecule
        and type(node2["attributes"]["properties"]["node_class"]) == Molecule
    ):
        similarity = molecule_similarity_matrix.loc[
            node2["attributes"]["properties"]["node_class"].uid,
            node1["attributes"]["properties"]["node_class"].uid,
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
):
    """To compute the cost of substituting one node with another, based on the selected fingerprints/similarity.
    The more different the nodes, the higher the cost.

    Returns:
        cost: a float between 0 and 1
    """
    # If both nodes are of the type 'ChemicalEquation', their Tanimoto similarity is computed
    if (
        type(node1["attributes"]["properties"]["node_class"]) == ChemicalEquation
        and type(node2["attributes"]["properties"]["node_class"]) == ChemicalEquation
    ):
        rdrxn1 = node1["attributes"]["properties"]["node_class"].rdrxn
        rdrxn2 = node2["attributes"]["properties"]["node_class"].rdrxn
        fp1 = cif.compute_reaction_fingerprint(
            rdrxn1, fp_name=reaction_fingerprints, params=reaction_fp_params
        )
        fp2 = cif.compute_reaction_fingerprint(
            rdrxn2, fp_name=reaction_fingerprints, params=reaction_fp_params
        )
        tanimoto = cif.compute_similarity(
            fp1, fp2, similarity_name=reaction_similarity_name
        )
        return 1.0 - tanimoto

    elif (
        type(node1["attributes"]["properties"]["node_class"]) == Molecule
        and type(node2["attributes"]["properties"]["node_class"]) == Molecule
    ):
        rdmol1 = node1["attributes"]["properties"]["node_class"].rdmol
        rdmol2 = node2["attributes"]["properties"]["node_class"].rdmol
        fp1 = cif.compute_mol_fingerprint(
            rdmol1,
            fp_name=molecular_fingerprint,
            parameters=molecular_fp_params,
            count_fp_vector=molecular_fp_count_vect,
        )
        fp2 = cif.compute_mol_fingerprint(
            rdmol2,
            fp_name=molecular_fingerprint,
            parameters=molecular_fp_params,
            count_fp_vector=molecular_fp_count_vect,
        )
        tanimoto = cif.compute_similarity(
            fp1, fp2, similarity_name=molecular_similarity_name
        )
        return 1.0 - tanimoto

    else:
        return 1.0


def compute_nodes_fingerprints(
    syngraph,
    reaction_fingerprints,
    molecular_fingerprint,
    reaction_fp_params=DEFAULT_GED["reaction_fp_params"]["value"],
    molecular_fp_params=DEFAULT_GED["molecular_fp_params"]["value"],
    molecular_fp_count_vect=DEFAULT_GED["molecular_fp_count_vect"]["value"],
):
    """To create two dictionaries, whose keys are the hashes of the SynGraph nodes and the values their fingerprints.

    Parameters:
        syngraph: A SynGraph object
        reaction_fingerprints: a string corresponding to the selected type of reaction fingerprint
        molecular_fingerprint: a string corresponding to the selected type of molecular fingerprint
        reaction_fp_params: a dictionary with the optional parameters for computing reaction fingerprints
        molecular_fp_params: a dictionary with the optional parameters for computing molecular fingerprints
        molecular_fp_count_vect: a boolean indicating whether 'GetCountFingerprint' should be used

    Returns:
        reaction_nodes_fingerprints: a dictionary containing the fingerprints of the ChemicalEquation nodes
        molecule_node_fingerprints: a dictionary containing the fingerprints of the Molecule nodes
    """
    reaction_nodes_fingerprints = {}
    molecule_node_fingerprints = {}

    for r, connections in syngraph.graph.items():
        if type(r) == ChemicalEquation:
            if r.uid not in reaction_nodes_fingerprints:
                reaction_nodes_fingerprints[r.uid] = cif.compute_reaction_fingerprint(
                    r.rdrxn, fp_name=reaction_fingerprints, params=reaction_fp_params
                )
        elif r.uid not in molecule_node_fingerprints:
            molecule_node_fingerprints[r.uid] = cif.compute_mol_fingerprint(
                r.rdmol,
                fp_name=molecular_fingerprint,
                parameters=molecular_fp_params,
                count_fp_vector=molecular_fp_count_vect,
            )

        for c in connections:
            if type(c) == ChemicalEquation:
                if c.uid not in reaction_nodes_fingerprints:
                    reaction_nodes_fingerprints[
                        c.uid
                    ] = cif.compute_reaction_fingerprint(
                        c.rdrxn,
                        fp_name=reaction_fingerprints,
                        params=reaction_fp_params,
                    )
            elif c.uid not in molecule_node_fingerprints:
                molecule_node_fingerprints[c.uid] = cif.compute_mol_fingerprint(
                    c.rdmol,
                    fp_name=molecular_fingerprint,
                    parameters=molecular_fp_params,
                    count_fp_vector=molecular_fp_count_vect,
                )
    return reaction_nodes_fingerprints, molecule_node_fingerprints


def build_similarity_matrix(
    d_fingerprints1, d_fingerprints2, similarity_name="tanimoto"
):
    """To build the similarity matrix between two routes with the selected method.

    Parameters:
        d_fingerprints1: a dictionary {hash: fingerprints} relative to the first graph, output of
                         compute_nodes_fingerprints
        d_fingerprints2: a dictionary {hash: fingerprints} relative to the second graph, output of
                         compute_nodes_fingerprints
        similarity_name: a string specifying the similarity method to be used

    Returns:
        matrix: a pandas dataframe (n nodes in graph1) x (n nodes in graph2) containing the similarity values
    """
    columns = list(d_fingerprints1.keys())
    rows = list(d_fingerprints2.keys())
    matrix = pd.DataFrame(
        np.zeros((len(rows), len(columns))), columns=columns, index=rows
    )

    for h1, fp1 in d_fingerprints1.items():
        for h2, fp2 in d_fingerprints2.items():
            if matrix.loc[h2, h1] == 0:
                sim = cif.compute_similarity(fp1, fp2, similarity_name=similarity_name)
                matrix.loc[h2, h1] = sim
    return matrix


def compute_distance_matrix(
    syngraphs: list,
    ged_method: str,
    ged_params=None,
    parallelization=False,
    n_cpu=mp.cpu_count(),
):
    """To compute the distance matrix of a set of routes.

    Parameters:
        syngraphs: a list of SynGraph objects
            The routes to use for computing the distance matrix
        ged_method: a string
            It indicates which method to be used for computed the graph edit distance
        ged_params: a dictionary (optional; default: None)
            If provided, it contains the parameters for fingerprints and similarity calculations
        parallelization: a boolean (optional; default: False)
            It indicates whether parallelization should be used
        n_cpu: an integer (optional; default: 'mp.cpu_count()')
            If parallelization is activated, it indicates the number of CPUs to be used

    Returns:
        matrix: a pandas DataFrame
            The distance matrix, with dimensions (n routes x n routes), with the graph distances
    """
    if len(syngraphs) < 2:
        raise (
            SingleRouteClustering(
                "Less than 2 routes were found: clustering not possible"
            )
        )

    routes = range(len(syngraphs))
    matrix = pd.DataFrame(columns=routes, index=routes)

    # Calculation with parallelization
    if parallelization:
        results = []
        pool = mp.Pool(n_cpu)
        for i in routes:
            in_routes = [(i, j, syngraphs[i], syngraphs[j]) for j in routes if j >= i]
            results.append(
                pool.starmap(
                    parallel_matrix_calculations,
                    [(tup, ged_method, ged_params) for tup in in_routes],
                )
            )
        pool.close()

        for result in results:
            for t in result:
                matrix.loc[t[1], t[0]] = t[2]
                matrix.loc[t[0], t[1]] = t[2]

    # Calculation without parallelization
    else:
        for i in routes:
            for j in routes:
                if j >= i:
                    sim = graph_distance_factory(
                        syngraphs[i],
                        syngraphs[j],
                        ged_method=ged_method,
                        ged_params=ged_params,
                    )
                    matrix.loc[j, i] = sim
                    matrix.loc[i, j] = sim
    return matrix


def parallel_matrix_calculations(data, ged_method, ged_params):
    """To compute the distance matrix elements in a fashion suitable for parallel computation.

    Parameters:
        data: a tuple (i, j, route1, route2)
            It contains the two indices and the two routes for computing an element of the distance matrix

    Returns:
        i, j, sim: a tuple
            It contains two indices of the matrix and the relative distance value
    """
    (i, j, r1, r2) = data
    sim = graph_distance_factory(r1, r2, ged_method=ged_method, ged_params=ged_params)
    return i, j, sim


def get_available_ged_algorithms():
    """Returns a dictionary with the available GED algorithms and some info"""
    return {
        f: additional_info["info"]
        for f, additional_info in GedFactory.available_ged.items()
    }


def get_ged_default_parameters():
    """Returns a dictionary with the default parameters used in GED calculation"""
    return {f: additional_info["info"] for f, additional_info in DEFAULT_GED.items()}


def get_ged_parameters():
    return {
        f: additional_info["general_info"] for f, additional_info in DEFAULT_GED.items()
    }
