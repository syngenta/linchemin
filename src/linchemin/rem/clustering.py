import abc

import hdbscan
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from linchemin.rem.graph_distance import SingleRouteClustering, compute_distance_matrix
from linchemin.rem.route_descriptors import descriptor_calculator

"""
Module containing classes and functions to compute the clustering of routes based on the distance matrix.

    AbstractClasses:
        ClusterCalculator
        
    Classes:
        HdbscanClusterCalculator(ClusterCalculator)
        AgglomerativeClusterCalculator(ClusterCalculator)
        
    Functions:
        clusterer(syngraphs: list, ged_method: str, clustering_method: str, ged_params=None, save_dist_matrix=False, **kwargs)
        
        
        compute_silhouette_score(dist_matrix, clusterer_labels)
        optimize_agglomerative_cluster(dist_matrix, linkage:str)
        
        get_clustered_routes_metrics(syngraphs:list, clustering_output)
        
        get_available_clustering
"""


class NoClustering(Exception):
    pass


class ClusterCalculator(metaclass=abc.ABCMeta):
    """Definition of the abstract class for ClusterCalculator"""

    @abc.abstractmethod
    def get_clustering(
        self, dist_matrix: pd.DataFrame, save_dist_matrix: bool, **kwargs
    ):
        """Applies the clustering algorithm to the provided distance matrix

        Parameters:
            dist_matrix: a pandas.DataFrame
                It contains the symmetric distance matrix for the routes

            save_dist_matrix: a boolean
                It indicates whether the distance matrix should be returned as an output

            kwargs: possible additional arguments specific for each clustering algorithm
        """
        pass


class HdbscanClusterCalculator(ClusterCalculator):
    """Subclass of ClusterCalculator to apply the Hdbscan algorithm"""

    def get_clustering(self, dist_matrix, save_dist_matrix, **kwargs):
        """Applies the Hdbscan algorithm. Possible optional arguments: min_cluster_size"""
        min_cluster_size = kwargs.get("min_cluster_size", 2)

        clustering = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, metric="precomputed"
        ).fit(dist_matrix.to_numpy(dtype=float))

        if clustering is None:
            raise NoClustering("Clustering was not successful")

        if 0 not in clustering.labels_:
            # hdbscan: with less than 15 datapoints, only noise is found
            raise (Exception("Bad clustering. Only noise was found"))
        s_score = compute_silhouette_score(dist_matrix, clustering.labels_)
        print(f"The Silhouette score is {round(s_score, 3):.3f}")
        return (
            (clustering, s_score, dist_matrix)
            if save_dist_matrix is True
            else (clustering, s_score)
        )


class AgglomerativeClusterCalculator(ClusterCalculator):
    """Subclass of ClusterCalculator to apply the Agglomerative Clustering algorithm"""

    def get_clustering(self, dist_matrix, save_dist_matrix, **kwargs):
        """Applies the Agglomerative Clustering algorithm. Possible optional arguments: linkage"""
        linkage = kwargs.get("linkage", "single")
        clustering, s_score, best_n_cluster = optimize_agglomerative_cluster(
            dist_matrix, linkage
        )

        if clustering is None:
            raise NoClustering("Clustering was not successful")

        if 0 not in clustering.labels_:
            raise (Exception("Bad clustering: only noise was found"))
        print(
            f"The number of clusters with the best Silhouette score is {best_n_cluster}"
        )

        print(f"The Silhouette score is {round(s_score, 3):.3f}")
        return (
            (clustering, s_score, dist_matrix)
            if save_dist_matrix == True
            else (clustering, s_score)
        )


class ClusterFactory:
    """Definition of the Cluster Factory to give access to the clustering algorithms.

    Attributes:
        available_clustering_algorithms: : a dictionary
            It maps the strings representing the 'name' of a clustering algorithm to the correct
            ClusterCalculator subclass
    """

    available_clustering_algorithms = {
        "hdbscan": {
            "value": HdbscanClusterCalculator(),
            "info": "HDBscan algorithm. Not working with less than 15 routes",
        },
        "agglomerative_cluster": {
            "value": AgglomerativeClusterCalculator(),
            "info": "Agglomerative Clustering algorithm. The number of clusters is optimized "
            "computing the silhouette score",
        },
    }

    def select_clustering_algorithms(
        self,
        syngraphs: list,
        ged_method: str,
        clustering_method: str,
        ged_params=None,
        save_dist_matrix=False,
        parallelization=False,
        n_cpu=None,
        **kwargs,
    ):
        if clustering_method not in self.available_clustering_algorithms:
            raise KeyError(
                f"Invalid clustering algorithm. Available algorithms are:"
                f"{self.available_clustering_algorithms.keys()}"
            )

        selector = self.available_clustering_algorithms[clustering_method]["value"]
        dist_matrix = compute_distance_matrix(
            syngraphs, ged_method, ged_params, parallelization, n_cpu
        )

        return selector.get_clustering(dist_matrix, save_dist_matrix, **kwargs)


def clusterer(
    syngraphs: list,
    ged_method: str,
    clustering_method: str,
    ged_params=None,
    save_dist_matrix=False,
    parallelization=False,
    n_cpu=None,
    **kwargs,
):
    """Gives access to the Cluster factory

    Parameters:
        syngraphs: a list of SynGraph objects
            The routes to be clustered
        ged_method: a string
            It indicates the algorithm to be used for GED calculations
        clustering_method: a string
            It indicates the method for clustering to be used
        save_dist_matrix: a boolean
            It indicates whether the distance matrix should be saved and returned as output
        ged_params: a dictionary (optional; default: None -> default parameters are used)
            It contains the optional parameters for ged calculations
        parallelization: a boolean (optional; default: False)
            It indicates whether parallelization should be used for computing distance matrix
        n_cpu: an integer (optional; default: 'mp.cpu_count()')
            If parallelization is activated, it indicates the number of CPUs to be used

        **kwargs:
            The optional parameters specific of the selected clustering algorithm

    Returns:
        clustering, score, (dist_matrix): the output of the clustering, a float and a pandas DataFrame
            The clustering algorithm output, the silhouette score and the distance matrix (save_dist_matrix=True)
    """
    if len(syngraphs) < 2:
        raise (
            SingleRouteClustering(
                "Less than 2 routes were found: clustering not possible"
            )
        )

    clustering_calculator = ClusterFactory()
    return clustering_calculator.select_clustering_algorithms(
        syngraphs,
        ged_method,
        clustering_method,
        ged_params,
        save_dist_matrix,
        parallelization,
        n_cpu,
        **kwargs,
    )


def compute_silhouette_score(dist_matrix, clusterer_labels) -> float:
    """To compute the silhouette score for the clustering of a distance matrix.

    Parameters:
        dist_matrix: a np.array containg a distance matrix
        clusterer_labels: the labels assigned by a clutering algorithm

    Returns:
        score: a float
    """
    return silhouette_score(dist_matrix, clusterer_labels, metric="precomputed")


def optimize_agglomerative_cluster(dist_matrix, linkage: str) -> tuple:
    """To optimize the number of clusters for the AgglomerativeClustering method.

    Parameters:
        dist_matrix: the distance matrix of the analzyed routes as numpy array
        linkage: a string indicating which type of linkage to use in the clustering

    Returns:
        best_clustering: the output of the clustering algorithm with the best silhouette score
        max_score: a float indicating the silhouette score relative to the best_clustering
        best_n_cluster: an integer indicating the number of clusters used to get the best silhouette score
    """

    best_clustering = None
    max_score = -1.0
    best_n_cluster = None

    for n_cluster in range(2, min(5, len(dist_matrix))):
        clustering = AgglomerativeClustering(
            n_clusters=n_cluster, affinity="precomputed", linkage=linkage
        ).fit(dist_matrix.to_numpy(dtype=float))
        score = compute_silhouette_score(dist_matrix, clustering.labels_)

        if score > max_score:
            max_score = score
            best_clustering = clustering
            best_n_cluster = n_cluster

    return best_clustering, max_score, best_n_cluster


def get_clustered_routes_metrics(syngraphs: list, clustering_output) -> pd.DataFrame:
    """To compute the metrics of the routes in the input list grouped by cluster.

    Parameters:
        syngraphs: a list containing the SynGraph/MonopartiteSynGraph for which the metrics should be computed
        clustering_output: the output of a clustering algorithm

    Returns:
         df1: a pandas DataFrame with columns ['routes_id', 'cluster', 'n_steps', 'n_branch']
    """
    labels = clustering_output.labels_
    unique_labels = set(clustering_output.labels_)

    col = ["routes_id", "cluster", "n_steps", "n_branch"]
    df1 = pd.DataFrame(columns=col)

    for k in unique_labels:
        cluster_routes = np.where(clustering_output.labels_ == k)[0].tolist()
        ids = [syngraphs[i].source for i in cluster_routes]
        graphs = [syngraphs[i] for i in cluster_routes]
        d = pd.DataFrame(columns=col)
        for n, graph in enumerate(graphs):
            n_step = descriptor_calculator(graph, "nr_steps")
            route_id = graph.source
            d.loc[n, "routes_id"] = route_id
            d.loc[n, "n_steps"] = n_step
            b = descriptor_calculator(graph, "nr_branches")
            d.loc[n, "n_branch"] = b
            d.loc[n, "cluster"] = k

        df1 = pd.concat([df1, d], ignore_index=True)
    return df1


def get_available_clustering():
    """Returns a dictionary with the available clustering algorithms and some info"""
    return {
        f: additional_info["info"]
        for f, additional_info in ClusterFactory.available_clustering_algorithms.items()
    }
