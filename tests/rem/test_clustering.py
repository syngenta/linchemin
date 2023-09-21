import json

import pytest

from linchemin.cgu.translate import translator
from linchemin.rem.clustering import (
    ClusteringError,
    clusterer,
    get_available_clustering,
    get_clustered_routes_metrics,
)


def test_clusterer(az_path):
    graph = json.loads(open(az_path).read())
    syngraphs = [
        translator("az_retro", g, "syngraph", out_data_model="monopartite_reactions")
        for g in graph
    ]

    # An Exception is raised if the clustering method identifies all the datapoints as noise.
    # E.g., if the dataset is composed of less than 15 routes, hdbscan classifies all points as noise.
    with pytest.raises(ClusteringError) as ke:
        clusterer(
            syngraphs,
            ged_method="nx_optimized_ged",
            clustering_method="hdbscan",
            ged_params={
                "reaction_fp": "structure_fp",
            },
        )
    assert "OnlyNoiseClustering" in str(ke.type)

    # An error is raised if an unavailable clustering method is selected
    with pytest.raises(ClusteringError) as ke:
        clusterer(syngraphs, ged_method="nx_ged", clustering_method="some_clustering")
    assert "UnavailableClusteringAlgorithm" in str(ke.type)

    with pytest.raises(ClusteringError) as ke:
        clusterer(
            [syngraphs[0]],
            ged_method="nx_optimized_ged",
            clustering_method="hdbscan",
            ged_params={"reaction_fp": "structure_fp"},
        )
    assert "SingleRouteClustering" in str(ke.type)

    # If everything works, the distance matrix, the clustering algorithm output and the silhouette score are
    # returned
    cluster1, score1, matrix = clusterer(
        syngraphs,
        ged_method="nx_optimized_ged",
        clustering_method="agglomerative_cluster",
        linkage="average",
        save_dist_matrix=True,
    )
    assert cluster1.labels_.any()
    assert score1
    assert len(matrix) == len(syngraphs)

    cluster2, score2, matrix2 = clusterer(
        syngraphs,
        ged_method="nx_optimized_ged",
        clustering_method="agglomerative_cluster",
        linkage="average",
        save_dist_matrix=True,
        parallelization=True,
        n_cpu=8,
    )

    assert cluster1.labels_.all() == cluster2.labels_.all()
    assert matrix.equals(matrix2)


def test_get_cluster_metrics(az_path):
    graph = json.loads(open(az_path).read())
    syngraphs = [
        translator("az_retro", g, "syngraph", out_data_model="monopartite_reactions")
        for g in graph
    ]
    cluster1, score1 = clusterer(
        syngraphs,
        ged_method="nx_optimized_ged",
        clustering_method="agglomerative_cluster",
    )
    df = get_clustered_routes_metrics(syngraphs, cluster1)
    assert len(df) == len(syngraphs)


def test_get_available_clustering():
    assert (
        type(get_available_clustering()) == dict
        and "hdbscan" in get_available_clustering()
    )
