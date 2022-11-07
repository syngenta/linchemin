from linchemin.rem.clustering import (clusterer, SingleRouteClustering, get_available_clustering,
                                      get_clustered_routes_metrics)
from linchemin.cgu.translate import translator
import json
import pytest


def test_clusterer():
    graph = json.loads(open("../cgu/data/az_retro_output_raw.json").read())
    syngraphs = [translator('az_retro', g, 'syngraph', out_data_model='monopartite_reactions') for g in graph]

    # An Exception is raised if the clustering method identifies all the datapoints as noise.
    # E.g., if the dataset is composed of less than 15 routes, hdbscan classifies all points as noise.
    with pytest.raises(Exception) as ke:
        clusterer(syngraphs, ged_method='nx_optimized_ged',
                  clustering_method='hdbscan',
                  ged_params={'reaction_fp': 'structure_fp',
                              })
    assert "Exception" in str(ke.type)

    # A KeyError is raised if an unavailable clustering method is selected
    with pytest.raises(KeyError) as ke:
        clusterer(syngraphs, ged_method='nx_ged', clustering_method='some_clustering')
    assert "KeyError" in str(ke.type)

    with pytest.raises(SingleRouteClustering) as ke:
        clusterer([syngraphs[0]], ged_method='nx_optimized_ged',
                  clustering_method='hdbscan',
                  ged_params={'reaction_fp': 'structure_fp',
                              })
    assert "SingleRouteClustering" in str(ke.type)

    # If everything works, the distance matrix, the clustering algorithm output and the silhouette score are
    # returned
    cluster1, score1, matrix = clusterer(syngraphs, ged_method='nx_optimized_ged',
                                         clustering_method='agglomerative_cluster', linkage='average',
                                         save_dist_matrix=True)
    assert cluster1.labels_.any()
    assert score1
    assert len(matrix) == len(syngraphs)

    cluster2, score2, matrix2 = clusterer(syngraphs, ged_method='nx_optimized_ged',
                                          clustering_method='agglomerative_cluster', linkage='average',
                                          save_dist_matrix=True, parallelization=True, n_cpu=8)

    assert cluster1.labels_.all() == cluster2.labels_.all()
    assert matrix.equals(matrix2)


def test_get_cluster_metrics():
    graph = json.loads(open("../cgu/data/az_retro_output_raw.json").read())
    syngraphs = [translator('az_retro', g, 'syngraph', out_data_model='monopartite_reactions') for g in graph]
    cluster1, score1 = clusterer(syngraphs, ged_method='nx_optimized_ged',
                                 clustering_method='agglomerative_cluster')
    df = get_clustered_routes_metrics(syngraphs, cluster1)
    assert len(df) == len(syngraphs)


def test_get_available_clustering():
    assert type(get_available_clustering()) == dict and \
           'hdbscan' in get_available_clustering()