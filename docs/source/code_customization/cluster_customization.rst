Adding clustering algorithms
============================

.. currentmodule:: linchemin.rem.clustering

The :mod:`~linchemin.rem.clustering` module stores all the classes and functions to perform routes
clustering, so if you would like to add a new algorithm, this is the module you will need to modify.
Below, we firstly give a brief description of the module architecture and then we show a practical example
for including a new algorithm.


clustering overview
--------------------

The module is composed by a factory structure. The subclasses of the abstract class
:class:`~ClusterCalculator` implement the concrete cluster
calculators for applying the clustering algorithms.
For each subclass the concrete implementation of the abstract method
:meth:`~ClusterCalculator.get_clustering` is developed.
The :class:`~ClusterFactory` class handles the calls to
the correct ``ClusterCalculator`` subclass based on the user's input.

The factory is wrapped by the facade function :func:`~clusterer`.
It takes a list of graph objects, the 'name' of the clustering algorithm that should be used
and a series of parameters related to the molecular and reaction fingerprints
and to the chemical similarity calculation to be used.


Implementing a new clustering algorithm
---------------------------------------

In order to include a new clustering algorithm among those available in LinChemIn, you
firstly need to create a new subclass of the abstract class
:class:`~ClusterCalculator` in the :mod:`~linchemin.rem.clustering` module and
implement its concrete :meth:`~ClusterCalculator.get_clustering` method.

.. code-block:: python

    class CustomClusteringAlgorithm(ClusterCalculator)
    """ Subclass of ClusterCalculator applying the CustomClusteringAlgorithm. """
        def get_clustering(self, dist_matrix, save_dist_matrix, **kwargs):
            # some super cool code
            return (clustering, dist_matrix) if save_dist_matrix == True else clustering


The last step is to add the 'name' of your algorithm to the ``available_clustering_algorithms`` dictionary,
to make it available to the factory.

.. code-block:: python

    available_clustering_algorithms = {
        'hdbscan': {'value': HdbscanClusterCalculator(),
                    'info': 'HDBscan algorithm. Not working with less than 15 routes'},
        'agglomerative_cluster': {'value': AgglomerativeClusterCalculator(),
                                  'info': 'Agglomerative Clustering algorithm. '
                                          'The number of clusters is optimized '
                                          'computing the silhouette score'},
        'new_cluster': {'value': CustomClusteringAlgorithm(),
                        'info': 'Brief description that will appear in the helper function'},
    }


You can now use your newly developed clustering algorithm by calling the
:func:`~clusterer` function:

.. code-block:: python

    from linchemin.rem.clustering import clusterer

    cluster1, matrix = clusterer(syngraphs, ged_method='nx_ged',
                                 clustering_method='new_cluster')

Your new clustering algorithm can also be used through the :func:`~linchemin.interfaces.facade.facade`
function:

.. code-block:: python

    from linchemin.interfaces.facade import facade

    cluster, metadata = facade('clustering', routes_list, clustering_method='new_cluster')
