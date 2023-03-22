Adding graph similarity algorithms
===================================

.. currentmodule:: linchemin.rem.graph_distance

The :mod:`~linchemin.rem.graph_distance` module stores all the classes and functions to calculate the
'similarity' between routes, so if you would like to add a new algorithm, this is the module you will need to modify.
Below, we firstly give a brief description of the module architecture and then we show a practical example
for including a new algorithm.


graph_distance overview
------------------------

The module is composed by a factory structure in which the subclasses of the abstract class
:class:`~Ged` implement the concrete Graph Edit Distance (GED)
calculators. For each subclass the concrete implementation the abstract method
:meth:`~Ged.compute_ged` is developed.
The :class:`~GedFactory` handles the calls to the correct ``Ged`` subclasses based on the user's input.

The factory is wrapped by the facade function :func:`~graph_distance_factory`.
It takes two graph objects, the 'name' of the GED algorithm to be used and a dictionary containing
the optional parameters for the chemical similarity calculations. The latter are used in the 'cost
functions' for substituting two nodes of the GED algorithms; in this way, the chemical
information is introduced in the GED algorithm.

In the :mod:`~linchemin.rem.graph_distance` are also implemented two 'cost functions'
that return the 'distance' between two nodes of the same type as (1 - chemical similarity)

Implementing a new GED algorithm
---------------------------------------

In order to include a new GED algorithm among those available in LinChemIn, you
firstly need to create a new subclass of the abstract class
:class:`~Ged` in the :mod:`~linchemin.rem.graph_distance` module and
implement its concrete :meth:`~Ged.compute_ged` method.

.. code-block:: python

    class CustomGEDAlgorithm(Ged)
    """ Subclass of Ged implementing the CustomGEDAlgorithm. """
        def compute_ged(self, syngraph1, syngraph2, reaction_fp, reaction_fp_params,
                        reaction_similarity_name, molecular_fingerprint, molecular_fp_params,
                        molecular_fp_count_vect, molecular_similarity_name):
            # some super cool code
            return ged


The last step is to add the 'name' of your algorithm to the ``available_ged`` dictionary of the
:class:`~GedFactory`, to make it available to the factory.

.. code-block:: python

    available_ged = {
        'nx_ged': {'value': GedNx,
                   'info': 'Standard NetworkX GED algorithm. The "root" argument is used'},
        'nx_ged_matrix': {'value': GedNxPrecomputedMatrix,
                          'info': 'Standard NetworkX GED algorithm. The distance matrix is computed in advance'
                                  'and the "root" algorithm is used'},
        'nx_optimized_ged': {'value': GedOptNx,
                             'info': 'Optimized NetworkX GED algorithm'},
        'new_ged': {'value': CustomGEDAlgorithm,
                    'info': 'Brief description that will appear in the helper function'},
    }

You can now compute your newly developed descriptor by calling the
:func:`~graph_distance_factory` function:

.. code-block:: python

    from linchemin.rem.graph_distance import graph_distance_factory

    ged = graph_distance_factory(syngraphs1, syngraphs2, ged_method='new_ged')

The newly developed algorithm for computing the GED (or another 'similarity' measure) is also
automatically available to the :func:`~linchemin.interfaces.facade.facade` function, from which
you will be able to work with a list of routes:

.. code-block:: python

    from linchemin.interfaces.facade import facade

    dist_matrix, metadata = facade('distance_matrix', routes_list, ged_method='new_ged')
