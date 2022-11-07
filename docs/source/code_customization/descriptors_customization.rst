Adding route descriptors
===========================

.. currentmodule:: linchemin.rem.route_descriptors

The :mod:`~linchemin.rem.route_descriptors` module stores all the classes and functions to calculate
route descriptors, so if you would like to add a new one, this is the module you will need to modify.
Below, we firstly give a brief description of the module architecture and then we show a practical example
for including a route descriptor.


route_descriptors overview
---------------------------

The module is composed by a factory structure in which the subclasses of the abstract class
:class:`~DescriptorCalculator` implement the concrete
calculators of the various descriptors, so that, for example, there is a
``NrBranches``, a ``NrReactionSteps`` etc...
For each subclass the concrete implementation the abstract method
:meth:`~DescriptorCalculator.compute_descriptor` is developed.
The :class:`~DescriptorsCalculatorFactory` class handles the calls to
the correct ``DescriptorCalculator``
subclass based on the user's input.

The factory is wrapped by the facade function :func:`~descriptor_calculator`.
It takes a graph object and the 'name' of the descriptor that should be computed
and returns the value of the descriptor. Currently, all the descriptors are
computed on SynGraph instances, however this is not mandatory and you can implement
your :meth:`~DescriptorCalculator.compute_descriptor` method so that it takes a different type
of graph.

Implementing a new descriptor
-------------------------------

In order to include a new descriptor among those available in LinChemIn, you
firstly need to create a new subclass of the abstract class
:class:`~DescriptorCalculator` in the :mod:`~linchemin.rem.route_descriptors` module and
implement its concrete :meth:`~DescriptorCalculator.compute_descriptor` method.

.. code-block:: python

    class CustomMetric(DescriptorCalculator)
    """ Subclass of DescriptorCalculator representing the CustomMetric. """
        def compute_descriptor(self, graph)
            # some super cool code
            return new_descriptor

The last step is to add the 'name' of your descriptor to the ``route_descriptors`` dictionary,
to make it available to the factory.

.. code-block:: python

    route_descriptors = {
        'longest_seq': {'value': LongestSequence,
                        'info': 'Computes the longest linear sequence in the input SynGraph'},
        'nr_steps': {'value': NrReactionSteps,
                     'info': 'Computes the number of chemical reactions in the input SynGraph'},
        'all_paths': {'value': PathFinder,
                      'info': 'Computes all the paths between the SynRoots and the SynLeaves in the input SynGraph'},
        'nr_branches': {'value': NrBranches,
                        'info': 'Computes the number of branches in the input SynGraph'},
        'branchedness': {'value': Branchedness,
                         'info': 'Computes the "branchedness" of the input SynGraph, weighting the number of '
                                 'branching nodes with their distance from the root'},
        'branching_factor': {'value': AvgBranchingFactor,
                             'info': 'Computes the average branching factor of the input SynGraph'},
        'convergence': {'value': Convergence,
                        'info': 'Computes the "convergence" of the input SynGraph, as the ratio between the '
                                'longest linear sequence and the number of steps'},
        'cdscore': {'value': CDScore,
                    'info': 'Computes the Convergent Disconnection Score of the input SynGraph'},
        'new_desc': {'value': CustomMetric,
                    'info': 'Brief description that will appear in the helper function'}
    }

You can now compute your newly developed descriptor by calling the
:func:`~descriptor_calculator` function:

.. code-block:: python

    from linchemin.rem.route_descriptors import descriptor_calculator

    descriptor_value = descriptor_calculator(graph, 'new_desc')

or use it through the :func:`~linchemin.interfaces.facade.facade` function with a list of routes.

.. code-block:: python

    from linchemin.interfaces.facade import facade

    descriptors, metadata = facade('routes_descriptors', routes_list, descriptors=['new_desc'])

