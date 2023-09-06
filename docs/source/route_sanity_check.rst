Routes' Sanity Checks
======================

.. currentmodule:: linchemin.interfaces
.. automodule:: linchemin.interfaces

Why do you need to check your routes?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The route predicted by the CASP tools might contain some "chemical" issues
that can translate into "graph" issues when the routes are re-built based on the chemical logic
of reactants --> products. In LinChemIn this happens, for example, when the routes are
passed through the atom-to-atom mapping machinery: you give a list of routes, the involved chemical reactions
are extracted and mapped (so that you can have more information about them) and, in the end, the
routes are re-built. In this case, you might notice that some descriptors seem incorrect
and if you look at the routes, you will probably see some funky structures, with cycles
or isolated nodes. This is due to the aforementioned "chemical" problems that have also become "graph" problems.

Since we do not want you to delete whole routes because of these issues, we implemented a facade
functionality that can identified some of these problems and handle them, by removing the extra nodes.
Currently we are able to identify two types of problems: (i) the presence of cycles, due to more than one reaction
producing the same molecule and (ii) the presence of isolated (sequence of) nodes,
usually related to a branch in the route that produces a reagent (instead of a reactant) of the next reaction.
Hopefully the list of these "issues" will not grow much, but whenever we will catch other tricky cases,
we will try to implement the related sanity check!

Perform the sanity checks
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to take care of your routes, you just need the :func:`~facade.facade` function and
its ``routes_sanity_checks`` functionality. You just need to pass the list of
routes that you would like to check as :class:`~syngraph.BipartiteSynGraph`
or :class:`~syngraph.MonopartiteReacSynGraph` objects. To determine which sanity checks should be performed,
we can pass a list of names
to the ``checks`` argument; if the latter is not specified all the implemented
will be applied. You can also decide in which data model the output routes should be returned
by specifying the ``out_data_model`` argument (the default is the one specified in the configuration file)

.. code-block:: python

    from linchemin.interfaces.facade import facade

    # all the available sanity checks are performed
    checked_routes, meta = facade('routes_sanity_checks',
                                  routes)           # the list of routes to be checked

    # only the cycle check is performed and the routes are returned as MonopartiteReacSynGraph objects
    checked_routes, meta = facade('routes_sanity_checks',
                                  routes,                   # the list of routes to be checked
                                  checks=['cycle_check'],   # a specific check is selected
                                  out_data_model='monopartite_reactions')   # the data model of the output routes is selected


If you want to know more of the options and default values for this functionality, you can call
the facade helper function:

.. code-block:: python

    from linchemin.interfaces.facade import facade_helper

    facade_helper(functionality='routes_sanity_checks')

