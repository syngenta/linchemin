Manual Route Manipulation
==========================

.. currentmodule:: linchemin.cgu.route_mining

So, you predicted a nice set of routes for a target, but checking them, you notice that
a particular reaction or a sequence of reactions that you think might be very useful
is not included in any of them.

No worries, you can provide the already existing routes and a list of reactions that
lead to a molecule appearing in any of the routes, and will be able to retrieve
the new routes (not included in the input list) that the addition of your reactions generates.
All you need is the :func:`~linchemin.cgu.route_mining.route_miner` function!

.. code-block:: python

    from linchemin.cgu.route_mining import route_miner

    new_routes = route_miner(original_routes,           # initial list of routes
                             ['CCO.CCC(O)=O>>CCOC(=O)CC'])  # list of reaction smiles to be added

The input routes should be :class:`~linchemin.cgu.syngraph.MonopartiteReacSynGraph` or
:class:`~linchemin.cgu.syngraph.BipartiteSynGraph` objects and the output, brand-new routes
will be :class:`~linchemin.cgu.syngraph.MonopartiteReacSynGraph` objects.