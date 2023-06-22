Manual Route Manipulation
==========================

.. currentmodule:: linchemin.cgu.route_mining

So, you predicted a nice set of routes for a target, but checking them, you notice that
a particular reaction or a sequence of reactions that you think might be very useful
is not included in any of them.

No worries, you can provide the already existing routes, the smiles of the target molecule and
a list of reactions that you want to add, and will be able to retrieve all
the routes from the obtained tree.
All you need is the :func:`~linchemin.cgu.route_mining.mine_routes` function!

.. code-block:: python

    from linchemin.cgu.route_mining import mine_routes

    input_list = [route1, route2]                       # the initial set of routes
    root = 'CCC(=O)Nc1ccc(cc1)C(=O)N[C@@H](CO)C(=O)O'   # the target molecule for which routes should be retrieved
    new_reaction_list = ['CC=O.O=O>>CC(=O)O']           # the reaction(s) to be added
    routes = mine_routes(input_list, root, new_reaction_list)   # all the mined routes

The input routes should be instances of any :class:`~linchemin.cgu.syngraph.SynGraph` subclasses
and the output routes
will be :class:`~linchemin.cgu.syngraph.MonopartiteReacSynGraph` objects.
