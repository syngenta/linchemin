Manual Route Manipulation
==========================


Being able to manipulate the predicted routes means that you can include your domain knowledge directly into
the output of the models!
LinChemIn allows you to perform various operations on single and multiple routes

.. currentmodule:: linchemin.cgu.syngraph_operations

Node Removal
---------------
If looking at the predicted route(s) you realize that are some molecules or chemical reactions
that you are not interested in, that re various functions and methods that you can use.

If you just want to remove a single node from a route, you can use the
:meth:`~linchemin.cgu.syngraph.remove_node` method, which is available for all types of SynGraph.
However, you need to build the :class:`~linchemin.cheminfo.models.Molecule` or the
:class:`~linchemin.cheminfo.models.ChemicalEquation` instance corresponding to the node to be removed.

.. code-block:: python

    from linchemin.cheminfo.constructors import MoleculeConstructor

    smiles_to_remove = 'CC(=O)O'   # the smiles of the molecule to be removed
    mol_to_remove = MoleculeConstructor().build_from_molecule_string(smiles_to_remove, 'smiles')    # the Molecule object is built
    syngraph.remove_node(mol_to_remove)


In case you want to remove not only a node, but also all its "parents" connections, to avoid leaving "dandling"
sequences of nodes, you can use the
:func:`~linchemin.cgu.syngraph_operations.remove_nodes_from_syngraph` function. In this case, you just need to
provide the smiles of the node you want to remove and a new SynGraph object will be returned.

.. code-block:: python

    from linchemin.cgu.syngraph_operations import remove_nodes_from_syngraph

    smiles_to_remove = 'CCN.CCOC(=O)CC>>CCNC(=O)CC'   # the smiles of the reaction to be removed
    new_graph = remove_nodes_from_syngraph(original_syngraph, smiles_to_remove)


.. currentmodule:: linchemin.cgu.route_mining

Route Mining and Node Addition
----------------------------------
So, you predicted a nice set of routes for a target,
but checking them, you notice that
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
