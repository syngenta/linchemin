Graph Data Structures
=====================

.. currentmodule:: linchemin.cgu

LinChemIn provides two internal data structures and relative methods to store and manipulate graphs:
:class:`~linchemin.cgu.iron.Iron` and :class:`~linchemin.cgu.syngraph.SynGraph`.

.. _iron_link:


Iron
----

.. currentmodule:: linchemin.cgu

:class:`~iron.Iron` is mainly a "service" class used as a sort of abstraction layer
between data structures. This is very useful for our :func:`~translate.translator` function,
in which the input graph is initially converted to an Iron instance and then transformed in the
selected output format.

The nodes of :class:`~iron.Iron` are instances of the class :class:`~iron.Node`, characterized by an id,
a dictionary of properties and a list of labels. Similarly, the edges are instances of the class
:class:`~iron.Edge`, also characterized by an id, a dictionary of properties and a list of labels;
in addition, the :class:`~iron.Edge` class has attributes storing the id of the nodes that
it connects and a direction. The direction of the edge is
itself an instance of another class, :class:`~linchemin.cgu.iron.Direction`, which stores
the id of the "parent" and "child" nodes.

An :class:`~iron.Iron` instance is composed of two dictionaries.
The first one stores the nodes with their properties and labels,
in the form ``{id: Node}``, while the other stores the edges with the relative information, also in the form
``{id: Edge}``.
When a new ``Iron`` instance is instantiated, an empty graph object is created;
it can then be populated by adding nodes and edges using the methods :meth:`~iron.Iron.add_node` and
:meth:`~iron.Iron.add_edge`.
The code snippet below shows how to create and populate an Iron instance.

.. code-block:: python

    from linchemin.cgu.iron import Iron, Node, Edge, Direction
    # The new Iron instance is initialized
    iron_graph = Iron()

    # Two nodes, instances of the Node class, are defined and added to the Iron instance
    parent_node = Node(properties={'node_smiles': 'CCN',    # "parent" node properties
                                   'prop1': 'some_value'},
                       iid='0',                 # "parent" node id
                       labels=['parent_node'])  # "parent" node labels

    iron_graph.add_node(parent_node.iid, parent_node)       # the "parent" node is added

    child_node = Node(properties={'node_smiles': 'CCC(=O)NCC',  # "child" node properties
                                  'prop2': 'some_other_value'},
                       iid='1',                         # "child" node id
                       labels=['child_node'])           # "child" node labels

    iron_graph.add_node(child_node.iid, child_node)     # the "child" node is added

    # An edge, instance of the Edge class, connecting the two previously defined nodes, is added
    d = Direction('{}>{}'.format(parent_node.iid, child_node.iid))  # edge direction is instantiated

    edge = Edge(iid='0',    # edge id
                a_iid=parent_node.iid,  # id of the "parent" node
                b_iid=child_node.iid,   # id of the "child" node
                direction=d,            # edge direction
                properties={'some property': 'some value'}, # edge properties
                labels=['some label'])                      # edge labels

    iron_graph.add_edge(edge.iid, edge) # the edge is added


We recommend to add a 'node_smiles'
key in the ``properties`` dictionary of the Iron nodes, so that the Iron object is suitable
to be translated into a SynGraph instance.

SynGraph
--------

.. currentmodule:: linchemin.cgu

The abstract class :class:`~syngraph.SynGraph` is
the implementation of the homonym data model and represents
the backbone of LinChemIn, being used as the underlying data structure for most of the code functions.
It contains a graph-like structure implemented as a dictionary of sets: the key encodes a "parent" node
having out edge(s), and the value is a python set containing all its "children" nodes.
Using a set ensures no duplicates among the "children" nodes.
While nodes are explicit, the edges stay implicit, and their direction is presumed to always be from
the "parent" node to the "children" nodes.

The subclasses of :class:`~syngraph.SynGraph` are :class:`~syngraph.BipartiteSynGraph`,
:class:`~syngraph.MonopartiteReacSynGraph` and :class:`~syngraph.MonopartiteMolSynGraph`,
each of which represents a specific data model.

An instance of any ``SynGraph`` can be initialized by passing an Iron instance
whose nodes have at least the property ``node_smiles``. This allows the builder to construct
the instances of the :class:`~linchemin.cheminfo.models.Molecule` or
:class:`~linchemin.cheminfo.models.ChemicalEquation` classes, which will be the nodes
of the :class:`~syngraph.SynGraph` object.
As an alternative, it is possible to pass a list of dictionaries containing reaction strings, such as SMILES,
in the form ``[{'query_id': reaction_id, 'output_string': reaction_string}]``.
The last option is to create an empty :class:`~syngraph.SynGraph` instance
and add nodes using the method
:meth:`~syngraph.SynGraph.add_node`.

.. code-block:: python

    from linchemin.cgu.syngraph import BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph
    # A BipartiteSynGraph is initiated by passing a route in Iron format
    bp_syngraph = BipartiteSynGraph(iron_route)

    # A MonopartiteReacSynGraph is initiated by passing a list of dictionaries of reaction smiles
    reactions_list =[{'query_id': 0, 'output_string': 'CC(=O)CC.CCN>>CC/N=C(C)\CC'},
                     {'query_id': 1, 'output_string': 'CC/N=C(C)\CC.N#CC[NaB]>>CCNC(C)CC'}]

    mpr_syngraph = MonopartiteReacSynGraph(reactions_list)

    # A MonopartiteMolSynGraph is initiated as an empty instance and then nodes are added
    mpm_syngraph = MonopartiteMolSynGraph()
    mpm_syngraph.add_node(('CCN', ['CCNC(=O)CC']))
    mpm_syngraph.add_node(('CCOC(=O)CC', ['CCNC(=O)CC']))


In order to convert one type of :class:`~syngraph.SynGraph` into another,
the :func:`~convert.converter` can be used.

.. code-block:: python

    from linchemin.cgu.convert import converter
    # A BipartiteSynGraph is converted into a MonopartiteReacSynGraph
    mpr_syngraph = converter(bp_syngraph, 'monopartite_reactions')

    # A MonopartiteReacSynGraph is converted into a BipartiteSynGraph
    bp_syngraph = converter(mpr_syngraphs, 'bipartite')


