References
==========

.. currentmodule:: linchemin

Core Graph Utilities (CGU)
---------------------------

Translate
^^^^^^^^^^

.. automodule:: linchemin.cgu.translate
.. autosummary::
    :toctree: generated
        :hidden:


    translator
    AbsTranslator
    AbsTranslator.from_iron
    AbsTranslator.to_iron
    TranslatorFactory

Iron
^^^^^

.. automodule:: linchemin.cgu.iron
.. autosummary::
    :toctree: generated
        :hidden:


    Iron
    Iron.add_node
    Iron.add_edge
    Node
    Edge
    Direction


Syngraph
^^^^^^^^^

.. automodule:: linchemin.cgu.syngraph
.. autosummary::
    :toctree: generated
        :hidden:


    SynGraph
    SynGraph.add_node
    BipartiteSynGraph
    MonopartiteReacSynGraph
    MonopartiteMolSynGraph


Convert
^^^^^^^^^

.. automodule:: linchemin.cgu.convert
.. autosummary::
    :toctree: generated
        :hidden:


    converter


Interfaces
----------

Workflows
^^^^^^^^^

.. automodule:: linchemin.interfaces.workflows
.. autosummary::
    :toctree: generated
        :hidden:

    process_routes
    get_workflow_options
    WorkflowOutput

Facade
^^^^^^^

.. automodule:: linchemin.interfaces.facade
.. autosummary::
    :toctree: generated
        :hidden:


    facade
    facade_helper

Route Evaluation and Metrics (REM)
-----------------------------------

Route Descriptors
^^^^^^^^^^^^^^^^^

.. automodule:: linchemin.rem.route_descriptors
.. autosummary::
    :toctree: generated
        :hidden:

    descriptor_calculator
    DescriptorCalculator
    DescriptorCalculator.compute_descriptor
    DescriptorsCalculatorFactory

Graph Distance
^^^^^^^^^^^^^^^^^

.. automodule:: linchemin.rem.graph_distance
.. autosummary::
    :toctree: generated
        :hidden:

    compute_distance_matrix
    Ged
    Ged.compute_ged
    GedFactory
    graph_distance_factory

Clustering
^^^^^^^^^^^

.. automodule:: linchemin.rem.clustering
.. autosummary::
    :toctree: generated
        :hidden:

    clusterer
    ClusterCalculator
    ClusterCalculator.get_clustering
    ClusterFactory


Cheminfo
--------

Molecule
^^^^^^^^^^^

.. automodule:: linchemin.cheminfo.molecule
.. autosummary::
    :toctree: generated
        :hidden:

    Molecule
    MoleculeConstructor

Reaction
^^^^^^^^^^^

.. automodule:: linchemin.cheminfo.reaction
.. autosummary::
    :toctree: generated
        :hidden:

    ChemicalEquation
    ChemicalEquationConstructor

