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


    SynGraph
    SynGraph.add_node
    BipartiteSynGraph
    MonopartiteReacSynGraph
    MonopartiteMolSynGraph
    extract_reactions_from_syngraph


Convert
^^^^^^^^^

.. automodule:: linchemin.cgu.convert
.. autosummary::
    :toctree: generated


    converter


Interfaces
----------

Workflows
^^^^^^^^^

.. automodule:: linchemin.interfaces.workflows
.. autosummary::
    :toctree: generated

    process_routes
    get_workflow_options
    WorkflowOutput

Facade
^^^^^^^

.. automodule:: linchemin.interfaces.facade
.. autosummary::
    :toctree: generated

    facade
    facade_helper

Route Evaluation and Metrics (REM)
-----------------------------------

Route Descriptors
^^^^^^^^^^^^^^^^^

.. automodule:: linchemin.rem.route_descriptors
.. autosummary::
    :toctree: generated

    descriptor_calculator
    DescriptorCalculator
    DescriptorCalculator.compute_descriptor
    DescriptorsCalculatorFactory

Graph Distance
^^^^^^^^^^^^^^^^^

.. automodule:: linchemin.rem.graph_distance
.. autosummary::
    :toctree: generated

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

    clusterer
    ClusterCalculator
    ClusterCalculator.get_clustering
    ClusterFactory


Cheminfo
--------

Constructors
^^^^^^^^^^^^^

.. automodule:: linchemin.cheminfo.constructors
.. autosummary::
    :toctree: generated

    MoleculeConstructor
    ChemicalEquationConstructor

Models
^^^^^^^

.. automodule:: linchemin.cheminfo.models
.. autosummary::
    :toctree: generated

    Molecule
    ChemicalEquation
    Ratam


Atom Mapping
^^^^^^^^^^^^^^

.. automodule:: linchemin.cheminfo.atom_mapping
.. autosummary::
    :toctree: generated

    perform_atom_mapping
    pipeline_atom_mapping
    MappingOutput
    get_available_mappers
    Mapper
    Mapper.map_chemical_equations


Configuration (Settings and Secrets)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: linchemin.configuration.config
.. autosummary::
    :toctree: generated

    ConfigurationFileHandler