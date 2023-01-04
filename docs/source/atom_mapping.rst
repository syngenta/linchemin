Atom-to-Atom Mapping
=====================

Being able to perform the atom-to-atom mapping of chemical reactions is fundamental to correctly
assign the role ("reactant", "reagent", "product") to each of the involved chemical compounds and,
thus, to correctly identify identical routes and to compute chemistry-aware route metrics.
However, many of the existing tools to perform the atom mapping have dependencies potentially leading
to conflicts or are proprietary, thus requiring authentication and a licence incompatible with the MIT one.

To minimize the potential conflicts, we preferred to wrap these tools into containers and to
expose their functionalities via REST APIs. A simple SDK operating the endpoints of each service API is also
provided as installable package to simplify the usage of the services from python code.

The containerized atom-to-atom mapping tools are part of our linchemin_services repository, freely
available at https://github.com/syngenta/linchemin_services, where you can also find the documentation
for their installation and usage. Here we only describe how these tools are used within LinChemIn.

atom_mapping overview
---------------------

The :mod:`~linchemin.cheminfo.atom_mapping` module stores all the classes and functions to interact withe the
REST APIs of the containerized atom mapping tools.

The module is composed of a factory structure in which the subclasses of the abstract class
:class:`~linchemin.cheminfo.atom_mapping.Mapper` implement the concrete mappers.
For each subclass the concrete implementation of the abstract method
:meth:`~linchemin.cheminfo.atom_mapping.Mapper.map_chemical_equations` is developed: it
sets up the connection with the url relative to the selected mapper, prepares the input
in the suitable format, submits the request and retrieves the output. The returned object
is an instance of the :class:`~linchemin.cheminfo.atom_mapping.MappingOutput` class.
The calls to the correct ``Mapper`` subclass based on the user's input is handled by
the :class:`~linchemin.cheminfo.atom_mapping.MapperFactory` class.

The factory is wrapped by a the facade function :func:`~linchemin.cheminfo.atom_mapping.perform_atom_mapping`,
which takes as input a list of dictionaries containing the reaction strings to be mapped
and the name of the selected mapper. The :func:`~linchemin.cheminfo.atom_mapping.perform_atom_mapping`
allows users to call a single mapper at time.

Each mapping tool has been developed in a different way and thus has its own characteristics:
namerxn is accurate and fast, but if it fails to classify the submitted reaction, it
does not return any mapping; rxnmapper, on the other side, is a slightly less accurate and
slower, but it always returns a mapping.
In order to efficiently exploit each mapper, we implemented a pipeline
(a chain of responsibility structure) on top of the factory. The pipeline initially
calls rxnmapper and submits all the input reaction strings to it.
If namerxn was able to map all the inputs, the results are returned
as a :class:`~linchemin.cheminfo.atom_mapping.MappingOutput` object and the workflow
ends. However, if not all the reactions were mapped, rxnmapper is called and the unmapped
inputs are submitted to it; the results of the two mappers are put together and returned
as a single :class:`~linchemin.cheminfo.atom_mapping.MappingOutput` instance.

The mapping pipeline can be called by using the
:func:`~linchemin.cheminfo.atom_mapping.pipeline_atom_mapping` function.


.. warning::
    To use the 'namerxn' mapper it is necessary to license HazELNut from NextMove Software and
    obtain the username/password to download the software. If you do not have it, it is recommended to
    use the "single mapper" option and to select one of the freely available tools.



Atom mapping in ChemicalEquation instances
-------------------------------------------

When a mapped smiles or a mapped RDKit ChemicalReaction object are used to instantiate a new
:class:`~linchemin.cheminfo.models.ChemicalEquation` object, a instance of the
:class:`~linchemin.cheminfo.models.Ratam` class is generated. The latter contains all the information
related to the atom to atom mapping of the :class:`~linchemin.cheminfo.models.ChemicalEquation`.

The ``full_map_info`` attribute of :class:`~linchemin.cheminfo.models.Ratam` is a dictionary
whose keys are identifiers of the
:class:`~linchemin.cheminfo.models.Molecule` objects involved in the reaction and the values
are lists of "mapping dictionaries" in the form {atom_id: atom_map_number}. In this way we can keep track
also of molecules that appear more than once in the reaction with different atom mapping.
While building this attribute, a sanity check of the mapping is performed, by making sure that each map
number connects only 2 atoms; if this is not the case, the mapping is considered invalid and an error is raised.

The second attribute of the :class:`~linchemin.cheminfo.models.Ratam` object is the ``atom_transformation``
list. The latter is a list of ``AtomTransformation`` namedtuples, each of which contains a map number,
the ids of atoms connected by the map number and the unique identifiers of the
:class:`~linchemin.cheminfo.models.Molecule` objects to which the atoms belong.

The :class:`~linchemin.cheminfo.models.Ratam` instance is then assigned to the ``mapping``
attribute of the :class:`~linchemin.cheminfo.models.ChemicalEquation` object.

You can find more information and examples about the usage of the atom mapping machinery in the
:ref:`tutorial <tutorial_atom_mapping>`.