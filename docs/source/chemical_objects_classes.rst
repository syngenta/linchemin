Chemical Objects Classes
==========================

.. currentmodule:: linchemin.cheminfo

LinChemIn provides two classes to store chemical objects:
:class:`~linchemin.cheminfo.models.Molecule` and :class:`~linchemin.cheminfo.models.ChemicalEquation`.


Molecule
----------

Instances of the :class:`~models.Molecule` class hold information regarding chemical compounds
and can be initialized by using the :class:`~linchemin.cheminfo.constructors.MoleculeConstructor`
constructor and its methods.

The :class:`~linchemin.cheminfo.constructors.MoleculeConstructor` class can be instantiated by passing
a ``molecular_identity_property_name`` string, indicating which property determines the identity
of the object (e.g. ‘smiles’) and an ``hash_list``, indicating which properties should be used to compute
additional hash keys.

:class:`~models.Molecule` objects are characterized by various attributes.
The ``molecular_identity_property_name`` attribute indicates which structured-derived molecular
identifier should be used to
compute the hash key (``uid`` attribute) of the Molecule instance. The ``rdmol`` attribute represents
the RDKIT Mol object corresponding to the the Molecule instance and can be used for the dynamic calculation
of molecular properties and fingerprints. The ``smiles`` attribute contains the canonical smiles string
associated with the Molecule instance.

.. code-block:: python

    from linchemin.cheminfo.construcotrs import MoleculeConstructor

    # The MoleculeConstructor is initiated
    molecule_constructor = MoleculeConstructor(molecular_identity_property_name='smiles',
                                               hash_list=['inchi_key', 'inchikey_KET_15T'])

    # The Molecule instance is created from its smiles
    mol_canonical = molecule_constructor.build_from_molecule_string(
                    molecule_string='CCNC(=O)CC', inp_fmt='smiles')


ChemicalEquation
------------------
Instances of the :class:`~models.ChemicalEquation` class hold information regarding chemical reactions
and can be initialized by using the :class:`~linchemin.cheminfo.constructors.ChemicalEquationConstructor`
constructor and its methods.

The :class:`~linchemin.cheminfo.constructors.ChemicalEquationConstructor` class can be instantiated by
passing a ``molecular_identity_property_name`` string, indicating which property determines the identity
of the molecules involved in the reaction (e.g. ‘smiles’) and a ``chemical_equation_identity_name`` string,
indicating which representation of the reaction should be used for determining the identity (e.g., 'r_p'
only considers reactants and products, while 'r_r_p' also takes into account the reagents.

:class:`~models.ChemicalEquation` objects are characterized by various attributes.
The ``catalog`` attribute stores the instances of the Molecule objects involved in the reaction
The involved Molecules instances also appear in the ``role_map`` attributes that maps each of them
on the role they has in the reaction, so that a Molecule can be a 'reagent', a 'reactant' or a 'product'.
The map of the role also determines the has key of the :class:`~models.ChemicalEquation` instance,
as it is based on the hash keys of the Molecule objects having the role of 'reactants' and 'products'.
The hash key is stored in the ``uid`` attribute of the instance.
The ``stoichiometry_coefficients`` dictionary maps the stoichiometry coefficients of each Molecule object
involved in the reaction.
The ``rdrxn`` attribute, storing the rdkit ChemicalReaction object associated with the ChemicalEquation instance,
can be used to compute reaction properties and fingerprints. The ``smiles`` attribute
contains the smiles string associated with the ChemicalEquation instance.


.. code-block:: python

    from linchemin.cheminfo.constructors import ChemicalEquationConstructor

    # The ChemicalEquationConstructor is initiated
    chemical_equation_constructor = ChemicalEquationConstructor(molecular_identity_property_name='smiles',
                                                                chemical_equation_identity_name='r_p')

    # The ChemicalEquation is created from its smiles
    chemical_equation = chemical_equation_constructor.build_from_reaction_string(
                        reaction_string='CCN.CCOC(=O)CC>>CCNC(=O)CC',
                        inp_fmt='smiles')


