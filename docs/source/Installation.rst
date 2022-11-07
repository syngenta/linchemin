Installation
============

LinChemIn requires python 3.9 or higher, as well as some common and reliable third party packages such as
``rdkit``, ``pydot``, ``networkx``, ``pandas``, ``numpy``, etc...

Usage installation
--------------------

After cloning the LinChemIn repository from git, you can use ``pip`` to install it:

.. code-block:: console

    $ git clone https://github.com/syngenta/linchemin
    $ cd linchemin
    $ pip install .


Development installation
------------------------

If you are planning to contribute to the code or if you want to implement some custom-defined
functionality, we recommend the installation for development. This will require some additional
packages. To install LinChemIn in development setting simply type:


.. code-block:: console

    $ git clone https://github.com/syngenta/linchemin
    $ cd linchemin
    $ pip install -e .[development]

