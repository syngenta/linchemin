Adding graph formats
====================


.. currentmodule:: linchemin.cgu.graph_transformations

The :mod:`~linchemin.cgu.graph_transformations.format_translators`
module stores all the classes
and functions to make translation between
graph formats. If you are planning to include a new format,
this is where you will need to start.
Below we firstly give a brief description of the module's architecture and
then we show a practical example
for including a new input format and one for including an output format.


Translate overview
-------------------

The module :mod:`~linchemin.cgu.graph_transformations.format_translators`
contains a simple factory architecture, in which the subclasses of the abstract class
:class:`~linchemin.cgu.graph_transformations.format_translators.GraphFormatTranslator`
implement the concrete translators to transform a graph object into
and from an :class:`~linchemin.cgu.iron.Iron` instance.
For each subclass the concrete implementation of at least one of two abstract methods
:meth:`~linchemin.cgu.graph_transformations.format_translators.GraphFormatTranslator.from_iron`
and
:meth:`linchemin.cgu.graph_transformations.format_translators.GraphFormatTranslator.to_iron`
is developed.
The :class:`~linchemin.cgu.graph_transformations.format_translators.GraphFormatCatalog`
class
is responsible to keep a registry of the implemented translators and
to instantiate the appropriate concrete class based on its name.


On the other hand, the conversion between data models is handled in the
:mod:`~linchemin.cgu.graph_transformations.data_model_converters` module,
which is also composed of a simple factory structure.
The conversion occurs through the concrete subclasses of the
:class:`~linchemin.cgu.graph_transformations.data_model_converters.DataModelConverter`
abstract class. The
:class:`~linchemin.cgu.syngraph.SynGraph` format is used as carrier of
the data model information and thus
each concrete factory must implement both
the
:meth:`~linchemin.cgu.graph_transformations.data_model_converters.DataModelConverter.iron_to_syngraph`
and the
:meth:`~linchemin.cgu.graph_transformations.data_model_converters.DataModelConverter.syngraph_to_iron` methods.
Also in  this case,
a :class:`~linchemin.cgu.graph_transformations.data_model_converters.DataModelCatalog` class
is used to register and call the appropriate concrete converters.


In the :mod:`~linchemin.cgu.translate` module, a series of handlers compose a
chain of responsibility the uses the Translators and Converters to enforce
a sequence of translations
from the selected input format to the output format. The steps of the sequence are:

- Input format is translated to Iron
- Iron is translated to SynGraph in the selected data model
- SynGraph is translated to Iron
- Iron is translated to the output format

Forcing the translation to pass through SynGraph ensures that the chemical information is handled
correctly, as :class:`~linchemin.cheminfo.models.Molecule` and/or
:class:`~linchemin.cheminfo.models.ChemicalEquation` instances are built while constructing
the SynGraph objects. Moreover, the conversion between data models (i.e., bipartite graph, monopartite
graph with only reactions or monopartite graph with only molecules) is handled exclusively by
SynGraph. This avoids the combinatorial explosion of possibilities to mix and match
data formats and data models.

Lastly, everything is wrapped by the facade function :func:`~linchemin.cgu.translate.translator`.
It takes the 'name' of the input format, the graph object in the input format,
the 'name' of the output format and the 'name' of the output data model and returns the graph
translated in the output format and in the selected data model.


Implementing a new input format
-------------------------------

In order to include a new input format among those that LinChemIn can 'read', you
firstly need to create a new subclass of the abstract class
:class:`~linchemin.cgu.graph_transformations.format_translators.GraphFormatTranslator`
in the :mod:`~linchemin.cgu.graph_transformations.format_translators` module.
The new subclass should also be decorated with the ``@GraphFormatCatalog.register_format``
decorator: it is to register
the new format among the available ones.
The decorator takes two arguments: the name that will be used
to select the format and a brief description that will appear in the helper functions.

.. code-block:: python

    @GraphFormatCatalog.register_format("new_input", "brief description")
    class TranslatorNewInputFormat(GraphFormatTranslator):
    """ Translator subclass to handle translations from NewInputFormat objects """
        as_input = None
        as_output = None

        def from_iron(self, graph: Iron):
            pass

        def to_iron(self, route) -> Iron:
            pass

What you are interested in registering the new translator as "input", you will need to implement the
:meth:`linchemin.cgu.graph_transformations.format_translators.GraphFormatTranslator.to_iron`
method, while the
:meth:`~linchemin.cgu.graph_transformations.format_translators.GraphFormatTranslator.from_iron`
can be left aside for the moment.

Now you need to take your time to develop the actual code that, starting from a graph object
in the format you are trying to add, returns an Iron instance. We recommend to add a 'node_smiles'
key in the ``properties`` dictionary of the Iron nodes, so that the Iron object is suitable
to be translated into a SynGraph instance. Also, remember that the translators work with single graph
objects, not list of objects.
If you need more information regarding the Iron format, you can have a look at the
:ref:`Iron <iron_link>` description.
You should also set the ``as_input`` attribute of the subclass to ``implemented``, so that
the new format will appear as available input format in the helper function.


When the code will be implemented you will have something similar
to this:

.. code-block:: python

    @GraphFormatCatalog.register_format("new_input", "brief description")
    class TranslatorNewInputFormat(GraphFormatTranslator):
    """ Translator subclass to handle translations from NewInputFormat objects """
        as_input = 'implemented'
        as_output = None

        def from_iron(self, graph: Iron):
            pass

        def to_iron(self, route) -> Iron:
            # some super cool code
            return iron_graph




All the available formats are stored in the ``_registered_graph_formats`` attributes of the
:class:`~linchemin.cgu.graph_transformations.format_translators.GraphFormatCatalog`` class.
As previously mentioned, the factory can
self-register new options via the decorator.

That's it, you are all done! Now your newly developed format is available to all the LinChemIn
functionalities that use the
:func:`~linchemin.cgu.translate.translator` function. For example, you can use it to translate a
single route with the :func:`~linchemin.cgu.translate.translator` function:

.. code-block:: python

    from linchemin.cgu.translate import translator

    syngraph = translator('new_input', new_input_graph, 'syngraph', 'bipartite')


or you can work with a list of routes through the :func:`~linchemin.interfaces.facade.facade` function.

.. code-block:: python

    from linchemin.interfaces.facade import facade

    routes, metadata = facade('translate', 'new_input', new_input_graph, 'syngraph', 'bipartite')


Implementing a new output format
---------------------------------

The procedure to add a new output format is the same as the one described above,
with the only difference that you now need to implement the
:meth:`~linchemin.cgu.graph_transformations.format_translators.GraphFormatTranslator.from_iron`
method.
In this case, your code should take an Iron instance as input and,
after the appropriate transformations,
return a graph object in the new format.

.. code-block:: python

    @GraphFormatCatalog.register_format("new_output", "brief description")
    class TranslatorNewOutputFormat(GraphFormatTranslator):
    """ Translator to handle translations from NewOutputFormat objects """
        as_input = None
        as_output = 'implemented'

        def from_iron(self, graph: Iron):
            # some super cool code
            return graph

        def to_iron(self, route) -> Iron:
            pass

Of course, if you want your format to be available as both input and output,
you will need to implement both methods and to set as ``'implemented'`` both the
``as_input`` and ``as_output`` attributes.
