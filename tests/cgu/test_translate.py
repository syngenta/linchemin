from unittest.mock import Mock, patch

import pytest

from linchemin.cgu.graph_transformations.data_model_converters import BipartiteGenerator
from linchemin.cgu.translate import (
    InputToIron,
    IronToOutput,
    IronToSynGraph,
    SynGraphToIron,
    TranslationParameters,
    UnavailableTranslation,
    get_available_data_models,
    get_available_formats,
    get_input_formats,
    get_output_formats,
    translator,
)


@pytest.fixture
def mock_data_model_converter(mocker):
    """To mock an instance of a DataModelConverter"""
    mock = Mock()
    mocker.patch(
        "linchemin.cgu.graph_transformations.data_model_converters.BipartiteGenerator",
        return_value=mock,
    )
    return mock


def test_translation_parameters(mock_data_model_converter):
    """To test that the instance of TranslationParameters is correctly created"""
    params = TranslationParameters(
        input_format="input",
        output_format="output",
        data_model_converter=mock_data_model_converter,
    )
    assert params.input_format == "input"
    assert params.output_format == "output"
    assert params.data_model_converter


@pytest.mark.parametrize(
    "input_format, mock_translator_path, data_fixture",
    [
        (
            "az_retro",
            "linchemin.cgu.graph_transformations.format_translators.AzRetro.to_iron",
            "az_as_dict",
        ),
        (
            "ibm_retro",
            "linchemin.cgu.graph_transformations.format_translators.IbmRetro.to_iron",
            "ibm1_as_dict",
        ),
    ],
)
def test_input_to_iron(
    request, input_format, mock_translator_path, data_fixture, mock_data_model_converter
):
    """To test that the correct GraphFormatTranslator is called based on the
    selected input_format in the first step of the chain"""
    route_data = request.getfixturevalue(data_fixture)

    with patch(mock_translator_path) as mock_translator:
        handler = InputToIron()
        params = TranslationParameters(
            input_format=input_format,
            output_format="iron",
            data_model_converter=mock_data_model_converter,
        )
        route = route_data[0]
        handler.translate(route, params)
        mock_translator.assert_called()


@patch(
    "linchemin.cgu.graph_transformations.data_model_converters.BipartiteGenerator.iron_to_syngraph"
)
def test_iron_to_syngraph(mock_method, iron_w_smiles):
    """To test the second step of the chain"""
    handler = IronToSynGraph()
    params = TranslationParameters(
        input_format="az_retro",
        output_format="syngraph",
        data_model_converter=BipartiteGenerator(),
    )
    handler.translate(iron_w_smiles, params)
    mock_method.assert_called()


@patch(
    "linchemin.cgu.graph_transformations.data_model_converters.BipartiteGenerator.syngraph_to_iron"
)
def test_syngraph_to_iron(mock_method, bp_syngraph_instance):
    """To test the third step of the chain"""
    handler = SynGraphToIron()
    params = TranslationParameters(
        input_format="az_retro",
        output_format="networkx",
        data_model_converter=BipartiteGenerator(),
    )
    handler.translate(bp_syngraph_instance, params)
    mock_method.assert_called()


@pytest.mark.parametrize(
    "output_format, mock_translator_path",
    [
        (
            "pydot",
            "linchemin.cgu.graph_transformations.format_translators.PyDot.from_iron",
        ),
        (
            "networkx",
            "linchemin.cgu.graph_transformations.format_translators.Networkx.from_iron",
        ),
    ],
)
def test_iron_to_output(
    output_format, mock_translator_path, mock_data_model_converter, iron_w_smiles
):
    """To test that the correct GraphFormatTranslator is called based on the
    selected output_format in the first step of the chain"""

    with patch(mock_translator_path) as mock_translator:
        handler = IronToOutput()
        params = TranslationParameters(
            input_format="az_retro",
            output_format=output_format,
            data_model_converter=mock_data_model_converter,
        )

        handler.translate(iron_w_smiles, params)
        mock_translator.assert_called()


@patch("os.remove")
@patch("os.system")
@patch("pydot.Dot.write")
@patch("linchemin.IO.io.write_rdkit_depict")
def test_pydot_visualization_translation(
    mock_lio, mock_dot, mock_system, mock_remove, bp_syngraph_instance
):
    translator(
        "syngraph",
        bp_syngraph_instance,
        "pydot_visualization",
        out_data_model="bipartite",
    )
    mock_lio.assert_called()
    mock_dot.assert_called()
    mock_system.assert_called()
    mock_remove.assert_called()


@patch("linchemin.cgu.graph_transformations.format_translators.NOC.from_iron")
def test_noc_translation(mock_noc, bp_syngraph_instance):
    translator("syngraph", bp_syngraph_instance, "noc", out_data_model="bipartite")
    mock_noc.assert_called()


def test_get_available_options():
    options = get_available_formats()
    assert isinstance(options, dict)
    assert "syngraph" in options["as_input"] and "syngraph" in options["as_output"]
    options = get_available_data_models()
    assert isinstance(options, dict)
    assert "bipartite" in options


def test_out_format():
    out = get_output_formats()
    assert isinstance(out, dict) and "pydot_visualization" in out and "iron" in out
    assert "askcosv1" not in out


def test_in_format():
    in_f = get_input_formats()
    assert isinstance(in_f, dict) and "ibm_retro" in in_f and "syngraph" in in_f
    assert "noc" not in in_f


def test_failing_chain(bp_syngraph_instance):
    """To test that an error is raised if an unavailable translation is requested"""
    with pytest.raises(UnavailableTranslation) as ke:
        translator(
            "syngraph", bp_syngraph_instance, "syngraph", out_data_model="bipartite"
        )
    assert "UnavailableTranslation" in str(ke.type)


def test_successful_translation(ibm1_as_dict):
    route = ibm1_as_dict[0]
    nx_route = translator("ibm_retro", route, "networkx", "bipartite")
    assert nx_route
