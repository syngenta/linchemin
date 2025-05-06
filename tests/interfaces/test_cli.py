from unittest.mock import patch

import pytest

from linchemin.interfaces.cli import linchemin_cli, parse_arguments


# Helper function to simulate command line arguments
def fake_args(args):
    return args.split()


# Test successful parsing of a key=value pair
def test_parse_key_value_pair():
    args = parse_arguments(fake_args("-input_dict key=value"))
    assert args.input_dict == {"key": "value"}


# Test that an invalid key=value pair raises an error
def test_invalid_key_value_pair():
    with pytest.raises(SystemExit):
        parse_arguments(fake_args("-input_dict not_a_key_value"))


# Test the CLI with a mock process_routes function
@patch("linchemin.interfaces.cli.process_routes")
def test_linchemin_cli_with_mock(mock_process_routes, capfd):
    test_args = "-input_dict key=value"
    linchemin_cli(fake_args(test_args))
    mock_process_routes.assert_called()
    out, _ = capfd.readouterr()
    assert "START: LinChemIn" in out
    assert "END: LinChemIn" in out


# Test the CLI without any arguments
def test_cli_no_arguments():
    with pytest.raises(SystemExit):
        linchemin_cli(fake_args(""))


# Test the CLI help message
def test_cli_help_message(capfd):
    with pytest.raises(SystemExit):
        linchemin_cli(fake_args("-h"))
    out, _ = capfd.readouterr()
    assert "usage:" in out
    assert "-input_dict" in out
