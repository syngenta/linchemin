import unittest.mock
import os


def test_cli_basic(capfd):
    os.system('python ../../src/linchemin/interfaces/cli.py -h')
    out, err = capfd.readouterr()
    assert 'usage' in out
    for arg in ['-input_dict', 'output_format', 'functionalities']:
        assert arg in out


def test_cli(capfd):
    os.system('python ../../src/linchemin/interfaces/cli.py -input_dict ../test_file/askos_output.json=askcos')
    out, err = capfd.readouterr()
    assert 'Translating the routes in the input file to a list of SynGraph....' in out
    os.remove('routes.json')

