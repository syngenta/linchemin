import unittest.mock
import os


def test_cli_basic(capfd, cli):
    cli_str = str(cli)
    os.system(f'python {cli_str} -h')
    out, err = capfd.readouterr()
    assert 'usage' in out
    for arg in ['-input_dict', 'output_format', 'functionalities']:
        assert arg in out


def test_cli(capfd, cli, mit_path):
    cli_str = str(cli)
    os.system(f'python {cli_str} -input_dict {mit_path}=askcos')
    out, err = capfd.readouterr()
    assert 'Translating the routes in the input file to a list of SynGraph....' in out
    os.remove('routes.json')

