import os
import unittest.mock


def test_cli_basic(capfd, cli):
    cli_str = str(cli)
    os.system(f"python {cli_str} -h")
    out, err = capfd.readouterr()
    assert "usage" in out
    for arg in ["-input_dict", "output_format", "functionalities"]:
        assert arg in out


def test_cli(capfd, cli, mit_path):
    cli_str = str(cli)
    os.system(
        f"python {cli_str} -input_dict {mit_path}=askcos -out_data_model monopartite_reactions -functionalities "
        f"merging"
    )
    out, err = capfd.readouterr()
    assert "Translating the routes in the input file to a list of SynGraph...." in out
    os.remove("routes.json")
    os.remove("tree.json")
