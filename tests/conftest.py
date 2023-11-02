from pathlib import Path

import pytest


@pytest.fixture
def ibm1_path():
    conftest_path = Path(__file__)
    data_path = conftest_path.parent / "test_file"
    return data_path.joinpath("ibmrxn_retro_output_raw.json")


@pytest.fixture
def ibm2_path():
    conftest_path = Path(__file__)
    data_path = conftest_path.parent / "test_file"
    return data_path.joinpath("ibm_output2.json")


@pytest.fixture
def az_path():
    conftest_path = Path(__file__)
    data_path = conftest_path.parent / "test_file"
    return data_path.joinpath("az_retro_output_raw.json")


@pytest.fixture
def mit_path():
    conftest_path = Path(__file__)
    data_path = conftest_path.parent / "test_file"
    return data_path.joinpath("askos_output.json")


@pytest.fixture
def cli():
    conftest_path = Path(__file__)
    data_path = conftest_path.parent.parent
    return data_path.joinpath("src/linchemin/interfaces/cli.py")
