from linchemin.cheminfo import chemical_toolkit


def test_chemical_toolkit_is_available():
    assert chemical_toolkit.get("name") == "rdkit"
