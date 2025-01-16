import pytest

from linchemin.cheminfo.constructors import MoleculeConstructor
from linchemin.cheminfo.models import Molecule, Ratam, ReactionComponents


@pytest.fixture
def sample_reaction_components():
    mol_constructor = MoleculeConstructor("smiles")
    r1 = mol_constructor.build_from_molecule_string("[BrH:4]", "smiles")
    r1.uid = 11
    r2 = mol_constructor.build_from_molecule_string("[CH2:3]=[CH2:2]", "smiles")
    r2.uid = 22
    rg = mol_constructor.build_from_molecule_string("O", "smiles")
    rg.uid = 33
    p1 = mol_constructor.build_from_molecule_string("[CH3:2][CH2:3][Br:4]", "smiles")
    p1.uid = 44
    rc = ReactionComponents()
    rc.reactants.extend([r1, r2])
    rc.reagents.extend([rg])
    rc.products.extend([p1])
    return rc


@pytest.fixture
def duplicated_reactant():
    r3 = MoleculeConstructor("smiles").build_from_molecule_string("[BrH:4]", "smiles")
    r3.uid = 11
    return r3


def test_reaction_components_get_catalog(
    sample_reaction_components, duplicated_reactant
):
    catalog = sample_reaction_components.get_molecule_catalog()
    assert len(catalog) == 4
    sample_reaction_components.reactants.append(duplicated_reactant)
    catalog = sample_reaction_components.get_molecule_catalog()
    assert len(catalog) == 4
    assert isinstance(catalog[11], Molecule)


def test_role_map(sample_reaction_components):
    role_map = sample_reaction_components.get_role_map()
    assert len(role_map["reactants"]) == 2
    assert len(role_map["products"]) == 1
    assert len(role_map["reagents"]) == 1


def test_stoichiometry(sample_reaction_components, duplicated_reactant):
    stoichiometry = sample_reaction_components.get_stoichiometry_coefficients()
    assert set(stoichiometry["reactants"].values()) == {1}
    assert set(stoichiometry["reagents"].values()) == {1}
    assert set(stoichiometry["products"].values()) == {1}

    sample_reaction_components.reactants.append(duplicated_reactant)
    stoichiometry = sample_reaction_components.get_stoichiometry_coefficients()
    assert stoichiometry["reactants"][11] == 2


@pytest.fixture
def sample_full_mapping_info():
    return {
        "products": {"P1": [{0: 2, 1: 4, 2: 3}]},
        "reactants": {"R1": [{0: 4}], "R2": [{0: 3, 1: 2}]},
        "reagents": {"Rg": [{0: 0}]},
    }


@pytest.fixture
def sample_ratam(sample_full_mapping_info):
    ratam = Ratam()
    ratam.full_map_info = sample_full_mapping_info
    return ratam


def test_ratam_role_map(sample_ratam):
    role_map = sample_ratam.get_role_map()
    assert len(role_map["reactants"]) == 2
    assert role_map["reactants"] == ["R1", "R2"]
    assert len(role_map["reagents"]) == 1
    assert role_map["reagents"] == ["Rg"]
    assert len(role_map["products"]) == 1
    assert role_map["products"] == ["P1"]

    sample_ratam.full_map_info["reactants"]["R1"] = [{0: 4}, {0: 6}]
    assert len(role_map["reactants"]) == 2
    assert role_map["reactants"] == ["R1", "R2"]


def test_ratam_stoichiometry(sample_ratam):
    stoichiometry = sample_ratam.get_stoichiometry_coefficients()
    assert set(stoichiometry["reactants"].values()) == {1}
    assert set(stoichiometry["reagents"].values()) == {1}
    assert set(stoichiometry["products"].values()) == {1}

    sample_ratam.full_map_info["reagents"]["Rg"] = [{0: 0}, {0: 9}]
    stoichiometry = sample_ratam.get_stoichiometry_coefficients()
    assert stoichiometry["reagents"]["Rg"] == 2
