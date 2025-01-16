from unittest.mock import MagicMock, Mock, patch

import pytest

import linchemin.cheminfo.functions as cif
from linchemin.cheminfo.constructors import ChemicalEquationConstructor
from linchemin.cheminfo.model_constructors.disconnection_constructor import (
    BondInfo,
    DisconnectionBuilder,
    ReactiveCenter,
    ReactiveCenterAnalyzer,
    create_disconnection,
)
from linchemin.cheminfo.models import ChemicalEquation, Disconnection, Molecule


@pytest.fixture
def sample_reaction():
    # sample reaction: Cl.Br.C=C>>ClCCBr
    reactants = [
        cif.Chem.MolFromSmiles("[BrH:4]"),
        cif.Chem.MolFromSmiles("[CH2:2]=[CH2:3]"),
        cif.Chem.MolFromSmiles("[ClH:1]"),
    ]
    product = cif.Chem.MolFromSmiles("[Cl:1][CH2:2][CH2:3][Br:4]")
    reaction_smiles_empty = ">>"
    rdrxn = cif.rdChemReactions.ReactionFromSmarts(
        reaction_smiles_empty, useSmiles=True
    )
    [rdrxn.AddReactantTemplate(r) for r in reactants]
    rdrxn.AddProductTemplate(product)

    cif.rdChemReactions.SanitizeRxn(rdrxn)
    return rdrxn


def test_get_reacting_atoms_map_numbers(sample_reaction):
    reacting_atoms_map = ReactiveCenterAnalyzer.get_reacting_atoms_map_numbers(
        sample_reaction
    )
    assert len(reacting_atoms_map) == 4
    assert all(isinstance(atom, cif.Chem.Atom) for atom in reacting_atoms_map.values())
    assert set(reacting_atoms_map.keys()) == {1, 2, 3, 4}


def test_check_hydrogenation():
    # Create two atoms with different number of hydrogens
    mol1 = cif.Chem.MolFromSmiles("C")
    mol2 = cif.Chem.MolFromSmiles("CC")

    atom1 = mol1.GetAtomWithIdx(0)
    atom2 = mol2.GetAtomWithIdx(0)

    result = ReactiveCenterAnalyzer.check_hydrogenation(atom2, atom1)
    assert result == (0, -1)  # Atom index 0, delta_h = -1

    # Test with no change in hydrogenation
    result = ReactiveCenterAnalyzer.check_hydrogenation(atom1, atom1)
    assert result is None


def test_get_bond_info(sample_reaction):
    reactants = sample_reaction.GetReactants()
    product = sample_reaction.GetProducts()[0]

    map_nr = 2
    for reactant in reactants:
        if r_atom := next(
            (atom for atom in reactant.GetAtoms() if atom.GetAtomMapNum() == map_nr),
            None,
        ):
            break
    p_atom = next(
        (atom for atom in product.GetAtoms() if atom.GetAtomMapNum() == map_nr), None
    )

    new_bonds, changed_bonds = ReactiveCenterAnalyzer.get_bond_info(
        r_atom, reactant, p_atom, product, set()
    )

    assert len(new_bonds) == 1  # New C-Cl bond
    assert len(changed_bonds) == 1  # Changed C=C to C-C bond

    new_bond = list(new_bonds)[0]
    changed_bond = list(changed_bonds)[0]

    assert isinstance(new_bond, BondInfo)
    assert isinstance(changed_bond, BondInfo)


def test_get_mapped_neighbors():
    mol = cif.Chem.MolFromSmiles("CC(=O)O")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)  # Set map numbers 1, 2, 3, 4

    central_atom = mol.GetAtomWithIdx(1)  # The central carbon
    neighbors = ReactiveCenterAnalyzer.get_mapped_neighbors(central_atom)

    assert len(neighbors) == 3  # Should have 3 neighbors
    assert set(sorted(neighbors.keys())) == {
        (2, 3),
        (2, 4),
        (1, 2),
    }  # Map number tuples
    assert all(isinstance(v, tuple) and len(v) == 2 for v in neighbors.values())


@pytest.fixture
def sample_chemical_equation():
    smiles = "[BrH:4].[CH2:3]=[CH2:2].[ClH:1]>>[Cl:1][CH2:2][CH2:3][Br:4]"
    ce_constructor = ChemicalEquationConstructor("smiles", "r_p")
    ce = ce_constructor.build_from_reaction_string(smiles, "smiles")
    return ce


@pytest.fixture
def sample_desired_product(sample_chemical_equation):
    # Create a sample Molecule object for the desired product
    return sample_chemical_equation.get_products()[0]


def test_disconnection_builder_initialization(
    sample_chemical_equation, sample_desired_product
):
    builder = DisconnectionBuilder(sample_chemical_equation, sample_desired_product)
    assert isinstance(builder.chemical_equation, ChemicalEquation)
    assert isinstance(builder.desired_product, Molecule)
    assert isinstance(builder.disconnection, Disconnection)
    assert isinstance(builder.reactive_center, ReactiveCenter)
    assert isinstance(builder.analyzer, ReactiveCenterAnalyzer)


def test_identify_reactive_center(sample_chemical_equation, sample_desired_product):
    builder = DisconnectionBuilder(sample_chemical_equation, sample_desired_product)
    builder._identify_reactive_center()
    assert len(builder.reactive_center.reactive_atoms) == 4
    assert len(builder.product_reactive_atoms) == 4


@patch(
    "linchemin.cheminfo.model_constructors.disconnection_constructor.DisconnectionBuilder._get_reactant_for_atom_transformation"
)
@patch(
    "linchemin.cheminfo.model_constructors.disconnection_constructor.DisconnectionBuilder._get_reactant_and_product_atoms"
)
@patch(
    "linchemin.cheminfo.model_constructors.disconnection_constructor.DisconnectionBuilder._check_and_add_hydrogenation"
)
@patch(
    "linchemin.cheminfo.model_constructors.disconnection_constructor.DisconnectionBuilder._check_and_add_bonds"
)
def test_process_atom_transformation(
    mock_bonds,
    mock_hydro_check,
    mock_get_atoms,
    mock_r_t,
    sample_chemical_equation,
    sample_desired_product,
):
    builder = DisconnectionBuilder(sample_chemical_equation, sample_desired_product)

    mock_r_t.return_value = MagicMock(spec=cif.Mol)

    mock_reactant_atom = Mock()
    mock_product_atom = Mock()
    mock_get_atoms.return_value = (mock_reactant_atom, mock_product_atom)

    builder._identify_reactive_center()
    atom_transformation = next(
        iter(builder.chemical_equation.mapping.atom_transformations)
    )
    builder._process_atom_transformation(atom_transformation, set())

    mock_r_t.assert_called()
    mock_get_atoms.assert_called()
    mock_hydro_check.assert_called()
    mock_bonds.assert_called()


def test_check_and_add_hydrogenation(sample_chemical_equation, sample_desired_product):
    builder = DisconnectionBuilder(sample_chemical_equation, sample_desired_product)
    builder._identify_reactive_center()
    # Mock product_atom and reactant_atom
    product_atom = MagicMock(specs=cif.Atom)
    product_atom.GetTotalNumHs.return_value = 2
    product_atom.GetIdx.return_value = 5
    reactant_atom = MagicMock(specs=cif.Atom)
    reactant_atom.GetTotalNumHs.return_value = 0

    builder._check_and_add_hydrogenation(product_atom, reactant_atom)
    # Assert that the correct value was added to the set
    assert (5, 2) in builder.reactive_center.hydrogenated_atoms


@patch(
    "linchemin.cheminfo.model_constructors.disconnection_constructor.ReactiveCenterAnalyzer.get_bond_info"
)
def test_check_and_add_bonds(
    mock_get_bond_info, sample_chemical_equation, sample_desired_product
):
    new_bond = BondInfo(product_atoms=(1, 2), product_bond=0)
    changed_bond = BondInfo(product_atoms=(3, 4), product_bond=1)
    mock_get_bond_info.return_value = ({new_bond}, {changed_bond})

    builder = DisconnectionBuilder(sample_chemical_equation, sample_desired_product)
    builder._identify_reactive_center()
    # Mock reactant_atom, reactant, and product_atom
    reactant_atom = builder.chemical_equation.get_reactants()[
        0
    ].rdmol_mapped.GetAtomWithIdx(0)
    reactant = builder.chemical_equation.get_reactants()[0]
    product_atom = builder.desired_product.rdmol_mapped.GetAtomWithIdx(0)
    builder._check_and_add_bonds(reactant_atom, reactant, product_atom, set())

    # Assert that get_bond_info was called
    mock_get_bond_info.assert_called()
    assert new_bond in builder.reactive_center.new_bonds
    assert changed_bond in builder.reactive_center.changed_bonds


def test_get_reactive_atom_transformations_for_desired_product(
    sample_chemical_equation, sample_desired_product
):
    builder = DisconnectionBuilder(sample_chemical_equation, sample_desired_product)
    builder._identify_reactive_center()
    transformations = builder._get_reactive_atom_transformations_for_desired_product()
    assert len(transformations) == 4
    assert len({t.product_uid for t in transformations}) == 1


def test_get_reactant_for_atom_transformation(
    sample_chemical_equation, sample_desired_product
):
    builder = DisconnectionBuilder(sample_chemical_equation, sample_desired_product)
    builder._identify_reactive_center()
    atom_transformation = next(
        iter(builder.chemical_equation.mapping.atom_transformations)
    )
    reactant = builder._get_reactant_for_atom_transformation(atom_transformation)
    assert isinstance(reactant, Molecule)


def test_get_reactant_and_product_atoms(
    sample_chemical_equation, sample_desired_product
):
    builder = DisconnectionBuilder(sample_chemical_equation, sample_desired_product)
    builder._identify_reactive_center()
    atom_transformation = next(
        iter(builder.chemical_equation.mapping.atom_transformations)
    )
    reactant_atom, product_atom = builder._get_reactant_and_product_atoms(
        atom_transformation
    )
    assert isinstance(reactant_atom, cif.Atom)
    assert reactant_atom.GetAtomMapNum() == atom_transformation.map_num
    assert isinstance(product_atom, cif.Atom)
    assert product_atom.GetAtomMapNum() == atom_transformation.map_num
    assert product_atom.GetIdx() == atom_transformation.prod_atom_id


def test_populate_disconnection(sample_chemical_equation, sample_desired_product):
    builder = DisconnectionBuilder(sample_chemical_equation, sample_desired_product)
    builder._identify_reactive_center()
    builder._populate_disconnection()
    assert builder.disconnection.molecule == sample_desired_product
    assert builder.disconnection.rdmol is not None
    assert sorted(builder.disconnection.reacting_atoms) == [0, 1, 2, 3]
    assert builder.disconnection.new_bonds == [(0, 2), (1, 3)]
    assert builder.disconnection.modified_bonds == [(2, 3)]
    assert builder.disconnection.hydrogenated_atoms == [(0, -1), (1, -1)]
    assert sample_desired_product == sample_chemical_equation.get_products()[0]


def test_build(sample_chemical_equation, sample_desired_product):
    builder = DisconnectionBuilder(sample_chemical_equation, sample_desired_product)
    disconnection = builder.build()
    assert isinstance(disconnection, Disconnection)


def test_disconnection_equality_for_equivalent_synthons():
    test_set = {  # fully balanced amide formation from carboxylic acid and amine
        "rnx_1": "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4]",
        # fully balanced amide formation from acyl chloride and amine (same disconnection as rnx_1)
        "rnx_2": "[CH3:1][C:2]([Cl:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4]",
    }

    ce_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    disconnections = {}
    for name, smiles in test_set.items():
        ce = ce_constructor.build_from_reaction_string(smiles, "smiles")
        disconnections[name] = DisconnectionBuilder(
            chemical_equation=ce, desired_product=ce.get_products()[0]
        ).build()
    # two reactions giving the same product from equivalent reactants
    # at synthon level have the same disconnection
    assert disconnections["rnx_1"] == disconnections["rnx_2"]


def test_disconnection_inequality_for_regioisomers():
    regioisomers = {  # aromatic addition: regioisomer 1
        "rnx_1": "[BrH:8].[ClH:1].[cH:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1>>[Cl:1][c:2]1[cH:3][cH:4][cH:5][cH:6][c:7]1[Br:8]",
        # aromatic addition: regioisomer 2
        "rnx_2": "[BrH:6].[ClH:1].[cH:2]1[cH:3][cH:4][cH:5][cH:7][cH:8]1>>[Cl:1][c:2]1[cH:3][cH:4][c:5]([Br:6])[cH:7][cH:8]1",
    }

    ce_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    disconnections = {}
    for name, smiles in regioisomers.items():
        ce = ce_constructor.build_from_reaction_string(smiles, "smiles")
        disconnections[name] = DisconnectionBuilder(
            chemical_equation=ce, desired_product=ce.get_products()[0]
        ).build()
    # two reactions with the same reactants and different
    # regioisomers as products have different disconnections
    assert disconnections["rnx_1"] != disconnections["rnx_2"]


def test_disconnection_for_deprotections():
    deprotections = {  # double alcohol deprotection (2*ester -> 2*alcohol)
        "rxn_1": "C[O:1][CH2:2][CH2:3][CH2:4][O:5]C>>[OH:5][CH2:4][CH2:3][CH2:2][OH:1]",
        # amine deprotection: same product as rxn_11 but different disconnection
        "rxn_2": "[CH3:1][CH2:2][NH:3]C(O)=O>>[CH3:1][CH2:2][NH2:3]",
    }
    ce_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    disconnections = {}
    for name, smiles in deprotections.items():
        ce = ce_constructor.build_from_reaction_string(smiles, "smiles")
        disconnections[name] = DisconnectionBuilder(
            chemical_equation=ce, desired_product=ce.get_products()[0]
        ).build()
    # a disconnection is created for deprotection reactions
    for name, disconnection in disconnections.items():
        # a disconnection is created for deprotection reactions
        assert disconnection
        # reacting atoms are identified
        assert len(disconnection.reacting_atoms) > 0, f"No reacting atoms for {name}"
        # hydrogenated atoms are identified
        assert (
            len(disconnection.hydrogenated_atoms) > 0
        ), f"No hydrogenated atoms for {name}"


def test_disconnection_for_hydrogenation():
    hydrogentations = {  # ketone reduction to alcohol (two hydrogenated atoms)
        "rxn_1": "[CH3:1][C:2](=[O:4])[CH3:3]>>[CH3:1][CH:2]([CH3:3])[OH:4]",
        # double hydrogenation of a -C#N leading to two H added on N and two on C
        "rxn_2": "[CH3:3][C:2]#[N:1]>>[CH3:3][CH2:2][NH2:1]",
        # single hydrogenation of a -C#N leading to 1 H added on N and one on C
        "rxn_3": "[CH3:3][C:2]#[N:1]>>[CH3:3][CH:2]=[NH:1]",
    }
    ce_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    disconnections = {}
    for name, smiles in hydrogentations.items():
        ce = ce_constructor.build_from_reaction_string(smiles, "smiles")
        disconnections[name] = DisconnectionBuilder(
            chemical_equation=ce, desired_product=ce.get_products()[0]
        ).build()
    for name, disconnection in disconnections.items():
        assert disconnection
        assert len(disconnection.reacting_atoms) > 0, f"No reacting atoms for {name}"
        assert (
            len(disconnection.hydrogenated_atoms) > 0
        ), f"No hydrogenated atoms for {name}"
    assert len(disconnections["rxn_1"].hydrogenated_atoms) == 2
    assert len(disconnections["rxn_2"].hydrogenated_atoms) == 2
    for hydrogenated_atoms in disconnections["rxn_2"].hydrogenated_atoms:
        assert hydrogenated_atoms[1] == 2
    for hydrogenated_atoms in disconnections["rxn_2"].hydrogenated_atoms:
        assert hydrogenated_atoms[1] == 2
    assert len(disconnections["rxn_3"].hydrogenated_atoms) == 2
    for hydrogenated_atoms in disconnections["rxn_3"].hydrogenated_atoms:
        assert hydrogenated_atoms[1] == 1
    assert disconnections["rxn_3"] != disconnections["rxn_2"]


def test_create_disconnection(sample_chemical_equation, sample_desired_product):
    disconnection = create_disconnection(
        sample_chemical_equation, sample_desired_product
    )
    assert disconnection is not None
