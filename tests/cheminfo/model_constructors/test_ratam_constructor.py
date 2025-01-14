import math

import pytest

import linchemin.cheminfo.functions as cif
from linchemin.cheminfo.constructors import MoleculeConstructor
from linchemin.cheminfo.model_constructors.ratam_constructor import (
    AtomTransformation,
    BadMapping,
    RatamBuilder,
    ReactionComponents,
    create_ratam,
)
from linchemin.cheminfo.models import Ratam


@pytest.fixture
def sample_reaction_components():
    mol_constructor = MoleculeConstructor("smiles")
    r1 = mol_constructor.build_from_molecule_string("[BrH:4]", "smiles")
    r1.uid = "R1"
    r2 = mol_constructor.build_from_molecule_string("[CH2:3]=[CH2:2]", "smiles")
    r2.uid = "R2"
    rg = mol_constructor.build_from_molecule_string("O", "smiles")
    rg.uid = "Rg"
    p1 = mol_constructor.build_from_molecule_string("[CH3:2][CH2:3][Br:4]", "smiles")
    p1.uid = "P1"
    rc = ReactionComponents()
    rc.reactants.extend([r1, r2])
    rc.reagents.extend([rg])
    rc.products.extend([p1])
    return rc


@pytest.fixture
def sample_desired_product(sample_reaction_components):
    # Create a sample Molecule object for the desired product
    return sample_reaction_components.products[0]


def test_create_initial_map_info(sample_reaction_components, sample_desired_product):
    builder = RatamBuilder(sample_reaction_components, sample_desired_product)
    builder._create_initial_map_info()

    assert "products" in builder.ratam.full_map_info
    assert sample_desired_product.uid in builder.ratam.full_map_info["products"]
    assert builder.ratam.full_map_info["products"][sample_desired_product.uid] == [
        {0: 2, 1: 4, 2: 3}
    ]

    assert len(builder.precursors) == 3
    assert next(p for p in builder.precursors if "R1" in p) == {"R1": [{0: 4}]}
    assert next(p for p in builder.precursors if "Rg" in p) == {"Rg": [{0: 0}]}


def test_reassign_roles(sample_reaction_components, sample_desired_product):
    builder = RatamBuilder(sample_reaction_components, sample_desired_product)
    builder._create_initial_map_info()
    builder._reassign_roles()

    assert "R2" in builder.ratam.full_map_info["reactants"]
    assert "Rg" in builder.ratam.full_map_info["reagents"]


def test_mapping_sanity_check_valid(sample_reaction_components, sample_desired_product):
    builder = RatamBuilder(sample_reaction_components, sample_desired_product)
    builder._create_initial_map_info()
    builder._reassign_roles()

    # This should not raise an exception
    builder._mapping_sanity_check()


def test_sanity_check_invalid_mapping(
    sample_reaction_components, sample_desired_product
):
    sample_reaction_components.reactants.append(
        MoleculeConstructor("smiles").build_from_molecule_string("[BrH:2]", "smiles")
    )
    builder = RatamBuilder(sample_reaction_components, sample_desired_product)
    builder._create_initial_map_info()

    with pytest.raises(BadMapping):
        builder._reassign_roles()


def test_create_atom_transformations(
    sample_reaction_components, sample_desired_product
):
    builder = RatamBuilder(sample_reaction_components, sample_desired_product)
    builder._create_initial_map_info()
    builder._reassign_roles()
    builder._create_atom_transformations()

    expected_transformations = {
        AtomTransformation(
            product_uid="P1",
            reactant_uid="R2",
            prod_atom_id=0,
            react_atom_id=1,
            map_num=2,
        ),
        AtomTransformation(
            product_uid="P1",
            reactant_uid="R2",
            prod_atom_id=2,
            react_atom_id=0,
            map_num=3,
        ),
        AtomTransformation(
            product_uid="P1",
            reactant_uid="R1",
            prod_atom_id=1,
            react_atom_id=0,
            map_num=4,
        ),
    }

    assert builder.ratam.atom_transformations == expected_transformations


def test_analyze_reactant_unmapped_atoms(
    sample_reaction_components, sample_desired_product
):
    reactant_w_unmapped = MoleculeConstructor("smiles").build_from_molecule_string(
        "[CH2:3]=C", "smiles"
    )
    reactant_w_unmapped.uid = "Ru"
    sample_reaction_components.reactants.append(reactant_w_unmapped)
    sample_reaction_components.reactants[:] = [
        item for item in sample_reaction_components.reactants if item.uid != "R2"
    ]
    builder = RatamBuilder(sample_reaction_components, sample_desired_product)
    builder._create_initial_map_info()
    builder._reassign_roles()
    builder._create_atom_transformations()
    builder._analyze_unmapped_atoms()

    assert builder.ratam.reactants_unmapped_atoms_info["unmapped_atoms"] == {
        "Ru": [{1}]
    }
    assert builder.ratam.reactants_unmapped_atoms_info["fraction"] == round(1 / 3, 2)


def test_analyze_product_unmapped_atoms(
    sample_reaction_components, sample_desired_product
):
    prod_w_unmapped = MoleculeConstructor("smiles").build_from_molecule_string(
        "[CH3:2][CH2:3]Br", "smiles"
    )
    prod_w_unmapped.uid = "Pu"
    sample_reaction_components.products.append(prod_w_unmapped)
    sample_reaction_components.products[:] = [
        item for item in sample_reaction_components.products if item.uid != "P1"
    ]
    sample_desired_product.uid = "Pu"
    builder = RatamBuilder(sample_reaction_components, sample_desired_product)
    builder._create_initial_map_info()
    builder._reassign_roles()
    builder._create_atom_transformations()
    builder._analyze_unmapped_atoms()
    assert builder.ratam.desired_product_unmapped_atoms_info["unmapped_atoms"] == {
        "Pu": [{1}]
    }
    assert builder.ratam.desired_product_unmapped_atoms_info["fraction"] == round(
        1 / 3, 2
    )


def test_build(sample_reaction_components, sample_desired_product):
    builder = RatamBuilder(sample_reaction_components, sample_desired_product)
    ratam = builder.build()

    assert isinstance(ratam, Ratam)
    assert "reactants" in ratam.full_map_info
    assert "reagents" in ratam.full_map_info
    assert "products" in ratam.full_map_info
    assert len(ratam.atom_transformations) > 0
    assert "unmapped_atoms" in ratam.reactants_unmapped_atoms_info
    assert "unmapped_atoms" in ratam.desired_product_unmapped_atoms_info


def test_create_ratam(sample_reaction_components, sample_desired_product):
    ratam = create_ratam(sample_reaction_components, sample_desired_product)
    assert isinstance(ratam, Ratam)


test_set = [
    # All initial reactants are actual reactants
    {
        "name": "rnx_1",
        "smiles": "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
        "expected": {
            "reactants": ["CC(=O)O", "CN"],
            "reagents": [],
            "products": ["O", "CNC(C)=O"],
        },
    },
    # An initial reagents is actually a reactant
    {
        "name": "rnx_2",
        "smiles": "[CH3:1][C:2]([OH:3])=[O:4]>[CH3:6][NH2:5]>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
        "expected": {
            "reactants": ["CC(=O)O", "CN"],
            "reagents": [],
            "products": ["O", "CNC(C)=O"],
        },
    },
    # A reagent is recognized as such
    {
        "name": "rnx_3",
        "smiles": "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>CO>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
        "expected": {
            "reactants": ["CC(=O)O", "CN"],
            "reagents": ["CO"],
            "products": ["O", "CNC(C)=O"],
        },
    },
    # The same molecule appears twice, once as reactant and once as reagent
    {
        "name": "rnx_4",
        "smiles": "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>CN>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
        "expected": {
            "reactants": ["CC(=O)O", "CN"],
            "reagents": ["CN"],
            "products": ["O", "CNC(C)=O"],
        },
    },
    # Bad mapping: the same map number is used more than twice
    {
        "name": "rnx_5",
        "smiles": "[CH3:3][C:2]([OH:1])=[O:4].[CH3:3][NH2:5]>CN>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:1]",
        "expected": {
            "reactants": ["CC(=O)O", "CN"],
            "reagents": ["CN"],
            "products": ["CNC(C)=O", "O"],
        },
    },
    # A reactant is missing and all atoms in the desired product
    # have a map number ("complete" option in namerxn)
    {
        "name": "rnx_6",
        "smiles": "[CH3:1][C:2]([OH:3])=[O:4]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4]",
        "expected": {
            "reactants": ["CC(=O)O"],
            "reagents": [],
            "products": ["CNC(C)=O"],
        },
    },
    # A reactant is missing and not all atoms in the desired
    # product have a map number ("matched" option in namerxn)
    {
        "name": "rnx_7",
        "smiles": "[CH3:1][C:2]([OH:3])=[O:4]>>[CH3][NH][C:2]([CH3:1])=[O:4]",
        "expected": {
            "reactants": ["CC(=O)O"],
            "reagents": [],
            "products": ["CNC(C)=O"],
        },
    },
]


@pytest.mark.parametrize("item", test_set, ids=[item["name"] for item in test_set])
def test_ratam_and_role_reassignment(item):
    mol_constructor = MoleculeConstructor(molecular_identity_property_name="smiles")
    rdrxn = cif.rdrxn_from_string(input_string=item["smiles"], inp_fmt="smiles")
    reaction_mols = cif.rdrxn_to_molecule_catalog(rdrxn, mol_constructor)
    catalog = {
        m.uid: m
        for m in set(
            reaction_mols["reactants"]
            + reaction_mols["reagents"]
            + reaction_mols["products"]
        )
    }

    reaction_components = ReactionComponents.from_dict(reaction_mols)
    ratam_constructor = RatamBuilder(reaction_components, reaction_mols["products"][0])

    if item["name"] == "rnx_5":
        with pytest.raises(BadMapping) as excinfo:
            ratam_constructor.build()
        assert "Invalid" in str(
            excinfo.value
        ), f"BadMapping not raised or incorrect error for {item['name']}"
    else:
        ratam = ratam_constructor.build()
        assert ratam, f"Ratam not created for {item['name']}"
        assert ratam.atom_transformations, f"No atom transformations for {item['name']}"

        for role, map_info in ratam.full_map_info.items():
            smiles_list = [m.smiles for uid, m in catalog.items() if uid in map_info]
            assert item["expected"][role] == smiles_list, (
                f"Mismatch in {role} for {item['name']}: \n"
                f"Expected: {item['expected'][role]}\n"
                f"Got: {smiles_list}"
            )

        if item["name"] in ["rnx_6", "rnx_7"]:
            assert math.isclose(
                ratam.desired_product_unmapped_atoms_info["fraction"],
                0.4,
                rel_tol=1e-09,
                abs_tol=1e-09,
            ), f"Incorrect unmapped atoms fraction for {item['name']}"
            unmapped_atoms = next(
                v
                for v in ratam.desired_product_unmapped_atoms_info[
                    "unmapped_atoms"
                ].values()
            )
            assert unmapped_atoms == [
                {0, 3}
            ], f"Incorrect unmapped atoms for {item['name']}: Got {unmapped_atoms}"
