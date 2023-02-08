from linchemin.cheminfo.template import (
    Pattern,
    PatternConstructor,
    Template,
    TemplateConstructor,
    rdchiral_extract_template,
)
from linchemin.utilities import create_hash


def test_pattern_creation():
    test_set = [
        {
            "name": "pattern_1",
            "smarts": "[NH2;D1;+0:4]-[C;H0;D3;+0:1](=[O;D1;H0:2])-[c:3]",
            "expected": {},
        },
        {
            "name": "pattern_2",
            "smarts": "[CH3:6][NH:5][C:2]([CH3:1])=[O:4]",
            "expected": {},
        },
        {
            "name": "pattern_3",
            "smarts": "[CH3:1][c:2]1[cH:3][cH:4][cH:5][cH:6][n:7]1",
            "expected": {},
        },
    ]

    pc = PatternConstructor()
    for item in test_set:
        pattern = pc.build_from_molecule_string(
            molecule_string=item.get("smarts"), inp_fmt="smarts"
        )
        print(f"\n{item.get('name')} {pattern.to_dict()}")
        assert pattern


def test_run_rdchiral_wrapper_on_list_of_reactions():
    list_input = [
        {
            "reaction_id": 1,
            "reaction_string": "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "inp_fmt": "smiles",
        },
        {
            "reaction_id": 1,
            "reaction_string": "[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]>>[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]",
            "inp_fmt": "smiles",
        },
    ]

    list_output = [rdchiral_extract_template(**item) for item in list_input]
    expected_list_output = [
        {
            "products": "[C;D1;H3:1]-[C;H0;D3;+0:2](=[O;D1;H0:3])-[NH;D2;+0:6]-[C;D1;H3:5].[OH2;D0;+0:4]",
            "reactants": "[C;D1;H3:1]-[C;H0;D3;+0:2](=[O;D1;H0:3])-[OH;D1;+0:4].[C;D1;H3:5]-[NH2;D1;+0:6]",
            "reaction_smarts": "[C;D1;H3:1]-[C;H0;D3;+0:2](=[O;D1;H0:3])-[NH;D2;+0:6]-[C;D1;H3:5].[OH2;D0;+0:4]>>[C;D1;H3:1]-[C;H0;D3;+0:2](=[O;D1;H0:3])-[OH;D1;+0:4].[C;D1;H3:5]-[NH2;D1;+0:6]",
            "intra_only": False,
            "dimer_only": False,
            "reaction_id": 1,
            "necessary_reagent": "",
        },
        {
            "products": "[C;D1;H3:1]-[C;H0;D3;+0:2](=[O;D1;H0:3])-[OH;D1;+0:6].[C;D1;H3:5]-[NH2;D1;+0:4]",
            "reactants": "[C;D1;H3:1]-[C;H0;D3;+0:2](=[O;D1;H0:3])-[NH;D2;+0:4]-[C;D1;H3:5].[OH2;D0;+0:6]",
            "reaction_smarts": "[C;D1;H3:1]-[C;H0;D3;+0:2](=[O;D1;H0:3])-[OH;D1;+0:6].[C;D1;H3:5]-[NH2;D1;+0:4]>>[C;D1;H3:1]-[C;H0;D3;+0:2](=[O;D1;H0:3])-[NH;D2;+0:4]-[C;D1;H3:5].[OH2;D0;+0:6]",
            "intra_only": False,
            "dimer_only": False,
            "reaction_id": 1,
            "necessary_reagent": "",
        },
    ]

    for a, b in zip(list_output, expected_list_output):
        assert a == b


def test_template_creation():
    print("IMPLEMENT: test_template_creation")
    test_set = [
        {
            "name": "rnx_1",
            "smiles": "[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]>>[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]",
            "expected": {},
        },
        {
            "name": "rnx_2",
            "smiles": "[CH3:6][NH:5][C:2]([CH3:1])=[O:4].[OH2:3]>>[CH3:1][C:2]([OH:3])=[O:4].[CH3:6][NH2:5]",
            "expected": {},
        },
    ]

    for item in test_set:
        tc = TemplateConstructor()
        template = tc.build_from_reaction_string(
            reaction_string=item.get("smiles"), inp_fmt="smiles"
        )
        assert isinstance(template, Template)


def test_template_hashing():
    reactions = {
        0: {
            "smiles": "[CH3:1][NH2:2].O[C:3]([CH3:4])=[O:5]>O>[CH3:1][NH:2][C:3]([CH3:4])=[O:5]"
        },
        1: {
            "smiles": "O[C:3]([CH3:4])=[O:5].[CH3:1][NH2:2]>O>[CH3:1][NH:2][C:3]([CH3:4])=[O:5]"
        },
        2: {"smiles": ">>[CH3:1][NH:2][C:3]([CH3:4])=[O:5]"},
        3: {
            "smiles": "[CH3:1][NH2:2].O[C:3]([CH3:4])=[O:5]>>[CH3:1][NH:2][C:3]([CH3:4])=[O:5]"
        },
        4: {"smiles": "[CH3:1][NH2:2].O[C:3]([CH3:4])=[O:5]>>"},
        5: {
            "smiles": "[CH3:1][NH2:2].O[C:3]([CH3:4])=[O:5]>>[CH3:1][NH:2][C:3]([CH3:4])=[O:5]"
        },
        6: {
            "smiles": "[CH3:1][NH:2][C:3]([CH3:4])=[O:5]>>[CH3:1][NH2:2].O[C:3]([CH3:4])=[O:5]"
        },
        7: {
            "smiles": "[CH3:1][NH2:2].[C:3]([CH3:4])(=[O:5])[OH:6]>>[CH3:1][NH:2][C:3]([CH3:4])=[O:5].[OH2:6]"
        },
        8: {
            "smiles": "[C:3]([CH3:4])(=[O:5])[OH:6].[CH3:1][NH2:2]>>[OH2:6].[CH3:1][NH:2][C:3]([CH3:4])=[O:5]"
        },
        9: {
            "smiles": "[CH3:1][NH:2][C:3]([CH3:4])=[O:5]>O>[CH3:1][NH2:2].O[C:3]([CH3:4])=[O:5]"
        },
    }
    # initialize the constructor
    template_constructor = TemplateConstructor(identity_property_name="smarts")
    results = {}
    for k, v in reactions.items():
        if template := template_constructor.build_from_reaction_string(
            reaction_string=v.get("smiles"), inp_fmt="smiles"
        ):
            h = template.hash_map
        else:
            h = None
        results[k] = h
        # print(k, h)

    # the hashes are calculated and have a non-null (None) value
    assert results.get(0).get("reactants")
    assert results.get(0).get("reagents")
    assert results.get(0).get("products")
    assert results.get(0).get("r_p")
    assert results.get(0).get("r_r_p")
    assert results.get(0).get("u_r_p")
    assert results.get(0).get("u_r_r_p")

    # the reactant hash is insensitive to the input order of reactants (reaction canonicalization OK)
    assert results.get(0).get("reactants") == results.get(1).get("reactants")

    # the product hash is insensitive to the input order of products (reaction canonicalization OK)
    assert results.get(7).get("products") == results.get(8).get("products")

    # the machinery does break when the reactants are missing: Template is None
    assert results.get(2) is None

    # the machinery does not break when the agents are missing
    assert results.get(3).get("reagents")

    # the machinery does break when the products are missing: Template is None
    assert results.get(4) is None

    # reagents are happily ignored

    # there is a special hash for missing roles (it is the hash of an empty string)
    assert results.get(3).get("reagents") == create_hash("")

    # the reactant and products hashes are conserved even when the reagents are missing
    assert results.get(0).get("reactants") == results.get(5).get("reactants")
    assert results.get(0).get("products") == results.get(5).get("products")

    # the base r>p hash is conserved if the agents are missing in one reaction
    assert results.get(0).get("r_p") == results.get(5).get("r_p")

    # the full r>a>p hash is conserved  if the reagents are missing in one reaction (reagents are ignored!!)
    assert results.get(0).get("r_r_p") == results.get(5).get("r_r_p")

    # the base r>>p hash is not conserved if the reaction is reversed
    assert results.get(0).get("r_p") != results.get(6).get("r_p")

    # the full r>a>p hash is not conserved if the reaction is reversed
    assert results.get(0).get("r_r_p") != results.get(6).get("r_r_p")

    # the reversible base r<>p hash is not conserved if the reaction is reversed (this comes from rdchiral teplate extraction)
    # in some special cases it might be true, but it not necessarily is
    assert results.get(0).get("u_r_p") != results.get(9).get("u_r_p")
    assert results.get(3).get("u_r_p") != results.get(6).get("u_r_p")

    # the reversible full r<a>p hash is not conserved if the reaction is reversed (this comes from rdchiral teplate extraction)
    # in some special cases it might be true, but it not necessarily is
    assert results.get(0).get("u_r_r_p") != results.get(9).get("u_r_r_p")
    assert results.get(3).get("u_r_r_p") != results.get(6).get("u_r_r_p")
