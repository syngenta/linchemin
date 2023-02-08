from linchemin.cheminfo.reaction import ChemicalEquation, ChemicalEquationConstructor
from linchemin.utilities import create_hash


def test_chemical_equation_hashing():
    reactions = {
        0: {"smiles": "CN.CC(O)=O>O>CNC(C)=O"},
        1: {"smiles": "CC(O)=O.CN>O>CNC(C)=O"},
        2: {"smiles": ">>CNC(C)=O"},
        3: {"smiles": "CC(O)=O.CN>>CNC(C)=O"},
        4: {"smiles": "CC(O)=O.CN>>"},
        5: {"smiles": "CN.CC(O)=O>>CNC(C)=O"},
        6: {"smiles": "CNC(C)=O>>CN.CC(O)=O"},
        7: {"smiles": "CN.CC(O)=O>>CNC(C)=O.O"},
        8: {"smiles": "CN.CC(O)=O>>O.CNC(C)=O"},
        9: {"smiles": "CNC(C)=O>O>CN.CC(O)=O"},
    }
    # initialize the constructor
    chemical_equation_constructor = ChemicalEquationConstructor(
        identity_property_name="smiles"
    )
    results = {}
    for k, v in reactions.items():
        chemical_equation = chemical_equation_constructor.build_from_reaction_string(
            reaction_string=v.get("smiles"), inp_fmt="smiles"
        )

        h = chemical_equation.hash_map
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

    # the machinery does not break when the reactants are missing
    assert results.get(2).get("reactants")

    # the machinery does not break when the agents are missing
    assert results.get(3).get("reagents")

    # the machinery does not break when the products are missing
    assert results.get(4).get("products")

    # there is a special hash for missing roles (it is the hash of an empty string)
    assert (
        results.get(2).get("reactants")
        == results.get(3).get("reagents")
        == results.get(4).get("products")
        == create_hash("")
    )

    # the reactant and products hashes are conserved even when the reagents are missing
    assert results.get(0).get("reactants") == results.get(5).get("reactants")
    assert results.get(0).get("products") == results.get(5).get("products")

    # the agent hash is different if the agents are missing
    assert results.get(0).get("reagents") != results.get(5).get("reagents")

    # the base r>p hash is conserved if the agents are missing in one reaction
    assert results.get(0).get("r_p") == results.get(5).get("r_p")

    # the full r>a>p hash is not conserved if the reagents are missing in one reaction
    assert results.get(0).get("r_r_p") != results.get(5).get("r_r_p")

    # the base r>>p hash is not conserved if the reaction is reversed
    assert results.get(0).get("r_p") != results.get(6).get("r_p")

    # the full r>a>p hash is not conserved if the reaction is reversed
    assert results.get(0).get("r_r_p") != results.get(6).get("r_r_p")

    # the reversible base r<>p hash is  conserved if the reaction is reversed
    assert results.get(0).get("u_r_p") == results.get(9).get("u_r_p")
    assert results.get(3).get("u_r_p") == results.get(6).get("u_r_p")

    # the reversible full r<a>p hash is  conserved if the reaction is reversed
    assert results.get(0).get("u_r_r_p") == results.get(9).get("u_r_r_p")
    assert results.get(3).get("u_r_r_p") == results.get(6).get("u_r_r_p")


def test_instantiate_chemical_equation():
    reaction_smiles_input = "CN.CC(O)=O>O>CNC(C)=O"
    chemical_equation_constructor = ChemicalEquationConstructor(
        identity_property_name="smiles"
    )
    chemical_equation = chemical_equation_constructor.build_from_reaction_string(
        reaction_string=reaction_smiles_input, inp_fmt="smiles"
    )
    # assert molecules are canonicalized
    # assert reaction is canonicalized


def test_create_reaction_smiles_from_chemical_equation():
    reaction_smiles_input = "CN.CC(O)=O>O>CNC(C)=O"
    chemical_equation_constructor = ChemicalEquationConstructor(
        identity_property_name="smiles"
    )
    chemical_equation = chemical_equation_constructor.build_from_reaction_string(
        reaction_string=reaction_smiles_input, inp_fmt="smiles"
    )
    reaction_smiles = chemical_equation.build_reaction_smiles()
    assert reaction_smiles


def test_reaction_canonicalization_from_molecules():
    reactions = {
        0: {"smiles": "CN.CC(O)=O>O>CNC(C)=O"},
        1: {"smiles": "CC(O)=O.CN>O>CNC(C)=O"},
        2: {"smiles": ">>CNC(C)=O"},
        3: {"smiles": "CC(O)=O.CN>>CNC(C)=O"},
        4: {"smiles": "CC(O)=O.CN>>"},
        5: {"smiles": "CN.CC(O)=O>>CNC(C)=O"},
        6: {"smiles": "CNC(C)=O>>CN.CC(O)=O"},
        7: {"smiles": "CN.CC(O)=O>>CNC(C)=O.O"},
        8: {"smiles": "CN.CC(O)=O>>O.CNC(C)=O"},
        9: {"smiles": "CNC(C)=O>O>CN.CC(O)=O"},
    }
    # initialize the constructor
    chemical_equation_constructor = ChemicalEquationConstructor(
        identity_property_name="smiles"
    )
    results = {}
    for k, v in reactions.items():
        chemical_equation = chemical_equation_constructor.build_from_reaction_string(
            reaction_string=v.get("smiles"), inp_fmt="smiles"
        )
        results[k] = chemical_equation
        # print(k, h)
    # the reaction smiles is insensitive to the input order of reactants (reaction canonicalization OK)

    assert results.get(0).smiles == results.get(1).smiles


def test_chemical_equation_equality():
    reactions = {
        0: {"smiles": "CN.CC(O)=O>O>CNC(C)=O"},  # R1
        1: {"smiles": "CN.CC(O)=O>O>CNC(C)=O"},  # R1
        2: {"smiles": "NC.CC(O)=O>O>CNC(C)=O"},  # R1
        3: {"smiles": "NC.CC(=O)O>O>CNC(C)=O"},  # R1
        4: {"smiles": "CC(O)=O.CN>O>CNC(C)=O"},  # R1
        5: {
            "smiles": "CC#N.Cl[C:2]([CH3:1])=[O:3].[CH3:4][NH2:5]>>O.[CH3:1][C:2](=[O:3])[NH:5][CH3:4]"
        },  # R2
        6: {
            "smiles": "CC#N.Cl[C:20]([CH3:1])=[O:3].[CH3:4][NH2:50]>>O.[CH3:1][C:20](=[O:3])[NH:50][CH3:4]"
        },
        # R2
        7: {
            "smiles": "Cl[C:20]([CH3:1])=[O:3].CC#N.[CH3:4][NH2:50]>>O.[CH3:1][C:20](=[O:3])[NH:50][CH3:4]"
        },
        # R2
    }

    ces1 = {}

    # initialize the constructor
    chemical_equation_constructor = ChemicalEquationConstructor(
        identity_property_name="smiles"
    )

    for k, v in reactions.items():
        chemical_equation = chemical_equation_constructor.build_from_reaction_string(
            reaction_string=v.get("smiles"), inp_fmt="smiles"
        )
        ces1[k] = chemical_equation

    assert ces1.get(0) == ces1.get(
        1
    )  # same reaction, one reactant has a different smiles: test mol canonicalization
    assert ces1.get(0) == ces1.get(
        2
    )  # same reaction, two reactant have a different smiles: test mol canonicalization
    assert ces1.get(0) == ces1.get(
        3
    )  # same reaction, two reactant have a different smiles: test mol canonicalization
    assert ces1.get(0) == ces1.get(
        4
    )  # same reaction, different reactant ordering: test reaction canonicalization
    assert ces1.get(5) == ces1.get(6)  # same reaction, different atom mapping
    assert ces1.get(5) == ces1.get(7)  # same reaction, different atom mapping,
    # different reactant ordering: test reaction canonicalization


def test_chemical_equation_builder():
    reaction_string_reference = "CC(=O)O.CN.CN>O>CNC(C)=O"

    # initialize the constructor
    cec = ChemicalEquationConstructor(identity_property_name="smiles")

    for reaction_string_test in [
        "CC(=O)O.CN.CN>O>CNC(C)=O",  # expected smiles
        "CC(=O)O.NC.CN>O>CNC(C)=O",  # test molecule canonicalization: change order of atoms in reactant molecules
        "CN.CC(=O)O.CN>O>CNC(C)=O",  # test reaction canonicalization: change order of molecules in reactants
    ]:
        chemical_equation = cec.build_from_reaction_string(
            reaction_string=reaction_string_test, inp_fmt="smiles"
        )
        reaction_string_calculated = chemical_equation.smiles
        assert reaction_string_calculated == reaction_string_reference


def test_chemical_equation_attributes_are_available():
    smiles = "[cH:5]1[cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]2[c:3]([cH:4]1)[C:2](=[O:1])O.[N-:13]=[N+:14]=[N-:15]>C(Cl)Cl.C(=O)(C(=O)Cl)Cl>[cH:5]1[cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]2[c:3]([cH:4]1)[C:2](=[O:1])[N:13]=[N+:14]=[N-:15]"
    # initialize the constructor
    chemical_equation_constructor = ChemicalEquationConstructor(
        identity_property_name="smiles"
    )
    chemical_equation = chemical_equation_constructor.build_from_reaction_string(
        reaction_string=smiles, inp_fmt="smiles"
    )

    disconnection = chemical_equation.disconnection
    template = chemical_equation.template
    assert disconnection
    assert template


def test_chemical_equation_attributes_are_not_available():
    smiles = "CN.CC(O)=O>O>CNC(C)=O"
    chemical_equation_constructor = ChemicalEquationConstructor(
        identity_property_name="smiles"
    )
    chemical_equation = chemical_equation_constructor.build_from_reaction_string(
        reaction_string=smiles, inp_fmt="smiles"
    )
    disconnection = chemical_equation.disconnection
    assert not disconnection
    template = chemical_equation.template
    assert not template
