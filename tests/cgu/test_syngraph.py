import json
import unittest

import linchemin.cheminfo.functions as cif
from linchemin.cgu.syngraph import (
    BipartiteSynGraph,
    MonopartiteMolSynGraph,
    MonopartiteReacSynGraph,
)
from linchemin.cgu.syngraph_operations import merge_syngraph
from linchemin.cgu.translate import translator
from linchemin.cheminfo.constructors import (
    ChemicalEquationConstructor,
    MoleculeConstructor,
)


def test_bipartite_syngraph_instance(az_path):
    """To test that a BipartiteSynGraph instance is correctly generated."""
    syngraph = BipartiteSynGraph()
    assert len(syngraph.graph.keys()) == 0 and len(syngraph.graph.values()) == 0

    graph = json.loads(open(az_path).read())
    syngraph = translator("az_retro", graph[0], "syngraph", out_data_model="bipartite")
    assert len(syngraph.graph.keys()) != 0 and len(syngraph.graph.values()) != 0
    assert syngraph.name is not None
    assert len(syngraph.get_roots()) == 1
    assert len(syngraph.get_leaves()) != 0

    syngraph2 = translator("az_retro", graph[0], "syngraph", out_data_model="bipartite")
    syngraph3 = translator("az_retro", graph[1], "syngraph", out_data_model="bipartite")
    assert syngraph == syngraph2
    assert syngraph3 != syngraph2


def test_add_new_node(ibm1_path):
    """To test that the SynGraph method 'add_node' correctly add new nodes to a SynGraph instance."""
    graph = json.loads(open(ibm1_path).read())
    syngraph = translator("ibm_retro", graph[0], "syngraph", out_data_model="bipartite")
    l1 = len(syngraph.graph)
    new_node = ("new_mol_smiles", ["new>reaction>smiles1", "new>reaction>smiles2"])
    syngraph.add_node(new_node)
    l2 = len(syngraph.graph)
    assert "new_mol_smiles" in syngraph.graph.keys()
    assert l1 != l2


def test_add_existing_node(ibm1_path):
    """To test that if an already existing node is added to a SynGraph instance, the node is not duplicated."""
    graph = json.loads(open(ibm1_path).read())
    syngraph1 = translator(
        "ibm_retro", graph[0], "syngraph", out_data_model="bipartite"
    )
    l1 = len(syngraph1.graph)
    molecule_constructor = MoleculeConstructor(
        molecular_identity_property_name="smiles"
    )

    reactant = molecule_constructor.build_from_molecule_string(
        molecule_string="CCN", inp_fmt="smiles"
    )

    chemical_equation_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    chemical_equation = chemical_equation_constructor.build_from_reaction_string(
        reaction_string="CCN>>CCC(=O)NCC", inp_fmt="smiles"
    )
    reaction = chemical_equation

    existing_node = (reactant, [reaction])
    syngraph1.add_node(existing_node)
    l2 = len(syngraph1.graph)
    assert l1 == l2


def test_add_existing_node_with_new_connections(ibm1_path):
    """To test that new connections for an existing node are correctly added, without duplicates."""
    graph = json.loads(open(ibm1_path).read())
    syngraph = translator("ibm_retro", graph[0], "syngraph", out_data_model="bipartite")
    l1 = len(syngraph.graph)

    molecule_constructor = MoleculeConstructor(
        molecular_identity_property_name="smiles"
    )

    reactant = molecule_constructor.build_from_molecule_string(
        molecule_string="CCN", inp_fmt="smiles"
    )

    chemical_equation_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    chemical_equation = chemical_equation_constructor.build_from_reaction_string(
        reaction_string="CCN>>CCC(=O)NCC", inp_fmt="smiles"
    )
    reaction = chemical_equation

    node = (reactant, [reaction, "C1CCOC1.CCOC(=O)CC.CCN>>CCC(=O)NCC"])
    syngraph.add_node(node)

    l2 = len(syngraph.graph)
    assert l1 == l2
    assert "C1CCOC1.CCOC(=O)CC.CCN>>CCC(=O)NCC" in syngraph[reactant]


def test_syngraph_name(az_path):
    """To test that the name attribute of a SynGraph instance is correctly assigned."""
    graph_az = json.loads(open(az_path).read())
    syngraph = translator(
        "az_retro", graph_az[1], "syngraph", out_data_model="bipartite"
    )

    assert "az" in syngraph.name


def test_monopartite_syngraph(ibm1_path):
    """To test that a MonopartiteMolSynGraph object is correctly generated"""
    graph_ibm = json.loads(open(ibm1_path).read())
    mp_syngraph = translator(
        "ibm_retro", graph_ibm[5], "syngraph", out_data_model="monopartite_molecules"
    )
    molecule_constructor = MoleculeConstructor(
        molecular_identity_property_name="smiles"
    )
    mol1 = molecule_constructor.build_from_molecule_string(
        molecule_string="CCC(=O)Cl", inp_fmt="smiles"
    )
    mol2 = molecule_constructor.build_from_molecule_string(
        molecule_string="CCNC(=O)CC", inp_fmt="smiles"
    )

    assert mol1 in mp_syngraph.get_leaves()
    assert mol2 in mp_syngraph.get_roots()
    assert len(mp_syngraph.graph) == len(mp_syngraph.get_unique_nodes())


def test_reaction_monopartite(az_path):
    graph_az = json.loads(open(az_path).read())
    mp_reac_syngraph = translator(
        "az_retro", graph_az[0], "syngraph", out_data_model="monopartite_reactions"
    )

    chemical_equation_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    ce_root = chemical_equation_constructor.build_from_reaction_string(
        reaction_string="Cc1cccc(C)c1NCC(=O)Nc1ccc(-c2ncon2)cc1.O=C("
        "O)C1CCS(=O)(=O)CC1>>Cc1cccc(C)c1N(CC(=O)Nc1ccc("
        "-c2ncon2)cc1)C(=O)C1CCS(=O)(=O)CC1",
        inp_fmt="smiles",
    )
    assert ce_root in mp_reac_syngraph.get_roots()
    mol_roots = mp_reac_syngraph.get_molecule_roots()
    assert "Cc1cccc(C)c1N(CC(=O)Nc1ccc(-c2ncon2)cc1)C(=O)C1CCS(=O)(=O)CC1" in [
        m.smiles for m in mol_roots
    ]
    mol_leaves = mp_reac_syngraph.get_molecule_leaves()
    leaves_smiles = sorted(
        [
            "O=C(O)C1CCS(=O)(=O)CC1",
            "Cc1cccc(C)c1NCC(=O)O",
            "Nc1ccc(-c2ncon2)cc1",
        ]
    )
    assert sorted([m.smiles for m in mol_leaves]) == leaves_smiles


def test_get_reaction_leaves(az_path):
    """To test the MonopartiteReacSynGraph method 'get_leaves' correctly identifies the leaves (ReactionStep)
    in the graph."""
    graph = json.loads(open(az_path).read())
    syngraphs = [
        translator("az_retro", g, "syngraph", out_data_model="bipartite") for g in graph
    ]
    tree = merge_syngraph(syngraphs)
    mol_roots = tree.get_roots()

    mp_syngraphs = [
        translator("az_retro", g, "syngraph", out_data_model="monopartite_reactions")
        for g in graph
    ]
    mp_tree = merge_syngraph(mp_syngraphs)
    reac_roots = mp_tree.get_roots()
    target = []
    for root in reac_roots:
        prod = root.smiles.split(">>")[-1]
        if prod not in target:
            target.append(prod)
    assert target[0] == mol_roots[0].smiles

    reac_leaves = mp_tree.get_leaves()
    assert len(reac_leaves) == 4


def test_read_dictionary(az_path, ibm1_path):
    # monopartite reactions
    d = [
        {
            "query_id": 0,
            "output_string": "Cc1cccc(C)c1NCC(=O)O.Nc1ccc(-c2ncon2)cc1>>Cc1cccc(C)c1NCC(=O)Nc1ccc(-c2ncon2)cc1",
        },
        {
            "query_id": 1,
            "output_string": "Cc1cccc(C)c1NCC(=O)Nc1ccc(-c2ncon2)cc1.O=C(O)C1CCS(=O)(=O)CC1>>Cc1cccc(C)c1N(CC("
            "=O)Nc1ccc(-c2ncon2)cc1)C(=O)C1CCS(=O)(=O)CC1",
        },
    ]
    syngraph = MonopartiteReacSynGraph(d)
    graph_az = json.loads(open(az_path).read())
    assert syngraph == translator(
        "az_retro", graph_az[0], "syngraph", "monopartite_reactions"
    )

    # bipartite
    d2 = [{"query_id": 0, "output_string": "CCC(=O)Cl.CCN>>CCNC(=O)CC"}]
    graph_ibm = json.loads(open(ibm1_path).read())
    syngraph = BipartiteSynGraph(d2)
    assert syngraph == translator("ibm_retro", graph_ibm[3], "syngraph", "bipartite")

    # monopartite molecules
    d3 = [{"query_id": 0, "output_string": "CCC(=O)Cl.CCN.ClCCl>>CCNC(=O)CC"}]
    mom_syngraph = MonopartiteMolSynGraph(d3)
    assert mom_syngraph == translator(
        "ibm_retro", graph_ibm[4], "syngraph", "monopartite_molecules"
    )
    assert len(mom_syngraph.graph) == len(mom_syngraph.get_unique_nodes())


def test_hashing(ibm2_path):
    graph = json.loads(open(ibm2_path).read())
    syngraph_mpr = translator(
        "ibm_retro", graph[0], "syngraph", "monopartite_reactions"
    )
    # The hash key is created
    assert syngraph_mpr.uid
    uid1 = syngraph_mpr.uid
    chemical_equation_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    ce = chemical_equation_constructor.build_from_reaction_string(
        reaction_string="Cc1cccc(C)c1NCC(=O)Nc1ccc(-c2ncon2)cc1.O=C("
        "O)C1CCS(=O)(=O)CC1>>Cc1cccc(C)c1N(CC(=O)Nc1ccc("
        "-c2ncon2)cc1)C(=O)C1CCS(=O)(=O)CC1",
        inp_fmt="smiles",
    )
    # If the SynGraph instance changes, the hash key is also modified
    syngraph_mpr.add_node((ce, []))
    assert syngraph_mpr.uid != uid1
    # prefixes of the uid indicate the type of SynGraph
    assert syngraph_mpr.uid.startswith("MPR")

    syngraph_mpm = translator(
        "ibm_retro", graph[0], "syngraph", "monopartite_molecules"
    )
    assert syngraph_mpm.uid.startswith("MPM")

    syngraph_mpm = translator("ibm_retro", graph[0], "syngraph", "bipartite")
    assert syngraph_mpm.uid.startswith("BP")


def test_bipartite_iron(az_path):
    graph_az = json.loads(open(az_path).read())
    nx_bp = translator("az_retro", graph_az[0], "networkx", "bipartite")
    syngraph_mpr = translator("networkx", nx_bp, "syngraph", "bipartite")
    syngraph = translator("az_retro", graph_az[0], "syngraph", "bipartite")
    assert syngraph.graph == syngraph_mpr.graph
    assert syngraph.uid == syngraph_mpr.uid
    assert len(syngraph.graph) == len(syngraph.get_unique_nodes())
    nx_bp = translator("az_retro", graph_az[0], "networkx", "bipartite")
    syngraph_mpr = translator("networkx", nx_bp, "syngraph", "monopartite_reactions")
    syngraph = translator("az_retro", graph_az[0], "syngraph", "monopartite_reactions")
    assert syngraph.graph == syngraph_mpr.graph
    assert syngraph.uid == syngraph_mpr.uid


def test_node_removal():
    d = [
        {
            "query_id": 0,
            "output_string": "Cc1cccc(C)c1NCC(=O)O.Nc1ccc(-c2ncon2)cc1>>Cc1cccc(C)c1NCC(=O)Nc1ccc(-c2ncon2)cc1",
        },
        {
            "query_id": 1,
            "output_string": "Cc1cccc(C)c1NCC(=O)Nc1ccc(-c2ncon2)cc1.O=C(O)C1CCS(=O)(=O)CC1>>Cc1cccc(C)c1N(CC("
            "=O)Nc1ccc(-c2ncon2)cc1)C(=O)C1CCS(=O)(=O)CC1",
        },
        {
            "query_id": 2,
            "output_string": "CCOC(=O)CNc1c(C)cccc1C>>Cc1cccc(C)c1NCC(O)=O",
        },
    ]
    syngraph = MonopartiteReacSynGraph(d)
    chemical_equation_constructor = ChemicalEquationConstructor(
        molecular_identity_property_name="smiles",
        chemical_equation_identity_name="r_r_p",
    )
    s = "CCC(=O)Cl.CCN>>CCNC(=O)CC"
    ce_not_present = chemical_equation_constructor.build_from_reaction_string(
        s, "smiles"
    )
    # if the selected node is not present, a warning is raised and the syngraph instance remains unchanged
    with unittest.TestCase().assertLogs("linchemin.cgu.syngraph", level="WARNING"):
        syngraph.remove_node(ce_not_present.uid)
    assert len(syngraph.graph) == 3
    # if the selected node is present,it is removed from the syngraph dictionary
    ce = chemical_equation_constructor.build_from_reaction_string(
        d[0]["output_string"], "smiles"
    )
    syngraph.remove_node(ce.uid)

    assert len(syngraph.graph) == 2
    assert len(syngraph.get_roots()) > 1


def test_isolated_ce_removal():
    route_isolated_nodes = [
        {
            "output_string": "Cl[C:2]([CH3:1])=[O:3].[CH3:4][OH:5]>>[CH3:1][C:2](=[O:3])[O:5][CH3:4]",
            "query_id": "0",
        },
        {
            "output_string": "[CH3:5][O:4][C:3]([CH3:2])=[O:1]>>[CH3:2][C:3]([OH:4])=[O:1]",
            "query_id": "1",
        },
        {
            "output_string": "[CH3:4][C:5](Cl)=[O:6].CC(O)=O.[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][O:3][C:5]([CH3:4])=[O:6]",
            "query_id": "2",
        },
        {
            "output_string": "O=[C:2](OC[CH3:4])[CH3:1].[Li][CH3:3]>>[CH2:1]=[C:2]([CH3:3])[CH3:4]",
            "query_id": "3",
        },
    ]
    cec = ChemicalEquationConstructor()
    removed_ces = [
        cec.build_from_reaction_string(
            "Cl[C:2]([CH3:1])=[O:3].[CH3:4][OH:5]>>[CH3:1][C:2](=[O:3])[O:5][CH3:4]",
            "smiles",
        ),
        cec.build_from_reaction_string(
            "[CH3:5][O:4][C:3]([CH3:2])=[O:1]>>[CH3:2][C:3]([OH:4])=[O:1]", "smiles"
        ),
    ]
    bp_syngraph = BipartiteSynGraph(route_isolated_nodes)
    assert len(bp_syngraph.get_roots()) == 1
    assert all(removed_ces) not in bp_syngraph
    mpr_syngraph = MonopartiteReacSynGraph(route_isolated_nodes)
    assert len(mpr_syngraph.get_roots()) == 1
    assert all(removed_ces) not in mpr_syngraph


def test_syngraph_from_chem_equation_list(reaction_list):
    reaction_smiles = [d["output_string"] for d in reaction_list]
    rdrxn_list = [cif.rdrxn_from_string(r, "smiles") for r in reaction_smiles]
    rxn_block_list = [cif.rdrxn_to_string(b, "rxn_blockV2K") for b in rdrxn_list]

    rxns = MonopartiteReacSynGraph.build_chemical_equations(rxn_block_list, "rxn_block")
    syngraph = MonopartiteReacSynGraph()
    syngraph.builder_from_reaction_list(rxns)
    assert len(syngraph.graph) == len(reaction_list)
