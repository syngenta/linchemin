import pytest

from linchemin.cgu.discgraph import DisconnectionGraph, MissingAtomMapping
from linchemin.cgu.syngraph import MonopartiteReacSynGraph
from linchemin.cheminfo.reaction import ChemicalEquationConstructor

mapped_reactions = {
    "r1": "CN(C)C=O.F[c:5]1[n:6][cH:7][cH:8][cH:9][c:10]1[F:11].O=C([O-])[O-].[CH3:1][CH:2]([CH3:3])["
    "SH:4].[K+].[K+]>>[CH3:1][CH:2]([CH3:3])[S:4][c:5]1[n:6][cH:7][cH:8][cH:9][c:10]1[F:11]"
}


def test_basic_disconnectionGraph():
    chemical_equation_constructor = ChemicalEquationConstructor(
        identity_property_name="smiles"
    )
    chemical_equation = chemical_equation_constructor.build_from_reaction_string(
        reaction_string=mapped_reactions["r1"], inp_fmt="smiles"
    )
    syngraph = MonopartiteReacSynGraph()
    syngraph.add_node((chemical_equation, []))
    discgraph = DisconnectionGraph(syngraph)
    assert discgraph

    ce_constructor = ChemicalEquationConstructor(identity_property_name="smiles")
    no_mapping_ce = ce_constructor.build_from_reaction_string(
        reaction_string="CCN.CCOC(=O)CC>>CCNC(=O)CC", inp_fmt="smiles"
    )
    not_mapped_syngraph = MonopartiteReacSynGraph()
    not_mapped_syngraph.add_node((no_mapping_ce, []))

    # if a not mapped reaction is present in the input SynGraph, an error is raised
    with pytest.raises(MissingAtomMapping) as e:
        DisconnectionGraph(not_mapped_syngraph)
    assert "MissingAtomMapping" in str(e.type)

    with pytest.raises(TypeError) as e:
        discgraph.add_disc_node((no_mapping_ce, []))
    assert "TypeError" in str(e.type)
