from linchemin.cgu.discgraph import DisconnectionGraph
from linchemin.cheminfo.reaction import ChemicalEquationConstructor
from linchemin.cgu.syngraph import MonopartiteReacSynGraph
import pytest

mapped_reactions = {'r1': 'CN(C)C=O.F[c:5]1[n:6][cH:7][cH:8][cH:9][c:10]1[F:11].O=C([O-])[O-].[CH3:1][CH:2]([CH3:3])['
                          'SH:4].[K+].[K+]>>[CH3:1][CH:2]([CH3:3])[S:4][c:5]1[n:6][cH:7][cH:8][cH:9][c:10]1[F:11]'}


def test_basic_disconnectionGraph():
    chemical_equation_constructor = ChemicalEquationConstructor(identity_property_name='smiles')
    chemical_equation = chemical_equation_constructor.build_from_reaction_string(
        reaction_string=mapped_reactions['r1'],
        inp_fmt='smiles')
    syngraph = MonopartiteReacSynGraph()
    syngraph.add_node((chemical_equation, []))
    discgraph = DisconnectionGraph(syngraph)
    assert discgraph


