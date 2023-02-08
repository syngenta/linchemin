import json

import pytest

from linchemin.cgu.translate import translator
from linchemin.rem.route_scoring import route_scorer


def test_scoring_factory(ibm1_path):
    with pytest.raises(KeyError) as ke:
        graph = json.loads(open(ibm1_path).read())
        r = translator("ibm_retro", graph[0], "syngraph", out_data_model="bipartite")
        route_scorer(r, "wrong_score")
    assert "KeyError" in str(ke.type)


def test_branch_score(ibm2_path):
    f = json.loads(open(ibm2_path).read())
    r = translator("ibm_retro", f[0], "syngraph", out_data_model="bipartite")
    branch_score = route_scorer(r, "branch_score")
    assert branch_score == 0.0

    mp_route = translator(
        "ibm_retro", f[2], "mp_syngraph", out_data_model="monopartite_reactions"
    )
    branch_score_mp = route_scorer(mp_route, "branch_score")
    assert branch_score_mp == 0.1
