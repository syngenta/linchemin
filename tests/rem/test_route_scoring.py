from linchemin.rem.route_scoring import route_scorer
from linchemin.cgu.translate import translator

import pytest
import json


def test_scoring_factory():
    with pytest.raises(KeyError) as ke:
        graph = json.loads(open("../test_file/ibmrxn_retro_output_raw.json").read())
        r = translator('ibm_retro', graph[0], 'syngraph', out_data_model='bipartite')
        route_scorer(r, 'wrong_score')
    assert "KeyError" in str(ke.type)


def test_branch_score():
    f = json.loads(open("../test_file/ibm_output2.json").read())
    r = translator('ibm_retro', f[0], 'syngraph', out_data_model='bipartite')
    branch_score = route_scorer(r, 'branch_score')
    assert branch_score == 0.0

    mp_route = translator('ibm_retro', f[2], 'mp_syngraph', out_data_model='monopartite_reactions')
    branch_score_mp = route_scorer(mp_route, 'branch_score')
    assert branch_score_mp == 0.1
