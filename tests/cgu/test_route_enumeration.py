from linchemin.cgu.route_enumeration import RouteEnumerator
from linchemin.cgu.syngraph import MonopartiteReacSynGraph

test_route = [
    {
        "output_string": "c1ccccc1.Cl>>Clc1ccccc1",
        "query_id": 0,
    },
    {
        "output_string": "N.Clc1ccccc1>>Nc1cccc(Cl)c1",
        "query_id": 1,
    },
]


def test_route_enumeration():
    original_route = MonopartiteReacSynGraph(test_route)

    leaf_to_substitute_uid = next(
        mol.uid for mol in original_route.get_molecule_leaves() if mol.smiles == "Cl"
    )
    reaction_products = [
        uid
        for ce in original_route.get_unique_nodes()
        for uid in ce.role_map["products"]
        if leaf_to_substitute_uid in ce.role_map["reactants"]
    ]
    alternative_leaves = ["ClCl", "ClC(=O)C(Cl)=O"]
    enumerator = RouteEnumerator(
        original_route=original_route,
        leaf_to_substitute=leaf_to_substitute_uid,
        reaction_products=reaction_products,
    )
    routes = enumerator.build_alternative_routes(
        alternative_leaves=alternative_leaves, inp_fmt="smiles"
    )
    assert len(routes) == len(alternative_leaves)
    for route in routes:
        unshared_nodes = [
            n
            for n in original_route.get_molecule_leaves()
            if n not in route.get_leaves()
        ]
        assert len(unshared_nodes) == 1
