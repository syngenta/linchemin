import multiprocessing as mp

"""
Module containing the definition of the default values for all functionalities in LinChemIn.
"""

DEFAULT_WORKFLOW = {
    "functionalities": None,
    "output_format": "json",
    "mapping": False,
}

DEFAULT_GED = {
    "reaction_fp": {
        "value": "structure_fp",
        "info": "Structural reaction fingerprints as defined in RDKit",
        "general_info": "Chemical reaction fingerprints to be used",
    },
    "reaction_fp_params": {
        "value": None,
        "info": "The default parameters provided by RDKit are used for the specified reaction "
        "fingerprints",
        "general_info": "Chemical reaction fingerprints parameters",
    },
    "reaction_similarity_name": {
        "value": "tanimoto",
        "info": "The Tanimoto similarity as defined in RDKit is used for computing the "
        "reaction similarity",
        "general_info": "Chemical similarity method for reactions",
    },
    "molecular_fp": {
        "value": "rdkit",
        "info": "The molecular fingerprints rdFingerprintGenerator.GetRDKitFPGenerator are used",
        "general_info": "Molecular fingerprints to be used",
    },
    "molecular_fp_params": {
        "value": None,
        "info": "The default parameters provided by RDKit are used for the specified molecular "
        "fingerprints",
        "general_info": "Molecular fingerprints parameters",
    },
    "molecular_fp_count_vect": {
        "value": False,
        "info": "False. If set to True, GetCountFingerprint is used for the molecular "
        "fingerprints",
        "general_info": "If set to True, GetCountFingerprint is used for the molecular "
        "fingerprints",
    },
    "molecular_similarity_name": {
        "value": "tanimoto",
        "info": "The Tanimoto similarity as defined in RDKit used to compute the molecular similarity",
        "general_info": "Chemical similarity method for molecules",
    },
    "n_cpu": {
        "value": 8,
        "info": "The number of CPU to be used for parallelization",
        "general_info": "The number of CPU to be used for parallelization",
    },
}

DEFAULT_FACADE = {
    "translate": {
        "value": {
            "data_format": "syngraph",
            "out_data_model": "bipartite",
            "parallelization": False,
            "n_cpu": mp.cpu_count(),
        },
        "info": "A list of instances of bipartite SynGraphs",
    },
    "routes_descriptors": {
        "value": {"descriptors": None},
        "info": "All the available descriptors are computed",
    },
    "distance_matrix": {
        "value": {
            "ged_method": "nx_ged",
            "ged_params": None,
            "parallelization": False,
            "n_cpu": mp.cpu_count(),
        },
        "info": "The standard NetworkX GED algorithm and the default parameters are used",
    },
    "clustering": {
        "value": {
            "clustering_method": None,
            "ged_method": "nx_ged",
            "ged_params": None,
            "save_dist_matrix": False,
            "compute_metrics": False,
            "parallelization": False,
            "n_cpu": 8,
        },
        "info": "AgglomerativeClustering is used when there are less than 15 routes, hdbscan otherwise."
        "The standard NetworkX algorithm and default parameters are used for the distance matrix"
        "calculation",
    },
    "merging": {
        "value": {"out_data_model": "bipartite"},
        "info": 'A new, "merged", bipartite SynGraph',
    },
    "atom_mapping": {
        "value": {"mapper": "rxnmapper", "out_data_model": "bipartite"},
        "info": "The rxnmapper tool is used; a list of bipartite SynGraph is generated",
    },
    "route_sanity_checks": {
        "value": {"checks": None, "out_data_model": "bipartite"},
        "info": "All the available sanity checks are performed",
    },
}

DEFAULT_CONSTRUCTORS = {
    "molecular_identity_property_name": "smiles",
    "chemical_equation_identity_name": "r_p",
    "pattern_identity_property_name": "smarts",
    "template_identity_property": "r_p",
    "molecular_hash_list": ["inchi_key", "inchikey_KET_15T"],
    "desired_product": None,
}

DEFAULT_CHEMICAL_SIMILARITY = {
    "includeAgents": True,
    "diff_fp_fpSize": 2048,
    "diff_fp_nonAgentWeight": 10,
    "agentWeight": 1,
    "diff_fp_bitRatioAgents": 0.0,
    "struct_fp_fpSize": 4096,
    "struct_fp_nonAgentWeight": 1,
    "struct_fp_bitRatioAgents": 0.2,
}

DEFAULT_CLUSTERING = {"min_cluster_size": 3, "linkage": "single"}

DEFAULT_SERVICES = {
    "namerxn": {"url": "http://127.0.0.1", "port": "8002/"},
    "rxnmapper": {"url": "http://127.0.0.1", "port": "8003/"},
}

DEFAULT_ROUTE_MINING = {
    "root": None,
    "new_reaction_list": None,
    "product_edge_label": "P",
    "reactant_edge_label": "R",
    "reagent_edge_label": "REAGENT",
    "molecule_node_label": "M",
    "chemicalequation_node_label": "CE",
}


def get_settings():
    """To assemble the default values read from the dictionaries above and written to the settings.yaml file"""
    d_ged = {param: d["value"] for param, d in DEFAULT_GED.items()}
    all_settings = {
        "GED": d_ged,
        "WORKFLOW": DEFAULT_WORKFLOW,
        "CONSTRUCTORS": DEFAULT_CONSTRUCTORS,
        "FACADE": {},
    }

    for functionality, d in DEFAULT_FACADE.items():
        all_settings["FACADE"].update(d["value"])
    all_settings["CHEMICAL_SIMILARITY"] = DEFAULT_CHEMICAL_SIMILARITY
    all_settings["CLUSTERING"] = DEFAULT_CLUSTERING
    all_settings["SERVICES"] = DEFAULT_SERVICES
    all_settings["ROUTE_MINING"] = DEFAULT_ROUTE_MINING
    return all_settings


package_name = "linchemin"

settings = get_settings()

secrets = {"SECRET_KEY": "<put the correct value here and keep it secret>"}
dynaconf_merge = True
