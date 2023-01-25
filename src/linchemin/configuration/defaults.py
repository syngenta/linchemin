import multiprocessing as mp

DEFAULT_WORKFLOW = {
    'functionalities': None,
    'output_format': 'json',
    'mapping': False,
}

DEFAULT_GED = {
    'reaction_fp': {'value': 'structure_fp',
                    'info': 'Structural reaction fingerprints as defined in RDKit',
                    'general_info': 'Chemical reaction fingerprints to be used'},
    'reaction_fp_params': {'value': None,
                           'info': 'The default parameters provided by RDKit are used for the specified reaction '
                                   'fingerprints',
                           'general_info': 'Chemical reaction fingerprints parameters'},
    'reaction_similarity_name': {'value': 'tanimoto',
                                 'info': 'The Tanimoto similarity as defined in RDKit is used for computing the '
                                         'reaction similarity',
                                 'general_info': 'Chemical similarity method for reactions'},
    'molecular_fp': {'value': 'rdkit',
                     'info': 'The molecular fingerprints rdFingerprintGenerator.GetRDKitFPGenerator are used',
                     'general_info': 'Molecular fingerprints to be used'},
    'molecular_fp_params': {'value': None,
                            'info': 'The default parameters provided by RDKit are used for the specified molecular '
                                    'fingerprints',
                            'general_info': 'Molecular fingerprints parameters'},
    'molecular_fp_count_vect': {'value': False,
                                'info': 'False. If set to True, GetCountFingerprint is used for the molecular '
                                        'fingerprints',
                                'general_info': 'If set to True, GetCountFingerprint is used for the molecular '
                                                'fingerprints'},
    'molecular_similarity_name': {'value': 'tanimoto',
                                  'info': 'The Tanimoto similarity as defined in RDKit is used for computing the '
                                          'molecular similarity',
                                  'general_info': 'Chemical similarity method for molecules'},
}

DEFAULT_FACADE = {
    'translate': {'value': {'data_format': 'syngraph',
                            'out_data_model': 'bipartite',
                            'parallelization': False,
                            'n_cpu': mp.cpu_count()},
                  'info': 'A list of instances of bipartite SynGraphs'},
    'routes_descriptors': {'value': {'descriptors': None},
                           'info': 'All the available descriptors are computed'},
    'distance_matrix': {'value': {'ged_method': 'nx_ged',
                                  'ged_params': None,
                                  'parallelization': False,
                                  'n_cpu': mp.cpu_count()},
                        'info': 'The standard NetworkX GED algorithm and the default parameters are used'},
    'clustering': {'value': {'clustering_method': None,
                             'ged_method': 'nx_ged',
                             'ged_params': None,
                             'save_dist_matrix': False,
                             'compute_metrics': False,
                             'parallelization': False,
                             'n_cpu': mp.cpu_count()},
                   'info': 'AgglomerativeClustering is used when there are less than 15 routes, hdbscan otherwise.'
                           'The standard NetworkX algorithm and default parameters are used for the distance matrix'
                           'calculation'},
    'merging': {'value': {'out_data_model': 'bipartite'},
                'info': 'A new, "merged", bipartite SynGraph'},
    'atom_mapping': {'value': {'mapper': None,
                               'out_data_model': 'bipartite'},
                     'info': 'The full atom mapping pipeline is used; a list of bipartite SynGraph is generated'},
}

DEFAULT_CONSTRUCTORS = {'molecular_identity_property_name': 'smiles',
                        'chemical_equation_identity_name': 'r_r_p',
                        'pattern_identity_property_name': 'smarts',
                        'template_identity_property': 'r_p'}

DEFAULT_CHEMICAL_SIMILARITY = {'includeAgents': True,
                               'diff_fp_fpSize': 2048,
                               'diff_fp_nonAgentWeight': 10,
                               'agentWeight': 1,
                               'diff_fp_bitRatioAgents': 0.0,
                               'struct_fp_fpSize': 4096,
                               'struct_fp_nonAgentWeight': 1,
                               'struct_fp_bitRatioAgents': 0.2,

                               }


def get_settings():
    all_settings = {}
    d_ged = {param: d['value'] for param, d in DEFAULT_GED.items()}
    all_settings['GED'] = d_ged
    all_settings['WORKFLOW'] = DEFAULT_WORKFLOW
    all_settings['CONSTRUCTORS'] = DEFAULT_CONSTRUCTORS
    all_settings['FACADE'] = {}
    for functionality, d in DEFAULT_FACADE.items():
        all_settings['FACADE'] |= d['value']
    all_settings['CHEMICAL_SIMILARITY'] = DEFAULT_CHEMICAL_SIMILARITY
    return all_settings


package_name = 'linchemin'

settings = get_settings()

secrets = {'SECRET_KEY': '<put the correct value here and keep it secret>'
           }
dynaconf_merge = True
