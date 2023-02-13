from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union

import pandas as pd

import linchemin.IO.io as lio
from linchemin import settings
from linchemin.cgu.syngraph import (BipartiteSynGraph, MonopartiteMolSynGraph,
                                    MonopartiteReacSynGraph)
from linchemin.cgu.translate import get_available_data_models
from linchemin.cheminfo.atom_mapping import get_available_mappers
from linchemin.interfaces.facade import facade
from linchemin.rem.clustering import get_available_clustering
from linchemin.rem.graph_distance import (get_available_ged_algorithms,
                                          get_ged_parameters)
from linchemin.rem.route_descriptors import get_available_descriptors
from linchemin.utilities import console_logger

"""
Module containing out-of-the-box "workflow", consisting of a sequence of facade functions.
The functionalities to be actually activated are selected by the user by setting the arguments of the function
'process_routes'.
"""
logger = console_logger(__name__)


class NoValidRoute(Exception):
    """ Raised if no valid routes are found """
    pass


@dataclass
class WorkflowOutput:
    """ Class to store the outcome of the 'process_routes' function.

        Attributes:
        ------------
             routes_list: a list of routes as instances of a SynGraph subclass

             descriptors: a pandas DataFrame containing the routes ids and the computed descriptors (returned only if
                          the 'compute_descriptors' functionality is activated)

             clustering: a tuple with the output of the cluster algorithm and the Silhouette score (returned only if
                          the 'clustering' or the 'clustering_and_d_matrix' functionalities are activated)

             clustered_descriptors: a pandas DataFrame containing the routes ids, the cluster labels and few descriptors
                                    (returned only if the 'clustering' or the 'clustering_and_d_matrix' functionalities
                                    are activated)

             distance_matrix: a pandas DataFrame containing the GED distance matrix (returned only if the
                              'distance_matrix' or the 'clustering_and_d_matrix' functionalities are activated)

             tree: an instances of a SynGraph subclass obtained from the merging of the input routes (returned only if
                   the 'merging' functionality is activated)

             reaction_strings: a list of reaction strings (returned only if the 'extracting_reactions' functionality
                               is activated)

             log: a dictionary containing the exceptions occurred during the process
    """
    routes_list: list = field(default_factory=list)
    descriptors: pd.DataFrame = field(default_factory=pd.DataFrame)
    clustering: tuple = field(default_factory=tuple)
    clustered_descriptors: pd.DataFrame = field(default_factory=pd.DataFrame)
    distance_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    tree: Union[BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph] | None = None
    reaction_strings: list = field(default_factory=list)
    log: dict = field(default_factory=dict)


# Functionalities factory
class WorkflowStep(ABC):
    """ Abstract class for the WorkflowStep taking care of the single functionalities """
    info: str

    @abstractmethod
    def perform_step(self, data: dict, output: WorkflowOutput) -> WorkflowOutput:
        pass


class TranslationStep(WorkflowStep):
    """ Handler handling the translate functionality of facade
        It translates the input list of graphs in syngraph objects
    """
    info = 'To translate routes'

    casps = {'ibmrxn': 'ibm_retro',
             'az': 'az_retro',
             'askcos': 'mit_retro'}

    def perform_step(self, params: dict, output: WorkflowOutput):
        print('Translating the routes in the input file to a list of SynGraph....')
        all_routes = []
        for file, casp in params['input'].items():
            routes = lio.read_json(file)
            if casp not in self.casps:
                logger.error(f"{casp} is not a valid casp name. Available casps are: {list(self.casps.keys())}")
                raise KeyError

            syngraph_routes, meta = facade('translate', input_format=self.casps[casp], input_list=routes,
                                           out_data_model=params['out_data_model'],
                                           parallelization=params['parallelization'],
                                           n_cpu=params['n_cpu'])
            all_routes += syngraph_routes
            output.log['_'.join(['translation', file])] = meta
            if len(all_routes) < 1:
                logger.error('No valid routes were found. Workflow interrupted.')
                raise NoValidRoute

        output.routes_list = all_routes
        return output


class DescriptorsStep(WorkflowStep):
    """ Handler handling the descriptors functionality of facade
        If 'compute_descriptors' is in 'functionalities', the selected descriptors are copmuted and written to the file
        'descriptors.csv'
    """
    info = 'To compute routes descriptors'

    def perform_step(self, params: dict, output: WorkflowOutput):
        print('Computing the routes descriptors...')
        descriptors, meta = facade('routes_descriptors', output.routes_list, descriptors=params['descriptors'])
        output.descriptors = descriptors
        output.log['compute_descriptors'] = meta
        lio.dataframe_to_csv(descriptors, 'descriptors.csv')
        return output


class ClusteringAndDistanceMatrixStep(WorkflowStep):
    """ Handler handling the clustering functionality of facade
        If the 'clustering' argument is in 'functionalities', the routes are clustered and a file 'cluster_metrics.csv'
        with the cluster label, the number of steps and the number of branches for each route is written.
    """
    info = 'To compute the distance matrix and clustering the routes'

    def perform_step(self, params: dict, output: WorkflowOutput):
        print('Clustering the routes...')
        cluster_output, metrics, meta = facade('clustering', output.routes_list,
                                               clustering_method=params['clustering_method'],
                                               ged_method=params['ged_method'],
                                               ged_params=params['ged_params'],
                                               save_dist_matrix=True,
                                               compute_metrics=True, parallelization=params['parallelization'],
                                               n_cpu=params['n_cpu'])
        if cluster_output is None and metrics is None:
            output.log['clustering_and_d_matrix'] = meta
        else:
            output.clustering = cluster_output[0]
            output.clustered_descriptors = metrics
            output.distance_matrix = cluster_output[2]
            output.log['clustering_and_d_matrix'] = meta
            lio.dataframe_to_csv(cluster_output[2], 'distance_matrix.csv')
            lio.dataframe_to_csv(metrics, 'cluster_metrics.csv')

        return output


class ClusteringStep(WorkflowStep):
    """ Handler handling the clustering functionality of facade
        If the 'clustering_dist_matrox' argument is in 'functionalities', the routes are clustered and a file
        'cluster_metrics.csv' with the cluster label, the number of steps and the number of branches for each route is
        written and the distance matrix is written in the 'distance_matrix.csv' file.
    """
    info = 'To clustering the routes'

    def perform_step(self, params: dict, output: WorkflowOutput):
        cluster_output, metrics, meta = facade('clustering', output.routes_list,
                                               clustering_method=params['clustering_method'],
                                               ged_method=params['ged_method'],
                                               ged_params=params['ged_params'],
                                               compute_metrics=True,
                                               parallelization=params['parallelization'],
                                               n_cpu=params['n_cpu'])
        if cluster_output is None:
            output.log['clustering_and_d_matrix'] = meta
        else:
            output.clustering = cluster_output[0]
            output.clustered_descriptors = metrics
            output.log['clustering'] = meta
            lio.dataframe_to_csv(metrics, 'cluster_metrics.csv')
        return output


class DistanceMatrixStep(WorkflowStep):
    """ Handler handling the distance matrix functionality of facade.
        If the 'distance_matrix' argument is in 'functionalities', the distance matrix is computed and written in the
        'distance_matrix.csv' file.
    """
    info = 'To compute the distance matrix'

    def perform_step(self, params: dict, output: WorkflowOutput):
        print('Computing the distance matrix...')
        d_matrix, meta = facade('distance_matrix', output.routes_list,
                                ged_method=params['ged_method'],
                                ged_params=params['ged_params'],
                                parallelization=params['parallelization'],
                                n_cpu=params['n_cpu'])
        if not d_matrix:
            output.log['distance_matrix'] = meta
        else:
            output.distance_matrix = d_matrix
            output.log['distance_matrix'] = meta
            lio.dataframe_to_csv(d_matrix, 'distance_matrix.csv')
        return output


class MergingStep(WorkflowStep):
    """ Handler handling the merging functionality of facade.
        If the 'merging' argument is in 'functionalities', the routes are merged and the obtained tree is written
        in the 'tree' file. The file extension is determined by the 'output_format' argument.
    """
    info = 'To merge the routes in a "SynTree"'

    def perform_step(self, params: dict, output: WorkflowOutput):
        print('Merging routes...')
        merged = facade('merging', output.routes_list, params['out_data_model'])
        output.tree = merged
        write_syngraph([merged], params['out_data_model'], params['output_format'], 'tree')
        return output


class ExtractingReactionsStep(WorkflowStep):
    """ Handler handling the extract_reactions_strings functionality of facade.
        If the 'extracting_reactions' argument is in 'functionalities', a list of reaction strings is extracted for
        each route.
    """
    info = 'To extract reactions strings from a list of SynGraph objects'

    def perform_step(self, params: dict, output: WorkflowOutput):
        print('Extracting reaction strings...')
        reactions, m = facade('extract_reactions_strings', output.routes_list)
        output.reaction_strings = reactions
        lio.write_json(reactions, 'reaction_strings.json')
        return output


class AtomMappingStep(WorkflowStep):
    """ Handler handling the atom_mapping functionality of facade.
        If the 'atom_mapping' argument is in 'functionalities', reaction strings are extracted from the routes and
        sent to the selected atom-to-atom mapping tools. The mapped reaction strings are then used to rebuild the routes
    """
    info = 'To map the reaction strings involved in the routes'

    def perform_step(self, params: dict, output: WorkflowOutput):
        print('Mapping the reactions')
        mapped_routes, m = facade('atom_mapping', output.routes_list, params['mapper'], params['out_data_model'])
        output.routes_list = mapped_routes
        return output


# Workflow Chain
class WorkflowHandler(ABC):
    """ Abstract handler for the concrete handlers taking care of the single steps in the workflow """

    @abstractmethod
    def execute(self, *args, **kwargs):
        pass


class WorkflowStarter(WorkflowHandler):
    """ Concrete handler to perform the first step in the workflow """

    def execute(self, params: dict, output: WorkflowOutput):
        """ Executes the first step of the workflow: translation of the routes in SynGraph objects and, if selected,
            the reactions' atom mapping
        """
        try:
            output = self.get_translation(params, output)
            if params['mapping'] is True:
                output = self.get_mapped_routes(params, output)

            return Executor().execute(output.routes_list, params, output)
        except NoValidRoute:
            return None

    def get_translation(self, params: dict, output: WorkflowOutput) -> WorkflowOutput:
        """ Calls the translation step of the workflow """
        output = TranslationStep().perform_step(params, output)
        return output

    def get_mapped_routes(self, params: dict, output: WorkflowOutput) -> WorkflowOutput:
        """ Calles the atom mapping step of the workflow """
        output = AtomMappingStep().perform_step(params, output)
        return output


class Executor(WorkflowHandler):
    """ Concrete handler performing the desired optional functionalities """
    steps_map = {
        'compute_descriptors': {'value': DescriptorsStep,
                                'info': DescriptorsStep.info},
        'clustering_and_d_matrix': {'value': ClusteringAndDistanceMatrixStep,
                                    'info': ClusteringAndDistanceMatrixStep.info},
        'clustering': {'value': ClusteringStep,
                       'info': ClusteringStep.info},
        'distance_matrix': {'value': DistanceMatrixStep,
                            'info': DistanceMatrixStep.info},
        'merging': {'value': MergingStep,
                    'info': MergingStep.info},
        'extracting_reactions': {'value': ExtractingReactionsStep,
                                 'info': ExtractingReactionsStep.info},
    }

    def execute(self, syngraph_routes: list, params: dict, output: WorkflowOutput):
        """ Executes the selected functionalities """
        if params['functionalities'] is not None:
            requests = params['functionalities']

            for request in requests:
                if request not in self.steps_map:
                    logger.error(f'"{request}" is not a valid functionality. Available functionalities are: '
                                 f'{list(self.steps_map.keys())}')
                    raise KeyError
                output = self.steps_map[request]['value']().perform_step(params, output)

        return Finisher().execute(params=params, output=output)

    def get_workflow_functions(self):
        return {f: additional_info['info'] for f, additional_info in self.steps_map.items()}


class Finisher(WorkflowHandler):
    """ Concrete handler to perform the last step in the workflow"""

    def execute(self, params: dict, output: WorkflowOutput):
        print('Writing the routes in the output file...')
        write_syngraph(output.routes_list, params['out_data_model'], params['output_format'], 'routes')
        return output


class WorkflowBuilder:
    """ Class to start the chain calling the handler of the first step """

    @staticmethod
    def initiate_workflow(params):
        output = WorkflowOutput()
        return WorkflowStarter().execute(params, output)


def process_routes(input_dict: dict,
                   output_format=settings.WORKFLOW.output_format,
                   mapping=settings.WORKFLOW.mapping,
                   functionalities=settings.WORKFLOW.functionalities,
                   mapper=settings.FACADE.mapper,
                   out_data_model=settings.FACADE.out_data_model,
                   descriptors=settings.FACADE.descriptors,
                   ged_method=settings.FACADE.ged_method,
                   ged_params=settings.FACADE.ged_params,
                   clustering_method=settings.FACADE.clustering_method,
                   parallelization=settings.FACADE.parallelization,
                   n_cpu=settings.FACADE.n_cpu, ):
    """ Function to start the workflow chain of responsibility: based on the input arguments, only the selected
        functionalities are performed. The mandatory start and stop actions are (i) to read a json file containing the
        routes predicted by a CASP tool, and (ii) to write the routes as dictionaries in an output file.
        Possible additional actions are:
            - performing the atom mapping of the reactions involved in the routes
            - computing route descriptors
            - computing the distance matrix between the routes
            - clustering the routes
            - merging the routes
            - extracting the reaction strings from the routes

        :param:
            input_dict: a dictionary
                It contains the path to the input files and the relative casp names in the form
                {'file_path': 'casp_name'}

            output_format: a string (optional; default: 'json')
                It indicates which format should be used while writing the routes (csv or json)

            mapping: a boolean (optional; default: False)
                It indicate whether the reactions involved in the routes should go through the atom-to-atom mapping

            functionalities: a list of strings (optional; default: None)
                It contains the names of the functionalities to be performed; if it is None, the input routes are read
                and written to a file

            mapper: a string (optional; default: None)
                It indicates which mapping tools should be used; if it is None, the mapping pipeline is used

            out_data_model: a string (optional; default: 'monopartite_reactions')
                It indicates the desired data model for the output routes

            descriptors: a list of strings (optional; default: None)
                It contains the names of the descriptos to be computed; if it is None, all the available are calculated

            ged_method: a string (optional; default: 'nx_ged')
                It indicates the method to be used for graph similarity calculations

            ged_params: a dictionary (optional; default: None)
                It contains the parameters for specifying reaction and molecular fingerprints and similarity functions;
                if it is None, the default values are used

            clustering_method: a string (optional; default: None)
                It indicates which clustering algorithm to be used for clustering the routes; if it is None, hdbscan
                is used when there are more than 15 routes, Agglomerative Clustering otherwise

            parallelization: a boolean (optional, default: False)
                It indicates whether parallel computing should be used where possible

            n_cpu: an integer (optional; default: 'mp.cpu_count()')
                It indicates the number of cpus to be used if parallelization is used.

        :return:
            output: a WorkflowOutput object
                Its attributes store the results of the selected functionalities. The outcomes are also written to files

    """

    params = {'input': input_dict,
              'output_format': output_format,
              'mapping': mapping,
              'functionalities': functionalities,
              'mapper': mapper,
              'out_data_model': out_data_model,
              'descriptors': descriptors,
              'ged_method': ged_method,
              'ged_params': ged_params,
              'clustering_method': clustering_method,
              'parallelization': parallelization,
              'n_cpu': n_cpu,
              }
    output = WorkflowBuilder().initiate_workflow(params)
    print('All done!')
    return output


#### Supporting functions and classes
# Writers factory
class SyngraphWriter(ABC):
    """ Abstract class for the SyngraphWriter taking care of writing the routes in different file formats """

    @abstractmethod
    def write_file(self, syngraphs: list, out_data_model: str, output_format: str, file_name: str) -> None:
        pass


class JsonWriter(SyngraphWriter):
    """ Writer to generate a Json file of the routes """

    def write_file(self, syngraphs: list, out_data_model: str, output_format: str, file_name: str):
        routes, meta = facade('translate', 'syngraph', syngraphs, 'noc', out_data_model=out_data_model)
        file_name = ''.join([file_name, '.', output_format])
        lio.write_json(routes, file_name)


class CsvWriter(SyngraphWriter):
    """ Writer to generate a csv file of the routes """

    def write_file(self, syngraphs: list, out_data_model: str, output_format: str, file_name: str):
        routes, meta = facade('translate', 'syngraph', syngraphs, 'noc', out_data_model=out_data_model)
        file_name = ''.join([file_name, '.', output_format])
        lio.dict_list_to_csv(routes, file_name)


class PngWriter(SyngraphWriter):
    """ Writer to generate png files of the routes """

    def write_file(self, syngraphs: list, out_data_model: str, output_format: str, file_name: str):
        facade('translate', 'syngraph', syngraphs, 'pydot_visualization', out_data_model=out_data_model)


class GraphMLWriter(SyngraphWriter):
    """ Writer to generate graphml files of the routes """

    def write_file(self, syngraphs: list, out_data_model: str, output_format: str, file_name: str):
        nx_routes, meta = facade('translate', 'syngraph', syngraphs, 'networkx', out_data_model=out_data_model)
        for nx_route in nx_routes:
            for n, data in nx_route.nodes.items():
                r_id = data['attributes']['source']
                del nx_route.nodes[n]["attributes"]

            for n1, n2, d in nx_route.edges(data=True):
                del d['attributes']
            fname = '.'.join([r_id, 'graphml'])
            lio.write_nx_to_graphml(nx_route, fname)


class SyngraphWriterFactory:
    file_formats = {'csv': CsvWriter,
                    'json': JsonWriter,
                    'png': PngWriter,
                    'graphml': GraphMLWriter,
                    }

    def select_writer(self, syngraphs: list, out_data_model: str, output_format: str, file_name: str):
        if output_format not in self.file_formats:
            logger.error(f"{output_format} is not a valid format. "
                         f"Available formats are {list(self.file_formats.keys())}")
            raise KeyError
        writer = self.file_formats[output_format]
        writer().write_file(syngraphs, out_data_model, output_format, file_name)


def write_syngraph(syngraphs: list, out_data_model: str, output_format: str, file_name: str):
    """
    Takes a list of SynGraph instances and writes them to a file

    :param:
        syngraphs: a list of SynGraph objects

        out_data_model: a string indicating the data model to be used for the output graphs

        output_format: a string indicating the format of the output file

        file_name: a string indicating the name of the output file
    """
    SyngraphWriterFactory().select_writer(syngraphs, out_data_model, output_format, file_name)


# Workflow helper
def get_workflow_options(verbose=False):
    """
        Returns the available options for the 'process_routes' function.

        :param:
            verbose: a boolean (optional; default: False)
                It indicates whether the information should be printed on the screen

        :return:
            a dictionary listing arguments, options and default values of the 'process_routes' function
    """
    if verbose:
        return print_options()
    else:
        return {'input_dict': {'name_or_flags': ['-input_dict'],
                               'default': None,
                               'required': True,
                               'type': dict,
                               'choices': None,
                               'help': 'Path to the input files and relative casp names in the form "file_path"="casp_name". Available CASP names are {}'.format(
                                   list(TranslationStep.casps.keys())),
                               'dest': 'input_dict'},
                'output_format': {'name_or_flags': ['-output_format'],
                                  'default': settings.WORKFLOW.output_format,
                                  'required': False,
                                  'type': str,
                                  'choices': list(SyngraphWriterFactory().file_formats.keys()),
                                  'help': 'Format of the output file containing the routes',
                                  'dest': 'output_format'},
                'mapping': {'name_or_flags': ['-mapping'],
                            'default': settings.WORKFLOW.mapping,
                            'required': False,
                            'type': bool,
                            'choices': [True, False],
                            'help': 'Whether the atom-to-atom mapping of the reactions should be performed',
                            'dest': 'mapping'},
                'mapper': {'name_or_flags': ['-mapper'],
                           'default': settings.FACADE.mapper,
                           'required': False,
                           'type': str,
                           'choices': get_available_mappers(),
                           'help': 'Which mapper should be used; if None, the mapping pipeline is used',
                           'dest': 'mapper'},
                'functionalities': {'name_or_flags': ['-functionalities'],
                                    'default': settings.WORKFLOW.functionalities,
                                    'required': False,
                                    'type': list,
                                    'choices': Executor().get_workflow_functions(),
                                    'help': 'List of functionalities to be performed',
                                    'dest': 'functionalities'},
                'out_data_model': {'name_or_flags': ['-out_data_model'],
                                   'default': settings.FACADE.out_data_model,
                                   'required': False,
                                   'type': str,
                                   'choices': get_available_data_models(),
                                   'help': 'Data model of the output graphs',
                                   'dest': 'out_data_model'},
                'descriptors': {'name_or_flags': ['-descriptors'],
                                'default': settings.FACADE.descriptors,
                                'required': False,
                                'type': list,
                                'choices': get_available_descriptors(),
                                'help': 'List of descriptors to be calculated; if None, all descriptors are calculated',
                                'dest': 'descriptors'},
                'ged_method': {'name_or_flags': ['-ged_method'],
                               'default': settings.FACADE.ged_method,
                               'required': False,
                               'type': str,
                               'choices': get_available_ged_algorithms(),
                               'help': 'Method to be used to calculate the GED',
                               'dest': 'ged_method'},
                'ged_params': {'name_or_flags': ['-ged_params'],
                               'default': settings.FACADE.ged_params,
                               'required': False,
                               'type': dict,
                               'choices': get_ged_parameters(),
                               'help': 'Parameters of the molecular and/or reaction chemical similarity',
                               'dest': 'ged_params'},
                'clustering_method': {'name_or_flags': ['-clustering_method'],
                                      'default': settings.FACADE.clustering_method,
                                      'required': False,
                                      'type': str,
                                      'choices': get_available_clustering(),
                                      'help': 'Method to be used to calculate the GED',
                                      'dest': 'ged_method'},
                'parallelization': {'name_or_flags': ['-parallelization'],
                                    'default': settings.FACADE.parallelization,
                                    'required': False,
                                    'type': bool,
                                    'choices': [True, False],
                                    'help': 'Whether parallelization should be used',
                                    'dest': 'parallelization'},
                'n_cpu': {'name_or_flags': ['-n_cpu'],
                          'default': 'All the available CPUs as given by "multiprocessing.cpu_count()"',
                          'required': False,
                          'type': int,
                          'choices': None,
                          'help': 'Number of CPUs to be used in parallelization',
                          'dest': 'n_cpu'},
                }


def print_options():
    """
        Prints the available options for the workflow function.
    """
    print('Workflow options and default:')
    data = get_workflow_options()
    for d, info in data.items():
        print('argument:', d)
        print('     info: ', info['help'])
        print('     default: ', info['default'])
        print('     available options: ', info['choices'])
    return data
