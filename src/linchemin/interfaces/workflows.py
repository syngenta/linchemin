from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Union

import pandas as pd

import linchemin.IO.io as lio
from linchemin import settings
from linchemin.cgu.convert import converter
from linchemin.cgu.syngraph import (
    BipartiteSynGraph,
    MonopartiteMolSynGraph,
    MonopartiteReacSynGraph,
)
from linchemin.cgu.syngraph_operations import merge_syngraph
from linchemin.cgu.translate import get_available_data_models
from linchemin.cheminfo.atom_mapping import get_available_mappers
from linchemin.interfaces.facade import facade
from linchemin.interfaces.utils_interfaces import get_ged_dict, get_parallelization_dict
from linchemin.interfaces.writers import SyngraphWriterFactory, write_syngraph
from linchemin.rem.clustering import get_available_clustering
from linchemin.rem.graph_distance import (
    get_available_ged_algorithms,
    get_ged_parameters,
)
from linchemin.rem.route_descriptors import get_available_descriptors
from linchemin.utilities import console_logger

"""
Module containing out-of-the-box "workflow", consisting of a sequence of facade functions.
The functionalities to be actually activated are selected by the user by setting the arguments of the function
'process_routes'.
"""
logger = console_logger(__name__)


class WorkflowError(Exception):
    """Base class for errors raised during workflow running"""


class NoValidRoute(WorkflowError):
    """Raised if no valid routes are found"""


class InvalidCasp(WorkflowError):
    """Raised if an unavailable CASP format is selected"""


@dataclass
class WorkflowOutput:
    """
    Class to store the outcome of the 'process_routes' function.

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
    tree: Union[
        BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph, None
    ] = None
    reaction_strings: list = field(default_factory=list)
    log: dict = field(default_factory=dict)


# Functionalities factory
class WorkflowStep(ABC):
    """Abstract class for the WorkflowStep taking care of the single functionalities"""

    info: str

    @abstractmethod
    def perform_step(self, data: dict, output: WorkflowOutput) -> WorkflowOutput:
        pass


class TranslationStep(WorkflowStep):
    """Handler handling the translation functionality of facade
    It translates the input list of graphs in syngraph objects
    """

    info = "To translate routes"

    casps = {
        "ibmrxn": "ibm_retro",
        "az": "az_retro",
        "askcos": "mit_retro",
        "reaxys": "reaxys",
    }

    def perform_step(self, params: dict, output: WorkflowOutput):
        print("Translating the routes in the input file to a list of SynGraph....")
        all_routes = []
        for file, casp in params["input"].items():
            routes = lio.read_json(file)
            if casp not in self.casps:
                logger.error(
                    f"{casp} is not a valid casp name. Available casps are: {list(self.casps.keys())}"
                )
                raise InvalidCasp

            syngraph_routes, meta = facade(
                functionality="translate",
                input_format=self.casps[casp],
                routes=routes,
                out_data_model=params["out_data_model"],
                parallelization=params["parallelization"],
                n_cpu=params["n_cpu"],
            )
            all_routes += syngraph_routes
            output.log["_".join(["translation", file])] = meta
            if len(all_routes) < 1:
                logger.error("No valid routes were found. Workflow interrupted.")
                raise NoValidRoute

        output.routes_list = all_routes
        return output


class DescriptorsStep(WorkflowStep):
    """Handler handling the descriptors functionality of facade
    If 'compute_descriptors' is in 'functionalities', the selected descriptors are computed and written to the file
    'descriptors.csv'
    """

    info = "To compute routes descriptors"

    def perform_step(self, params: dict, output: WorkflowOutput):
        print("Computing the routes descriptors...")
        descriptors, meta = facade(
            functionality="routes_descriptors",
            routes=output.routes_list,
            descriptors=params["descriptors"],
        )
        output.descriptors = descriptors
        output.log["compute_descriptors"] = meta
        lio.dataframe_to_csv(descriptors, "descriptors.csv")
        return output


class ClusteringAndDistanceMatrixStep(WorkflowStep):
    """Handler handling the clustering functionality of facade
    If the 'clustering' argument is in 'functionalities',
    the routes are clustered and a file 'cluster_metrics.csv'
    with the cluster label, the number of steps and the
    number of branches for each route is written.
    """

    info = "To compute the distance matrix and clustering the routes"

    def perform_step(self, params: dict, output: WorkflowOutput):
        print("Clustering the routes...")
        cluster_output, meta = facade(
            functionality="clustering",
            routes=output.routes_list,
            clustering_method=params["clustering_method"],
            ged_method=params["ged_method"],
            ged_params=params["ged_params"],
            save_dist_matrix=True,
            compute_metrics=True,
            parallelization=params["parallelization"],
            n_cpu=params["n_cpu"],
        )
        if len(cluster_output) == 2:
            clustering, metrics = cluster_output
        else:
            clustering = cluster_output
            metrics = None
        if cluster_output is None and metrics is None:
            output.log["clustering_and_d_matrix"] = meta
        else:
            output.clustering = clustering[0]
            output.clustered_descriptors = metrics
            output.distance_matrix = clustering[2]
            output.log["clustering_and_d_matrix"] = meta
            lio.dataframe_to_csv(clustering[2], "distance_matrix.csv")
            lio.dataframe_to_csv(metrics, "cluster_metrics.csv")

        return output


class ClusteringStep(WorkflowStep):
    """Handler handling the clustering functionality of facade
    If the 'clustering_dist_matrox' argument is in 'functionalities', the routes are clustered and a file
    'cluster_metrics.csv' with the cluster label, the number of steps and the number of branches for each route is
    written and the distance matrix is written in the 'distance_matrix.csv' file.
    """

    info = "To clustering the routes"

    def perform_step(self, params: dict, output: WorkflowOutput):
        cluster_output, meta = facade(
            functionality="clustering",
            routes=output.routes_list,
            clustering_method=params["clustering_method"],
            ged_method=params["ged_method"],
            ged_params=params["ged_params"],
            compute_metrics=True,
            parallelization=params["parallelization"],
            n_cpu=params["n_cpu"],
        )
        if len(cluster_output) == 2:
            clustering, metrics = cluster_output
        else:
            clustering = cluster_output
            metrics = None
        if clustering is None:
            output.log["clustering_and_d_matrix"] = meta
        else:
            output.clustering = clustering[0]
            output.clustered_descriptors = metrics
            output.log["clustering"] = meta
            lio.dataframe_to_csv(metrics, "cluster_metrics.csv")
        return output


class DistanceMatrixStep(WorkflowStep):
    """Handler handling the distance matrix functionality of facade.
    If the 'distance_matrix' argument is in 'functionalities', the distance matrix is computed and written in the
    'distance_matrix.csv' file.
    """

    info = "To compute the distance matrix"

    def perform_step(self, params: dict, output: WorkflowOutput):
        print("Computing the distance matrix...")
        d_matrix, meta = facade(
            functionality="distance_matrix",
            routes=output.routes_list,
            ged_method=params["ged_method"],
            ged_params=params["ged_params"],
            parallelization=params["parallelization"],
            n_cpu=params["n_cpu"],
        )
        if not d_matrix:
            output.log["distance_matrix"] = meta
        else:
            output.distance_matrix = d_matrix
            output.log["distance_matrix"] = meta
            lio.dataframe_to_csv(d_matrix, "distance_matrix.csv")
        return output


class MergingStep(WorkflowStep):
    """Handler handling the merging functionality of facade.
    If the 'merging' argument is in 'functionalities', the routes are merged and the obtained tree is written
    in the 'tree' file. The file extension is determined by the 'output_format' argument.
    """

    info = 'To merge the routes in a "SynTree"'

    def perform_step(self, params: dict, output: WorkflowOutput):
        print("Merging routes...")
        merged = merge_syngraph(list_syngraph=output.routes_list)
        merged_converted = converter(merged, params["out_data_model"])
        output.tree = merged_converted
        write_syngraph(
            [merged_converted],
            params["out_data_model"],
            params["output_format"],
            "tree",
        )
        return output


class ExtractingReactionsStep(WorkflowStep):
    """Handler handling the extract_reactions_strings functionality of facade.
    If the 'extracting_reactions' argument is in 'functionalities', a list of reaction strings is extracted for
    each route.
    """

    info = "To extract reactions strings from a list of SynGraph objects"

    def perform_step(self, params: dict, output: WorkflowOutput):
        print("Extracting reaction strings...")
        reactions, m = facade(
            functionality="extract_reactions_strings", routes=output.routes_list
        )
        output.reaction_strings = reactions
        lio.write_json(reactions, "reaction_strings.json")
        return output


class AtomMappingStep(WorkflowStep):
    """Handler handling the atom_mapping functionality of facade.
    If the 'atom_mapping' argument is in 'functionalities', reaction strings are extracted from the routes and
    sent to the selected atom-to-atom mapping tools. The mapped reaction strings are then used to rebuild the routes
    """

    info = "To map the reaction strings involved in the routes"

    def perform_step(self, params: dict, output: WorkflowOutput):
        print("Mapping the reactions")
        mapped_routes, m = facade(
            functionality="atom_mapping",
            routes=output.routes_list,
            mapper=params["mapper"],
        )
        mapped_routes_converted = [
            converter(r, params["out_data_model"]) for r in mapped_routes
        ]
        output.routes_list = mapped_routes_converted
        return output


# Workflow Chain
class WorkflowHandler(ABC):
    """Abstract handler for the concrete handlers taking care of the single steps in the workflow"""

    @abstractmethod
    def execute(self, params: dict, output: WorkflowOutput):
        pass


class WorkflowStarter(WorkflowHandler):
    """Concrete handler to perform the first step in the workflow"""

    def execute(self, params: dict, output: WorkflowOutput):
        """Executes the first step of the workflow: translation of the routes in SynGraph objects and, if selected,
        the reactions' atom mapping
        """
        try:
            output = self._get_translation(params, output)
            if params["mapping"] is True:
                output = self._get_mapped_routes(params, output)

            return Executor().execute(params, output)
        except NoValidRoute:
            return None

    @staticmethod
    def _get_translation(params: dict, output: WorkflowOutput) -> WorkflowOutput:
        """Calls the translation step of the workflow"""
        output = TranslationStep().perform_step(params, output)
        return output

    @staticmethod
    def _get_mapped_routes(params: dict, output: WorkflowOutput) -> WorkflowOutput:
        """Calls the atom mapping step of the workflow"""
        output = AtomMappingStep().perform_step(params, output)
        return output


class Executor(WorkflowHandler):
    """Concrete handler performing the desired optional functionalities"""

    steps_map = {
        "compute_descriptors": {"value": DescriptorsStep, "info": DescriptorsStep.info},
        "clustering_and_d_matrix": {
            "value": ClusteringAndDistanceMatrixStep,
            "info": ClusteringAndDistanceMatrixStep.info,
        },
        "clustering": {"value": ClusteringStep, "info": ClusteringStep.info},
        "distance_matrix": {
            "value": DistanceMatrixStep,
            "info": DistanceMatrixStep.info,
        },
        "merging": {"value": MergingStep, "info": MergingStep.info},
        "extracting_reactions": {
            "value": ExtractingReactionsStep,
            "info": ExtractingReactionsStep.info,
        },
    }

    def execute(self, params: dict, output: WorkflowOutput):
        """Executes the selected functionalities"""
        if params["functionalities"] is not None:
            requests = params["functionalities"]

            for request in requests:
                if request not in self.steps_map:
                    logger.warning(
                        f'"{request}" is not a valid functionality: it will be ignored.'
                        "Available functionalities are: "
                        f"{list(self.steps_map.keys())}"
                    )
                    continue
                output = self.steps_map[request]["value"]().perform_step(params, output)

        return Finisher().execute(params=params, output=output)

    def get_workflow_functions(self):
        return {
            f: additional_info["info"] for f, additional_info in self.steps_map.items()
        }


class Finisher(WorkflowHandler):
    """Concrete handler to perform the last step in the workflow"""

    def execute(self, params: dict, output: WorkflowOutput):
        print("Writing the routes in the output file...")
        write_syngraph(
            syngraphs=output.routes_list,
            out_data_model=params["out_data_model"],
            output_format=params["output_format"],
            file_name="routes",
        )
        return output


class WorkflowBuilder:
    """Class to start the chain calling the handler of the first step"""

    @staticmethod
    def initiate_workflow(params):
        output = WorkflowOutput()
        return WorkflowStarter().execute(params, output)


def process_routes(
    input_dict: dict,
    output_format: str = settings.WORKFLOW.output_format,
    mapping: bool = settings.WORKFLOW.mapping,
    functionalities: Union[List[str], None] = settings.WORKFLOW.functionalities,
    mapper: Union[str, None] = settings.FACADE.mapper,
    out_data_model: str = settings.FACADE.out_data_model,
    descriptors: List[str] = settings.FACADE.descriptors,
    ged_method: str = settings.FACADE.ged_method,
    ged_params: Union[dict, None] = settings.FACADE.ged_params,
    clustering_method: Union[str, None] = settings.FACADE.clustering_method,
    parallelization: bool = settings.FACADE.parallelization,
    n_cpu: int = settings.FACADE.n_cpu,
) -> WorkflowOutput:
    """
    Function process routed predicted by CASP tools: based on the input arguments, only the selected
    functionalities are performed. The mandatory start and stop actions are (i) to read a json file containing the
    routes predicted by a CASP tool, and (ii) to write the routes in an output file.
    Possible additional actions are:
        - performing the atom mapping of the reactions involved in the routes
        - computing route descriptors
        - computing the distance matrix between the routes
        - clustering the routes
        - merging the routes
        - extracting the reaction strings from the routes

    Parameters:
    ------------
    input_dict: dict
        The path to the input files and the relative casp names in the form {'file_path': 'casp_name'}
    output_format: Optional[str]
        The type of file to which the routes should be written (default 'json')
    mapping: Optional[bool]
        Whether the reactions involved in the routes should go through the atom-to-atom mapping (default False)
    functionalities: Optional[Union[List[str], None]]
        The list of the functionalities to be performed; if it is None, the input routes are read
        and written to a file (default None)
    mapper: Optional[str]
        The name of the mapping tool to be used; if it is None, the mapping pipeline is used (default None)
    out_data_model: Optional[str]
        The data model for the output routes (default 'bipartite')
    descriptors: Optional[Union[List[str], None]]
        The list of the descriptos to be computed; if it is None, all the available are calculated (default None)
    ged_method: Optional[str]
        The method to be used for graph similarity calculations (default 'nx_ged')
    ged_params: Optional[Union[dict, None]]
        The dictionary with the parameters for specifying reaction and molecular fingerprints and similarity functions;
        if it is None, the default values are used (default None)
    clustering_method: Optional[Union[str, None]]
        The clustering algorithm to be used for clustering the routes; if it is None, hdbscan
        is used when there are more than 15 routes, Agglomerative Clustering otherwise (default None)
    parallelization: Optional[bool]
        Whether parallel computing should be used where possible (default False)
    n_cpu: Optional[int]
        The number of cpus to be used if parallelization is used (default 8)

    Returns:
    ---------
    output: WorkflowOutput
        Its attributes store the results of the selected functionalities. The outcomes are also written to files.

    Raises:
    --------
    NoValidRoute: if the input file(s) does not contain any valid route

    KeyError: if a selected option is not available

    Example:
    ---------
    >>> output = process_routes({'ibmrxn_file.json': 'ibmrxn',  # path to json file from ibmrxn
    >>>                         'az_file.json': 'az'},         # path to json file from az casp
    >>>                         functionalities=[              # the functionalities to be activated
    >>>                            'compute_descriptors',      # calculation of routes descriptors
    >>>                            'clustering_and_d_matrix',  # calculation of distance matrix and clustering
    >>>                            'merging'])                 # merging of the routes to obtain a "tree"

    """

    params = {
        "input": input_dict,
        "output_format": output_format,
        "mapping": mapping,
        "functionalities": functionalities,
        "mapper": mapper,
        "out_data_model": out_data_model,
        "descriptors": descriptors,
        "ged_method": ged_method,
        "ged_params": ged_params,
        "clustering_method": clustering_method,
        "parallelization": parallelization,
        "n_cpu": n_cpu,
    }
    output = WorkflowBuilder().initiate_workflow(params)
    print("All done!")
    return output


# Supporting functions and classes


# Workflow helper
def get_workflow_options(verbose=False):
    """
    Returns the available options for the 'process_routes' function.

    Parameters:
    -----------
    verbose: Optional[bool]]
        It indicates whether the information should be printed on the screen (default False)

    Returns:
    --------
    available options: dict
        The dictionary listing arguments, options and default values of the 'process_routes' function

    Example:
    --------
    >>> options = get_workflow_options(verbose=True)

    """
    if verbose:
        return print_options()
    else:
        d = {
            "input_dict": {
                "name_or_flags": ["-input_dict"],
                "default": None,
                "required": True,
                "type": dict,
                "choices": None,
                "help": 'Path to the input files and relative casp names in the form "file_path"="casp_name". Available CASP names are {}'.format(
                    list(TranslationStep.casps.keys())
                ),
                "dest": "input_dict",
            },
            "output_format": {
                "name_or_flags": ["-output_format"],
                "default": settings.WORKFLOW.output_format,
                "required": False,
                "type": str,
                "choices": SyngraphWriterFactory.list_writers(),
                "help": "Format of the output file containing the routes",
                "dest": "output_format",
            },
            "mapping": {
                "name_or_flags": ["-mapping"],
                "default": settings.WORKFLOW.mapping,
                "required": False,
                "type": bool,
                "choices": [True, False],
                "help": "Whether the atom-to-atom mapping of the reactions should be performed",
                "dest": "mapping",
            },
            "mapper": {
                "name_or_flags": ["-mapper"],
                "default": settings.FACADE.mapper,
                "required": False,
                "type": str,
                "choices": get_available_mappers(),
                "help": "Which mapper should be used; if None, the mapping pipeline is used",
                "dest": "mapper",
            },
            "functionalities": {
                "name_or_flags": ["-functionalities"],
                "default": settings.WORKFLOW.functionalities,
                "required": False,
                "type": list,
                "choices": Executor().get_workflow_functions(),
                "help": "List of functionalities to be performed",
                "dest": "functionalities",
            },
            "out_data_model": {
                "name_or_flags": ["-out_data_model"],
                "default": settings.FACADE.out_data_model,
                "required": False,
                "type": str,
                "choices": get_available_data_models(),
                "help": "Data model of the output graphs",
                "dest": "out_data_model",
            },
            "descriptors": {
                "name_or_flags": ["-descriptors"],
                "default": settings.FACADE.descriptors,
                "required": False,
                "type": list,
                "choices": get_available_descriptors(),
                "help": "List of descriptors to be calculated; if None, all descriptors are calculated",
                "dest": "descriptors",
            },
            "clustering_method": {
                "name_or_flags": ["-clustering_method"],
                "default": settings.FACADE.clustering_method,
                "required": False,
                "type": str,
                "choices": get_available_clustering(),
                "help": "Method to be used to calculate the GED",
                "dest": "ged_method",
            },
        }
        d.update(get_ged_dict())
        d.update(get_parallelization_dict())
        return d


def print_options():
    """
    Prints the available options for the workflow function.
    """
    print("Workflow options and default:")
    data = get_workflow_options()
    for d, info in data.items():
        print("argument:", d)
        print("     info: ", info["help"])
        print("     default: ", info["default"])
        print("     available options: ", info["choices"])
    return data
