from __future__ import annotations

import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import pandas as pd

from linchemin import settings
from linchemin.cgu.convert import Converter, converter
from linchemin.cgu.route_sanity_check import (
    get_available_route_sanity_checks,
    route_checker,
)
from linchemin.cgu.syngraph import (
    BipartiteSynGraph,
    MonopartiteMolSynGraph,
    MonopartiteReacSynGraph,
    SynGraph,
)
from linchemin.cgu.syngraph_operations import (
    extract_reactions_from_syngraph,
    merge_syngraph,
    remove_reaction_from_syngraph,
)
from linchemin.cgu.translate import (
    TranslationError,
    get_available_data_models,
    get_input_formats,
    get_output_formats,
    translator,
)
from linchemin.cheminfo.atom_mapping import (
    get_available_mappers,
    perform_atom_mapping,
    pipeline_atom_mapping,
    MappingOutput,
)
from linchemin.configuration.defaults import DEFAULT_FACADE
from linchemin.rem.clustering import (
    ClusteringError,
    clusterer,
    get_available_clustering,
    get_clustered_routes_metrics,
)
from linchemin.rem.graph_distance import (
    GraphDistanceError,
    compute_distance_matrix,
    get_available_ged_algorithms,
    get_ged_parameters,
)
from linchemin.rem.route_descriptors import (
    DescriptorError,
    descriptor_calculator,
    find_duplicates,
    get_available_descriptors,
    is_subset,
    get_configuration,
)

"""
Module containing high level functionalities/"user stories" to work in stream; it provides a simplified interface for the user.
The functionalities are implemented as methods of the Functionality class.
"""


class FacadeError(Exception):
    """Base class for exceptions leading to unsuccessful execution of facade functionalities"""

    pass


class UnavailableFunctionality(FacadeError):
    """Raised if the selected functionality is not among the available ones"""

    pass


class Facade(ABC):
    """Definition of the abstract class for high level functionalities' facade."""

    @abstractmethod
    def perform_functionality(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_available_options(self):
        pass

    @abstractmethod
    def print_available_options(self):
        pass


class TranslateFacade(Facade):
    """Subclass of Facade for translating list of routes between formats."""

    functionality = "translate"
    info = ("To translate a list of routes from a format to another one.",)

    def perform_functionality(
        self,
        input_format: str,
        input_list: List,
        out_format: str = settings.FACADE.data_format,
        out_data_model: str = settings.FACADE.out_data_model,
        parallelization: bool = settings.FACADE.parallelization,
        n_cpu: int = settings.FACADE.n_cpu,
    ) -> tuple:
        """
        Takes a list of routes in the specified input format and converts it into the desired format (default: SynGraph).
        Returns the converted routes and some metadata.

        Parameters:
        ------------
        input_format: str
            The format of the input file
        input_list: List
            The list of routes in the specified input format
        out_format: Optional[str]
            The desired output format (format as for the translator factory) (default syngraph)
        out_data_model: Optional[str]
            The desired output data model (default bipartite)
        parallelization: Optional[bool]
            Whether parallelization should be used (default False)
        n_cpu: Optional[int]
            The number of cpus to be used in the parallel calculation (default 8)

        Returns:
        ---------
        tuple
            output: a list whose elements are the converted routes

            meta: a dictionary storing information about the original file and the CASP tool that produced the routes
        """
        exceptions: List = []
        out_routes: List = []

        try:
            if parallelization:
                pool = mp.Pool(n_cpu)
                converted_routes = pool.starmap(
                    translator,
                    [
                        (input_format, route, out_format, out_data_model)
                        for route in input_list
                    ],
                )
            else:
                converted_routes = [
                    translator(input_format, route, out_format, out_data_model)
                    for route in input_list
                ]

            out_routes.extend([r for r in converted_routes if r is not None])

        except TranslationError as te:
            exceptions.append(te)

        except Exception as e:
            exceptions.append(e)

        invalid_routes = len(input_list) - len(out_routes)

        meta = {
            "nr_routes_in_input_list": len(input_list),
            "input_format": input_format,
            "nr_output_routes": len(out_routes),
            "invalid_routes": invalid_routes,
        }

        return out_routes, meta

    def get_available_options(self) -> dict:
        """
        Returns the available options for output formats as a dictionary.
        """
        return {
            "input_list": {
                "name_or_flags": ["-input_list"],
                "default": None,
                "required": True,
                "type": list,
                "choices": None,
                "help": "A list of routes to be translated",
                "dest": "input_list",
            },
            "input_format": {
                "name_or_flags": ["-input_format"],
                "default": None,
                "required": True,
                "type": str,
                "choices": get_input_formats(),
                "help": "CASP tool that generated the input file",
                "dest": "casp",
            },
            "out_format": {
                "name_or_flags": ["-out_format"],
                "default": settings.FACADE.data_format,
                "required": False,
                "type": str,
                "choices": get_output_formats(),
                "help": "Format of the output graphs",
                "dest": "out_format",
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
            "parallelization": {
                "name_or_flags": ["-parallelization"],
                "default": settings.FACADE.parallelization,
                "required": False,
                "type": bool,
                "choices": [True, False],
                "help": "Whether parallelization should be used",
                "dest": "parallelization",
            },
            "n_cpu": {
                "name_or_flags": ["-n_cpu"],
                "default": settings.FACADE.n_cpu,
                "required": False,
                "type": int,
                "choices": None,
                "help": "Number of CPUs to be used in parallelization",
                "dest": "n_cpu",
            },
        }

    def print_available_options(self):
        """
        Prints the available options for output formats.
        """
        print('"Translate" options and default:')
        data = self.get_available_options()
        for d, info in data.items():
            print("argument:", d)
            print("     info: ", info["help"])
            print("     default: ", info["default"])
            print("     available options: ", info["choices"])
        return data


class RoutesDescriptorsFacade(Facade):
    """Subclass of Facade for computing routes metrics."""

    info = "To compute metrics of a list of SynGraph objects"

    def perform_functionality(
        self,
        routes: List[
            Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
        ],
        descriptors: Union[List[str], None] = settings.FACADE.descriptors,
    ) -> tuple:
        """
        Computes the desired descriptors (default: all the available descriptors) for the routes in the provided list.

        Parameters:
        ------------
        routes: List[Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]]
            The list of SynGraph instances
        descriptors: Optional[Union[List, None]]
            The list of strings indicating the desired descriptors to be computed
            (default None -> all the available descriptors)

        Returns:
        --------
        tuple
            output: a pandas dataframe

            meta: a dictionary storing information about the computed descriptors
        """

        output = pd.DataFrame()

        if descriptors is None:
            descriptors = get_available_descriptors()
            descriptors.pop("all_paths")

        exceptions = []
        checked_routes = [r for r in routes if r is not None]
        invalid_routes = len(routes) - len(checked_routes)
        output["route_id"] = [route.uid for route in checked_routes]
        configuration: List = []

        for d in descriptors:
            try:
                output[d] = [
                    descriptor_calculator(route, d) for route in checked_routes
                ]
                configuration.append(get_configuration(d))
            except DescriptorError as ke:
                exceptions.append(ke)

            except Exception as e:
                exceptions.append(e)
        output.configuration = configuration
        meta = {
            "descriptors": descriptors,
            "invalid_routes": invalid_routes,
            "errors": exceptions,
        }
        return output, meta

    def get_available_options(self) -> dict:
        """
        Returns the available options for route descriptors as a dictionary.
        """
        return {
            "routes": {
                "name_or_flags": ["-routes"],
                "default": None,
                "required": True,
                "type": List[SynGraph],
                "choices": None,
                "help": "List of SynGraph objects for which the selected descriptors should be calculated",
                "dest": "routes",
            },
            "descriptors": {
                "name_or_flags": ["-descriptors"],
                "default": DEFAULT_FACADE["routes_descriptors"]["value"],
                "required": False,
                "type": List[str],
                "choices": get_available_descriptors(),
                "help": "List of descriptors to be calculated",
                "dest": "descriptors",
            },
        }

    def print_available_options(self):
        """
        Prints the available options for routes descriptors.
        """
        print("Routes descriptors options and default:")
        data = self.get_available_options()
        for d, info in data.items():
            print("argument:", d)
            print("     info: ", info["help"])
            print("     default: ", info["default"])
            print("     available options: ", info["choices"])
        return data


class GedFacade(Facade):
    """Subclass of Facade for computing the distance matrix of a list of routes with GED algorithms."""

    info = "To compute the distance matrix of a list of SynGraph objects via a Graph Edit Distance algorithm"

    def perform_functionality(
        self,
        routes: List[
            Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
        ],
        ged_method: str = DEFAULT_FACADE["distance_matrix"]["value"]["ged_method"],
        ged_params: Union[dict, None] = DEFAULT_FACADE["distance_matrix"]["value"][
            "ged_params"
        ],
        parallelization: bool = DEFAULT_FACADE["distance_matrix"]["value"][
            "parallelization"
        ],
        n_cpu: int = DEFAULT_FACADE["distance_matrix"]["value"]["n_cpu"],
    ) -> tuple:
        """
        Computes the distance matrix for the routes in the provided list.

        Parameters:
        ------------
        routes: List[Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]]
            The list of SynGraph instances for which should be computed; it is recommended to use the monopartite
            representation for performance reasons
        ged_method: Optional[str]
            The GED algorithm to be used (default 'nx_optimized_ged')
        ged_params: Optional[Union[dict, None]]
            The optional parameters for chemical similarity and fingerprints
            (default None ->the default parametrs are used)
        parallelization: Optional[bool]
            Whether parallelization should be used (default False)
        n_cpu: Optional[int]
            The number of cpus to be used in the parallel calculation (default 8)

        Returns:
        -----------
        tuple
            dist_matrix: a pandas DataFrame (n routes) x (n routes) with the ged values

            meta: a dictionary storing information about the type of graph (mono or bipartite), the algorithm used
                  for the ged calculations and the parameters for chemical similarity and fingerprints
        """

        exceptions: List = []
        checked_routes = [r for r in routes if r is not None]
        try:
            dist_matrix = compute_distance_matrix(
                checked_routes,
                ged_method=ged_method,
                ged_params=ged_params,
                parallelization=parallelization,
                n_cpu=n_cpu,
            )
            meta = {
                "ged_algorithm": ged_method,
                "ged_params": ged_params,
                "graph_type": "monopartite"
                if type(routes[0]) == MonopartiteReacSynGraph
                else "bipartite",
                "invalid_routes": len(routes) - len(checked_routes),
                "errors": exceptions,
            }

        except GraphDistanceError as ke:
            exceptions.append(ke)
            meta = {
                "ged_algorithm": ged_method,
                "ged_params": ged_params,
                "graph_type": "monopartite"
                if type(routes[0]) == MonopartiteReacSynGraph
                else "bipartite",
                "invalid_routes": len(routes) - len(checked_routes),
                "errors": exceptions,
            }
            dist_matrix = pd.DataFrame()
        return dist_matrix, meta

    def get_available_options(self) -> dict:
        """
        Returns the available options for GED calculations.
        """
        return {
            "routes": {
                "name_or_flags": ["-routes"],
                "default": None,
                "required": True,
                "type": List[SynGraph],
                "choices": None,
                "help": "List of SynGraph objects for which the distance matrix should be calculated",
                "dest": "routes",
            },
            "ged_method": {
                "name_or_flags": ["-ged_method"],
                "default": DEFAULT_FACADE["distance_matrix"]["value"]["ged_method"],
                "required": False,
                "type": str,
                "choices": get_available_ged_algorithms(),
                "help": "Method to be used to calculate the GED",
                "dest": "ged_method",
            },
            "ged_params": {
                "name_or_flags": ["-ged_params"],
                "default": DEFAULT_FACADE["distance_matrix"]["value"]["ged_params"],
                "required": False,
                "type": dict,
                "choices": get_ged_parameters(),
                "help": "Parameters of the molecular and/or reaction chemical similarity",
                "dest": "ged_params",
            },
            "parallelization": {
                "name_or_flags": ["-parallelization"],
                "default": DEFAULT_FACADE["distance_matrix"]["value"][
                    "parallelization"
                ],
                "required": False,
                "type": bool,
                "choices": [True, False],
                "help": "Whether parallelization should be used",
                "dest": "parallelization",
            },
            "n_cpu": {
                "name_or_flags": ["-n_cpu"],
                "default": 'All the available CPUs as given by "multiprocessing.cpu_count()"',
                "required": False,
                "type": int,
                "choices": None,
                "help": "Number of CPUs to be used in parallelization",
                "dest": "n_cpu",
            },
        }

    def print_available_options(self):
        """
        Prints the available options for GED calculations as a dictionary.
        """
        print("GED options and default:")
        data = self.get_available_options()
        for d, info in data.items():
            print("argument:", d)
            print("     info: ", info["help"])
            print("     default: ", info["default"])
            print("     available options: ", info["choices"])
        return data


class ClusteringFacade:
    """Facade Factory to give access to the functionalities"""

    info = "To cluster a list of SynGraph objects"

    def perform_functionality(
        self,
        routes: List[
            Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
        ],
        clustering_method: Union[str, None] = DEFAULT_FACADE["clustering"]["value"][
            "clustering_method"
        ],
        ged_method: str = DEFAULT_FACADE["clustering"]["value"]["ged_method"],
        ged_params: Union[dict, None] = DEFAULT_FACADE["clustering"]["value"][
            "ged_params"
        ],
        save_dist_matrix: bool = DEFAULT_FACADE["clustering"]["value"][
            "save_dist_matrix"
        ],
        compute_metrics: bool = DEFAULT_FACADE["clustering"]["value"][
            "compute_metrics"
        ],
        parallelization: bool = DEFAULT_FACADE["clustering"]["value"][
            "parallelization"
        ],
        n_cpu: int = DEFAULT_FACADE["clustering"]["value"]["n_cpu"],
        **kwargs,
    ):
        """
        Performs clustering of the routes in the provided list.

        Parameters:
        -----------
        routes: List[Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]]
            The input list of SynGraph
        clustering_method: Optional[Union[str, None]]
            The clustering algorithm to be used. If None is given,
            agglomerative_cluster is used when there are less than 15 routes, otherwise hdbscan is used (default None)
        ged_method: Optional[str]
            The GED algorithm to be used (default 'nx_optimized_ged')
        ged_params: Union[dict, None]
            The optional parameters for chemical similarity and fingerprints
            (default None ->the default parametrs are used)
        save_dist_matrix: Optional[bool]
            Whether the distance matrix should be saved and returned as output
        compute_metrics: Optional[bool]
            Whether the average metrics for each cluster should be computed
        arallelization: Optional[bool]
            Whether parallelization should be used (default False)
        n_cpu: Optional[int]
            The number of cpus to be used in the parallel calculation (default 8)
        kwargs: the type of linkage can be indicated when using the agglomerative_cluster; the minimum size of the
                clusters can be indicated when using hdbscan

        Returns:
        ----------
        tuple
            results: a tuple with: clustering, score, (dist_matrix), corresponding to the output of the clustering,
                                 its silhouette score (and the distance matrix as a pandas dataframe if save_dist_matrix=True)

            meta: a dictionary storing information about the original file and the CASP tool that produced the routes,
                the type of graph (mono or bipartite), information regarding the clustering, information regarding
                the ged calculations and the parameters for chemical similarity and fingerprints
        """

        if clustering_method is None:
            clustering_method = (
                "hdbscan" if len(routes) > 15 else "agglomerative_cluster"
            )

        exceptions: List = []
        checked_routes = [r for r in routes if r is not None]
        metrics = pd.DataFrame()
        try:
            results = clusterer(
                checked_routes,
                ged_method=ged_method,
                clustering_method=clustering_method,
                ged_params=ged_params,
                save_dist_matrix=save_dist_matrix,
                parallelization=parallelization,
                n_cpu=n_cpu,
                **kwargs,
            )
            meta = {
                "graph_type": "monopartite"
                if type(routes[0]) == MonopartiteReacSynGraph
                else "bipartite",
                "clustering_algorithm": clustering_method,
                "clustering_params": kwargs,
                "ged_algorithm": ged_method,
                "ged_parameters": ged_params,
                "invalid_routes": len(routes) - len(checked_routes),
                "errors": exceptions,
            }

            if compute_metrics:
                metrics = get_clustered_routes_metrics(routes, results[0])

        except ClusteringError as sre:
            exceptions.append(sre)
            meta = {
                "graph_type": "monopartite"
                if type(checked_routes[0]) == MonopartiteReacSynGraph
                else "bipartite",
                "clustering_algorithm": clustering_method,
                "clustering_params": kwargs,
                "ged_algorithm": ged_method,
                "ged_parameters": ged_params,
                "invalid_routes": len(routes) - len(checked_routes),
                "errors": exceptions,
            }
            results = None

        return (results, metrics, meta) if compute_metrics else (results, meta)

    def get_available_options(self) -> dict:
        """
        Returns the available options for clustering calculations.
        """
        return {
            "routes": {
                "name_or_flags": ["-routes"],
                "default": None,
                "required": True,
                "type": List[SynGraph],
                "choices": None,
                "help": "List of SynGraph objects to be clustered",
                "dest": "routes",
            },
            "clustering_method": {
                "name_or_flags": ["-clustering_method"],
                "default": DEFAULT_FACADE["clustering"]["value"]["clustering_method"],
                "required": False,
                "type": str,
                "choices": get_available_clustering(),
                "help": "Method to be used to calculate the GED",
                "dest": "ged_method",
            },
            "ged_method": {
                "name_or_flags": ["-ged_method"],
                "default": DEFAULT_FACADE["distance_matrix"]["value"]["ged_method"],
                "required": False,
                "type": str,
                "choices": get_available_ged_algorithms(),
                "help": "Method to be used to calculate the GED",
                "dest": "ged_method",
            },
            "ged_params": {
                "name_or_flags": ["-ged_params"],
                "default": DEFAULT_FACADE["distance_matrix"]["value"]["ged_params"],
                "required": False,
                "type": dict,
                "choices": get_ged_parameters(),
                "help": "Parameters of the molecular and/or reaction chemical similarity",
                "dest": "ged_params",
            },
            "save_dist_matrix": {
                "name_or_flags": ["-save_dist_matrix"],
                "default": DEFAULT_FACADE["clustering"]["value"]["save_dist_matrix"],
                "required": False,
                "type": bool,
                "choices": [True, False],
                "help": "Whether the distance matrix should be saved and returned as output",
                "dest": "save_dist_matrix",
            },
            "compute_metrics": {
                "name_or_flags": ["-compute_metrics"],
                "default": DEFAULT_FACADE["clustering"]["value"]["compute_metrics"],
                "required": False,
                "type": bool,
                "choices": [True, False],
                "help": "Whether metrics aggregated by clusters should be computed",
                "dest": "compute_metrics",
            },
            "parallelization": {
                "name_or_flags": ["-parallelization"],
                "default": DEFAULT_FACADE["clustering"]["value"]["parallelization"],
                "required": False,
                "type": bool,
                "choices": [True, False],
                "help": "Whether parallelization should be used",
                "dest": "parallelization",
            },
            "n_cpu": {
                "name_or_flags": ["-n_cpu"],
                "default": 'All the available CPUs as given by "multiprocessing.cpu_count()"',
                "required": False,
                "type": int,
                "choices": None,
                "help": "Number of CPUs to be used in parallelization",
                "dest": "n_cpu",
            },
        }

    def print_available_options(self):
        """
        Prints the available options for clustering calculations as a dictionary.
        """
        print("Clustering options and default:")
        data = self.get_available_options()
        for d, info in data.items():
            print("argument:", d)
            print("     info: ", info["help"])
            print("     default: ", info["default"])
            print("     available options: ", info["choices"])
        return data


class SubsetsFacade(Facade):
    """Subclass of Facade for searching subsets in a list of routes."""

    info = "To check whether there are subsets in a list of SynGraph objects"

    def perform_functionality(self, routes: List) -> list:
        """
        To check the presence of subsets in the provided list of SynGraph objects.

        Parameters:
        -----------
        routes: list
            The list of SynGraph instances

        Returns:
        --------
        subsets: list
            The list of uid pairs in which the first is subset of the second
        """
        subsets = []
        for route1 in routes:
            subsets.extend(
                [route1.uid, route2.uid]
                for route2 in routes
                if is_subset(route1, route2)
            )

        return subsets

    def get_available_options(self) -> dict:
        return {
            "routes": {
                "name_or_flags": ["-routes"],
                "default": None,
                "required": True,
                "type": List[SynGraph],
                "choices": None,
                "help": "List of SynGraph objects in which subsets should be searched",
                "dest": "routes",
            }
        }

    def print_available_options(self):
        print("Subset search options and default:")
        data = self.get_available_options()
        for d, info in data.items():
            print("argument:", d)
            print("     info: ", info["help"])
            print("     default: ", info["default"])
            print("     available options: ", info["choices"])
        return data


class DuplicatesFacade(Facade):
    """Subclass of Facade for searching duplicates in a list of routes."""

    info = "To check whether there are duplicates in a list of SynGraph objects"

    def perform_functionality(self, routes: list):
        """
        To check the presence of duplicated routes in the provided list of SynGraph instances.

        Parameters:
        ------------
        routes: list
            The list of SynGraph instances

        Returns:
        list: The list of duplicates
        """
        routes1 = routes[: len(routes) // 2]
        routes2 = routes[len(routes) // 2 :]
        return find_duplicates(routes1, routes2)

    def get_available_options(self) -> dict:
        return {
            "routes": {
                "name_or_flags": ["-routes"],
                "default": None,
                "required": True,
                "type": List[SynGraph],
                "choices": None,
                "help": "List of SynGraph objects in which duplicates should be searched",
                "dest": "routes",
            }
        }

    def print_available_options(self):
        print("Duplicates search options and default:")
        data = self.get_available_options()
        for d, info in data.items():
            print("argument:", d)
            print("     info: ", info["help"])
            print("     default: ", info["default"])
            print("     available options: ", info["choices"])
        return data


class MergingFacade(Facade):
    """Subclass of Facade for merging a list of routes."""

    info = "To merge a list of SynGraph objects"

    def perform_functionality(
        self,
        routes: List[
            Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
        ],
        out_data_model: str = DEFAULT_FACADE["merging"]["value"]["out_data_model"],
    ) -> Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]:
        """
        To merge the provided list of SynGraph instances in a new SynGraph instance.

        Parameters:
        -------------
        routes: list
            The list of SynGraph instances
        out_data_model: str
            the data model of the output SynGraph

        Returns:
        ----------
        merged: Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
            The new SynGraph instance resulting from the merging of the input routes
        """
        routes_checked_type = [converter(r, out_data_model) for r in routes]
        merged_graph = merge_syngraph(routes_checked_type)
        return converter(merged_graph, out_data_model)

    def get_available_options(self) -> dict:
        return {
            "routes": {
                "name_or_flags": ["-routes"],
                "default": None,
                "required": True,
                "type": List[SynGraph],
                "choices": None,
                "help": "List of SynGraph objects in which duplicates should be searched",
                "dest": "routes",
            },
            "out_data_model": {
                "name_or_flags": ["-out_data_model"],
                "default": DEFAULT_FACADE["merging"]["value"]["out_data_model"],
                "required": False,
                "type": str,
                "choices": get_available_data_models(),
                "help": "Data model of the output graphs",
                "dest": "out_data_model",
            },
        }

    def print_available_options(self):
        print("Merging search options and default:")
        data = self.get_available_options()
        for d, info in data.items():
            print("argument:", d)
            print("     info: ", info["help"])
            print("     default: ", info["default"])
            print("     available options: ", info["choices"])
        return data


class ReactionExtractionFacade(Facade):
    """Subclass of Facade for extracting unique reaction strings from a list of routes."""

    info = (
        "To extract a list of unique reaction strings from a list of SynGraph objects"
    )

    def perform_functionality(
        self,
        routes: List[
            Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
        ],
    ) -> Tuple[List, dict]:
        """
        To extract a list of reaction strings

        Parameters:
        ---------------
        routes: List[Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]]
            The list of SynGraph instances

        Returns:
        ---------
        Tuple[List, dict]:
            output: the list of dictionaries with the extracted reactions smiles
            meta: the dictionary storing information about the run

        """
        checked_routes = [r for r in routes if r != {}]

        exceptions: list = []
        output: list = []
        try:
            for route in checked_routes:
                reactions = extract_reactions_from_syngraph(route)
                output.append({route.uid: reactions})

        except TypeError as ke:
            print("Found route in wrong format: only SynGraph object are accepted.")
            exceptions.append(ke)

        except Exception as e:
            exceptions.append(e)

        invalid_routes = len(routes) - len(checked_routes)
        meta = {
            "nr_routes_in_input_list": len(routes),
            "invalid_routes": invalid_routes,
            "errors": exceptions,
        }

        return output, meta

    def get_available_options(self) -> dict:
        return {
            "routes": {
                "name_or_flags": ["-routes"],
                "default": None,
                "required": True,
                "type": List[SynGraph],
                "choices": None,
                "help": "List of SynGraph objects from which reaction strings should be extracted",
                "dest": "routes",
            }
        }

    def print_available_options(self):
        print("Reaction strings extraction options and default:")
        data = self.get_available_options()
        for d, info in data.items():
            print("argument:", d)
            print("     info: ", info["help"])
            print("     default: ", info["default"])
            print("     available options: ", info["choices"])
        return data


class AtomMappingFacade(Facade):
    """Subclass of Facade for mapping the chemical equations in a list of routes."""

    info = "To get a list of mapped SynGraph objects"

    def perform_functionality(
        self,
        routes: List[
            Union[BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph]
        ],
        mapper: Union[str, None] = DEFAULT_FACADE["atom_mapping"]["value"]["mapper"],
        out_data_model: str = DEFAULT_FACADE["atom_mapping"]["value"]["out_data_model"],
    ) -> Tuple[List, dict]:
        """
        To generate a list of SynGraph objects with mapped chemical equations

        Parameters:
        ------------
        routes: List[Union[BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph]]
            The list of SynGraph instances
        mapper: Optional[Union[str, None]]
            The mapper to be used; if it is not provided, the mapping pipeline is used (default rxnmapper)
        out_data_model: Optional[str]
            The desired output data model (default bipartite)

        Returns:
        ---------
        Tuple[List, dict]:
            output: a list of SynGraph objects with mapped chemical reaction
            meta: The dictionary storing information about the run
        """
        out_syngraphs: List = []
        out_syngraph_type = Converter.out_datamodels[out_data_model]
        tot_success_rate: Union[float, int] = 0
        exceptions = []
        for route in routes:
            source = route.source
            route_id = route.uid
            reaction_list = extract_reactions_from_syngraph(route)
            mapping_out = self.map_reaction_strings(mapper, reaction_list)
            try:
                mapped_route = out_syngraph_type(mapping_out.mapped_reactions)
                mapped_route.source = source
                out_syngraphs.append(mapped_route)
            except Exception as e:
                exceptions.append({"route_uid": route_id, "exception": e})

            tot_success_rate += mapping_out.success_rate

        tot_success_rate = float(tot_success_rate / len(routes))

        meta = {
            "mapper": mapper,
            "mapping_success_rate": tot_success_rate,
            "exception": exceptions,
            "nr_invalid_routes": len(routes) - len(out_syngraphs),
        }

        return out_syngraphs, meta

    @staticmethod
    def map_reaction_strings(
        mapper: Union[None, str], reaction_list: List
    ) -> MappingOutput:
        """To perform the mapping of the reaction strings"""
        if mapper is None:
            # the full pipeline is used
            mapping_out = pipeline_atom_mapping(reaction_list)
        else:
            # the selected mapper is used
            mapping_out = perform_atom_mapping(reaction_list, mapper)
        if mapping_out.success_rate != 1:
            # if not all the reactions are mapped, a warning is raised and
            # the output graph is built using all the mapped and the unmapped reactions (so that it is complete)
            mapping_out.mapped_reactions.extend(mapping_out.unmapped_reactions)
        return mapping_out

    def get_available_options(self) -> dict:
        return {
            "routes": {
                "name_or_flags": ["-routes"],
                "default": None,
                "required": True,
                "type": List[SynGraph],
                "choices": None,
                "help": "List of SynGraph objects in which duplicates should be searched",
                "dest": "routes",
            },
            "mapper": {
                "name_or_flags": ["-mapper"],
                "default": DEFAULT_FACADE["atom_mapping"]["value"]["mapper"],
                "required": False,
                "type": str,
                "choices": get_available_mappers(),
                "help": "Which mapper should be used",
                "dest": "mapper",
            },
            "out_data_model": {
                "name_or_flags": ["-out_data_model"],
                "default": DEFAULT_FACADE["atom_mapping"]["value"]["out_data_model"],
                "required": False,
                "type": str,
                "choices": get_available_data_models(),
                "help": "Data model of the output graphs",
                "dest": "out_data_model",
            },
        }

    def print_available_options(self):
        print("Atom mapping of routes reactions options and default:")
        data = self.get_available_options()
        for d, info in data.items():
            print("argument:", d)
            print("     info: ", info["help"])
            print("     default: ", info["default"])
            print("     available options: ", info["choices"])
        return data


class RouteSanityCheckFacade(Facade):
    """Subclass of Facade for performing sanity checks on a list of routes."""

    info = "To perform sanity checks on a list of routes"

    def perform_functionality(
        self,
        routes: List[
            Union[BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph]
        ],
        checks: Union[List[str], None] = DEFAULT_FACADE["route_sanity_checks"]["value"][
            "checks"
        ],
        out_data_model: str = DEFAULT_FACADE["route_sanity_checks"]["value"][
            "out_data_model"
        ],
    ) -> Tuple[List, dict]:
        """
        Returns a list of routes in which possible issues are removed

        Parameters:
        ------------
        routes: List[Union[BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph]]
            The list of SynGraph instances
        check: Optional[Union[List[str], None]]
            The list of sanity checks to be performed; if it is not provided, all the available
            sanity check are applied (default None)
        out_data_model: Optional[str]
            The desired output data model (default bipartite)

        Returns:
        ---------
        Tuple[List, dict]:
            output: The list of checked SynGraph objects

            meta: The dictionary storing information about the run
        """
        if checks is None:
            checks = get_available_route_sanity_checks()

        checked_routes: List = []
        exceptions = []
        valid_routes = [r for r in routes if r is not None]
        invalid_routes = len(routes) - len(valid_routes)

        for r in valid_routes:
            checked_route = r
            for check in checks:
                try:
                    checked_route = route_checker(checked_route, check, fix_issue=True)
                except Exception as ke:
                    exceptions.append(ke)
            checked_routes.append(checked_route)

        out_routes = [converter(r, out_data_model) for r in checked_routes]

        meta = {
            "checks": checks,
            "invalid_routes": invalid_routes,
            "errors": exceptions,
        }
        return out_routes, meta

    def get_available_options(self):
        """
        Returns the available options for route sanity checks as a dictionary.
        """
        return {
            "routes": {
                "name_or_flags": ["-routes"],
                "default": None,
                "required": True,
                "type": List[SynGraph],
                "choices": None,
                "help": "List of SynGraph objects for which the selected sanity checks should be performed",
                "dest": "routes",
            },
            "checks": {
                "name_or_flags": ["-checks"],
                "default": DEFAULT_FACADE["route_sanity_checks"]["value"],
                "required": False,
                "type": List[str],
                "choices": get_available_descriptors(),
                "help": "List of sanity checks to be performed",
                "dest": "checks",
            },
            "out_data_model": {
                "name_or_flags": ["-out_data_model"],
                "default": DEFAULT_FACADE["route_sanity_checks"]["value"][
                    "out_data_model"
                ],
                "required": False,
                "type": str,
                "choices": get_available_data_models(),
                "help": "Data model of the output graphs",
                "dest": "out_data_model",
            },
        }

    def print_available_options(self):
        """
        Prints the available options for routes sanity checks.
        """
        print("Routes sanity checks options and default:")
        data = self.get_available_options()
        for d, info in data.items():
            print("argument:", d)
            print("     info: ", info["help"])
            print("     default: ", info["default"])
            print("     available options: ", info["choices"])
        return data


class NodeRemovalFacade(Facade):
    """Subclass of Facade for removing a node and its connections from a list of SynGraph."""

    info = "To remove a node from a list of routes"

    def perform_functionality(
        self,
        routes: List[
            Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
        ],
        node_to_remove: str,
    ) -> Tuple[List, dict]:
        """
        Returns a list of routes from which the selected node and its connections are removed

        Parameters:
        ------------
        routes: List[Union[BipartiteSynGraph, MonopartiteReacSynGraph, MonopartiteMolSynGraph]]
            The list of SynGraph instances
        node_to_remove: str
            The smiles of the reaction to be removed

        Returns:
        ---------
        Tuple[List, dict]:
            output: The list of SynGraph objects without the selected node and its connections

            meta: The dictionary storing information about the run
        """
        exceptions = []
        valid_routes = [r for r in routes if r is not None]
        unchanged_routes: List = []
        modified_routes: List = []
        invalid_routes = len(routes) - len(valid_routes)

        for r in valid_routes:
            try:
                modified_route = remove_reaction_from_syngraph(r, node_to_remove)
                if modified_route == r:
                    unchanged_routes.append(r)
                else:
                    modified_routes.append(modified_route)

            except Exception as ke:
                exceptions.append(ke)

        meta = {
            "removed_node": node_to_remove,
            "invalid_routes": invalid_routes,
            "modified_routes": len(modified_routes),
            "unchanged_routes": len(unchanged_routes),
            "errors": exceptions,
        }
        out_routes = unchanged_routes + modified_routes
        return out_routes, meta

    def get_available_options(self):
        """
        Returns the available options for route sanity checks as a dictionary.
        """
        return {
            "routes": {
                "name_or_flags": ["-routes"],
                "default": None,
                "required": True,
                "type": List[SynGraph],
                "choices": None,
                "help": "List of SynGraph objects from which the selected node should be removed",
                "dest": "routes",
            },
            "node_to_remove": {
                "name_or_flags": ["-node_to_remove"],
                "default": None,
                "required": True,
                "type": str,
                "choices": None,
                "help": "Smiles of the node to be removed",
                "dest": "node_to_remove",
            },
        }

    def print_available_options(self):
        """
        Prints the available options for routes sanity checks.
        """
        print("Node removal options and default:")
        data = self.get_available_options()
        for d, info in data.items():
            print("argument:", d)
            print("     info: ", info["help"])
            print("     default: ", info["default"])
            print("     available options: ", info["choices"])
        return data


class FacadeFactory:
    functionalities = {
        "translate": {"value": TranslateFacade, "info": TranslateFacade.info},
        "routes_descriptors": {
            "value": RoutesDescriptorsFacade,
            "info": RoutesDescriptorsFacade.info,
        },
        "distance_matrix": {"value": GedFacade, "info": GedFacade.info},
        "clustering": {"value": ClusteringFacade, "info": ClusteringFacade.info},
        "subsets": {"value": SubsetsFacade, "info": SubsetsFacade.info},
        "duplicates": {"value": DuplicatesFacade, "info": DuplicatesFacade.info},
        "merging": {"value": MergingFacade, "info": MergingFacade.info},
        "extract_reactions_strings": {
            "value": ReactionExtractionFacade,
            "info": ReactionExtractionFacade.info,
        },
        "atom_mapping": {"value": AtomMappingFacade, "info": AtomMappingFacade.info},
        "routes_sanity_checks": {
            "value": RouteSanityCheckFacade,
            "info": RouteSanityCheckFacade.info,
        },
        "node_removal": {"value": NodeRemovalFacade, "info": NodeRemovalFacade.info},
    }

    def select_functionality(self, functionality: str, *args, **kwargs):
        """Takes a string indicating a functionality and its arguments and performs the functionality"""
        if functionality not in self.functionalities:
            raise UnavailableFunctionality(
                f"'{functionality}' is not a valid functionality."
                f"Available functionalities are: {self.functionalities.keys()}"
            )

        performer = self.functionalities[functionality]["value"]
        return performer().perform_functionality(*args, **kwargs)

    def get_helper(self, functionality):
        """Takes a string indicating a functionality and returns the available options"""
        helper = self.functionalities[functionality]["value"]
        return helper().get_available_options()

    def get_helper_verbose(self, functionality):
        """Takes a string indicating a functionality and prints the available options"""
        helper = self.functionalities[functionality]["value"]
        helper().print_available_options()
        return helper().get_available_options()


def facade(functionality: str, *args, **kwargs):
    """
    To perform one of the main functionality of the package.

    Parameters:
    -------------
    functionality: str
        The name of the functionality to be performed
    *args:
        the mandatory arguments of the selected functionality
    **kwargs:
        the optional arguments of the selected functionality

    Returns:
    --------
    the output of the selected functionality

    """
    functionality_selector = FacadeFactory()
    return functionality_selector.select_functionality(functionality, *args, **kwargs)


def facade_helper(
    functionality: Union[str, None] = None, verbose: bool = False
) -> dict:
    """
    Returns the available facade functions if no input is provided; if the name of a functionality is specified,
    the available parameters options for it are returned.

    Parameters:
    ------------
    functionality: Optional[Union[str, None]]
        If provided, it indicates the functionality for which the helper is invoked. If it is None,
        the helper for the facade is returned. (default None)

    verbose: Optional[bool]
        Whether to print the available options and the default parameters are printed on the screen (default False)

    Returns:
    ----------
    dict:
       A dictionary with the available options and default parameters.

    Example:
    >>> # to get info for the entire facade and print information on the screen
    >>> facade_helper(verbose=True)
    >>> # to get info for a specific functionality and return it as a dictionary
    >>> info = facade_helper(functionality='translate')
    """
    if functionality is None:
        av_functionalities = {
            f: additional_info["info"]
            for f, additional_info in FacadeFactory.functionalities.items()
        }
        if verbose:
            print("Available functionalities are:")
            for f, info in av_functionalities.items():
                print(" ", f, ":", info)
        return av_functionalities

    helper_selector = FacadeFactory()
    if verbose:
        return helper_selector.get_helper_verbose(functionality)
    return helper_selector.get_helper(functionality)


if __name__ == "__main__":
    print("main")
