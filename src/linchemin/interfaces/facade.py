# standard library imports
import json
import multiprocessing as mp
import pprint
from abc import ABC, abstractmethod

import pandas as pd

from linchemin.cgu.convert import converter
from linchemin.cgu.syngraph import (
    MonopartiteReacSynGraph,
    SynGraph,
    extract_reactions_from_syngraph,
    merge_syngraph,
)

# local imports
from linchemin.cgu.translate import (
    get_available_data_models,
    get_available_formats,
    get_input_formats,
    get_output_formats,
    translator,
)
from linchemin.rem.clustering import (
    NoClustering,
    SingleRouteClustering,
    clusterer,
    get_available_clustering,
    get_clustered_routes_metrics,
)
from linchemin.rem.graph_distance import (
    compute_distance_matrix,
    get_available_ged_algorithms,
    get_ged_default_parameters,
    get_ged_parameters,
)
from linchemin.rem.route_descriptors import (
    NoneInput,
    descriptor_calculator,
    find_duplicates,
    get_available_descriptors,
    is_subset,
)

"""
Module containing high level functionalities/"user stories" to work in stream; it provides a simplified interface for the user.
The functionalities are implemented as methods of the Functionality class.

    AbstractClasses::
        Facade
        
    Classes:
        FacadeFactory
        
        TranslateFacade(Facade)
        # ConvertAndWriteFacade(Facade)
        RoutesMetricsFacade(Facade)
        GedFacade(Facade)
        ClusteringFacade
        SubsetsFacade(Facade)
        DuplicatesFacade(Facade)
    
    Functions
        facade
        facade_helper
"""

DEFAULT_FACADE = {
    "translate": {
        "value": {
            "data_format": "syngraph",
            "data_model": "bipartite",
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
            "n_cpu": mp.cpu_count(),
        },
        "info": "AgglomerativeClustering is used when there are less than 15 routes, hdbscan otherwise."
        "The standard NetworkX algorithm and default parameters are used for the distance matrix"
        "calculation",
    },
    "merging": {
        "value": {"out_data_model": "bipartite"},
        "info": 'A new, "merged", bipartite SynGraph',
    },
}


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
        input_list: list,
        out_format=DEFAULT_FACADE["translate"]["value"]["data_format"],
        out_data_model=DEFAULT_FACADE["translate"]["value"]["data_model"],
        parallelization=DEFAULT_FACADE["translate"]["value"]["parallelization"],
        n_cpu=DEFAULT_FACADE["translate"]["value"]["n_cpu"],
    ) -> tuple:
        """
        Takes a list of routes in the specified input format and converts it into the desired format (default: SynGraph).
        Returns the converted routes and some metadata.

        Parameters:
            input_list: a list of routes in the specified input format
            input_format: a string indicating the format of the input file
            out_format: a string indicating the desired output format (format as for the translator factory)
            out_data_model: a string indicting the desired output data model
            parallelization: a boolean indicating whether parallelization should be used
            n_cpu: an integer specifying the number of cpus to be used in the parallel calculation

        Returns:
            output: a list whose elements are the converted routes
            meta: a dictionary storing information about the original file and the CASP tool that produced the routes
        """

        # Removing empty routes in the input list
        checked_routes = [r for r in input_list if r != {}]

        if parallelization:
            pool = mp.Pool(n_cpu)
            converted_routes = pool.starmap(
                translator,
                [
                    (input_format, route, out_format, out_data_model)
                    for route in checked_routes
                ],
            )
        else:
            converted_routes = [
                translator(input_format, route, out_format, out_data_model)
                for route in checked_routes
            ]
        out_routes = [r for r in converted_routes if r is not None]
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
                "default": DEFAULT_FACADE["translate"]["value"]["data_format"],
                "required": False,
                "type": str,
                "choices": get_output_formats(),
                "help": "Format of the output graphs",
                "dest": "out_format",
            },
            "out_data_model": {
                "name_or_flags": ["-out_data_model"],
                "default": DEFAULT_FACADE["translate"]["value"]["data_model"],
                "required": False,
                "type": str,
                "choices": get_available_data_models(),
                "help": "Data model of the output graphs",
                "dest": "out_data_model",
            },
            "parallelization": {
                "name_or_flags": ["-parallelization"],
                "default": DEFAULT_FACADE["translate"]["value"]["parallelization"],
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
        routes: list,
        descriptors=DEFAULT_FACADE["routes_descriptors"]["value"]["descriptors"],
    ) -> tuple:
        """
        Computes the desired descriptors (default: all the available descriptors) for the routes in the provided list.

        Parameters:
            routes: a list of SynGraph instances
            descriptors: a list of strings indicating the desired descriptors to be computed
                    (default: None -> all the available descriptors)

        Returns:
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

        try:
            output["route_id"] = [route.source for route in checked_routes]

            for m in descriptors:
                values = [descriptor_calculator(route, m) for route in checked_routes]
                output[m] = values

            meta = {
                "descriptors": descriptors,
                "invalid_routes": invalid_routes,
                "errors": exceptions,
            }

            return output, meta

        except KeyError as ke:
            print(ke)
            exceptions.append(ke)
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
                "type": list[SynGraph],
                "choices": None,
                "help": "List of SynGraph objects for which the selected descriptors should be calculated",
                "dest": "routes",
            },
            "descriptors": {
                "name_or_flags": ["-descriptors"],
                "default": DEFAULT_FACADE["routes_descriptors"]["value"],
                "required": False,
                "type": list[str],
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
        routes: list,
        ged_method=DEFAULT_FACADE["distance_matrix"]["value"]["ged_method"],
        ged_params=DEFAULT_FACADE["distance_matrix"]["value"]["ged_params"],
        parallelization=DEFAULT_FACADE["distance_matrix"]["value"]["parallelization"],
        n_cpu=DEFAULT_FACADE["distance_matrix"]["value"]["n_cpu"],
    ) -> tuple:
        """
        Computes the distance matrix for the routes in the provided list.

        Parameters:
            routes: a list of SynGraph instances; it is recommended to use the monopartite
                    representation for performance reasons.
            ged_method: a string indicating which algorithm should be used to compute the ged
                        (default: nx_optimized_ged)
            ged_params: a dictionary indicating the optional parameters for chemical similarity and fingerprints
            parallelization: a boolean indicating whether parallelization should be used
            n_cpu: an integer specifying the number of cpus to be used in the parallel calculation

        Returns:
            dist_matrix: a pandas DataFrame (n routes) x (n routes) with the ged values
            meta: a dictionary storing information about the type of graph (mono or bipartite), the algorithm used
                  for the ged calculations and the parameters for chemical similarity and fingerprints
        """

        exceptions = []
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
            return dist_matrix, meta

        except KeyError as ke:
            print("The computation of the distance matrix was not successful")
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

        except SingleRouteClustering as se:
            print(
                "The computation of the distance matrix was not successful: less than 2 routes were found."
            )
            exceptions.append(se)
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
                "type": list[SynGraph],
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
        routes: list,
        clustering_method=DEFAULT_FACADE["clustering"]["value"]["clustering_method"],
        ged_method=DEFAULT_FACADE["clustering"]["value"]["ged_method"],
        ged_params=DEFAULT_FACADE["clustering"]["value"]["ged_params"],
        save_dist_matrix=DEFAULT_FACADE["clustering"]["value"]["save_dist_matrix"],
        compute_metrics=DEFAULT_FACADE["clustering"]["value"]["compute_metrics"],
        parallelization=DEFAULT_FACADE["clustering"]["value"]["parallelization"],
        n_cpu=DEFAULT_FACADE["clustering"]["value"]["n_cpu"],
        **kwargs,
    ):
        """
        Performs clustering of the routes in the provided list.

        Parameters:
            routes: a list of SynGraph instances
            clustering_method: a string indicating which algorithm to use for clustering. If None is given,
                            agglomerative_cluster is used when there are less than 15 routes, otherwise hdbscan is used
            ged_method: a string indicating which algorithm to use for computing the ged (default: nx_optimized_ged)
            ged_params: a dictionary indicating the optional parameters for chemical similarity and fingerprints
            save_dist_matrix: a boolean indicating whether the distance matrix should be saved and returned as output
            compute_metrics: a boolean indicating whether the average metrics for each cluster should be computed
            parallelization: a boolean indicating whether parallelization should be used
            n_cpu: an integer specifying the number of cpus to be used in the parallel calculation
            kwargs: the type of linkage can be indicated when using the agglomerative_cluster; the minimum size of the
                    clusters can be indicated when using hdbscan

        Returns:
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

        exceptions = []
        checked_routes = [r for r in routes if r is not None]
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
            if not compute_metrics:
                return results, meta

            metrics = get_clustered_routes_metrics(routes, results[0])
            return results, metrics, meta

        except SingleRouteClustering as sre:
            print("The clustering was not successful: only one route was found.")
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
            if compute_metrics:
                return None, pd.DataFrame(), meta
            else:
                return None, meta

        except NoClustering as nc:
            print("Clustering was not successful")
            exceptions.append(nc)
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
            if compute_metrics:
                return None, pd.DataFrame(), meta
            else:
                return None, meta

        except KeyError as ke:
            print("The clustering was not successful.")
            exceptions.append(ke)
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

            if compute_metrics:
                return None, pd.DataFrame(), meta
            else:
                return None, meta

    def get_available_options(self) -> dict:
        """
        Returns the available options for clustering calculations.
        """
        return {
            "routes": {
                "name_or_flags": ["-routes"],
                "default": None,
                "required": True,
                "type": list[SynGraph],
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

    def perform_functionality(self, routes: list):
        """
        Checks the presence of subsets in the provided list of SynGraph objects.

        Parameters:
            routes: list of SynGraph instances

        Returns:
            subsets: list of pairs
        """
        subsets = []
        for route1 in routes:
            subsets.extend(
                [route1.source[0], route2.source[0]]
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
                "type": list[SynGraph],
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
        Checks the presence of duplicated routes in the provided list of SynGraph instances.

        Parameters:
                routes: list of SynGraph instances

        Returns:
                a list of duplicates
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
                "type": list[SynGraph],
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
        routes: list,
        out_data_model=DEFAULT_FACADE["merging"]["value"]["out_data_model"],
    ):
        """
        Merges the provided list of SynGraph instances in a new SynGraph instance.

        Parameters:
                routes: list of SynGraph instances
                out_data_model: the data model of the output SynGraph

        Returns:
                a new SynGraph instance resulting from the merging of the input routes
        """
        routes_checked_type = [converter(r, out_data_model) for r in routes]
        return merge_syngraph(routes_checked_type)

    def get_available_options(self) -> dict:
        return {
            "routes": {
                "name_or_flags": ["-routes"],
                "default": None,
                "required": True,
                "type": list[SynGraph],
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

    def perform_functionality(self, routes: list) -> tuple[list, dict]:
        """
        Extract a dictionary in the form {route_id: list[Dict]}

        Parameters:
                routes: list of SynGraph instances

        Returns:

        """
        checked_routes = [r for r in routes if r != {}]

        exceptions = []
        output = []
        try:
            for route in checked_routes:
                reactions = extract_reactions_from_syngraph(route)
                output.append({route.source: reactions})

            invalid_routes = len(routes) - len(checked_routes)

            meta = {
                "nr_routes_in_input_list": len(routes),
                "invalid_routes": invalid_routes,
                "errors": exceptions,
            }

            return output, meta

        except TypeError as ke:
            print("Found route in wrong format: only SynGraph object are accepted.")
            exceptions.append(ke)
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
                "type": list[SynGraph],
                "choices": None,
                "help": "List of SynGraph objects in which duplicates should be searched",
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
    }

    def select_functionality(self, functionality: str, *args, **kwargs):
        """Takes a string indicating a functionality and its arguments and performs the functionality"""
        if functionality not in self.functionalities:
            raise KeyError(
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


def facade(functionality, *args, **kwargs):
    """
    Gives access to the 'facade', which wraps each of the main functionalities of the package.

    Parameters:
        functionality: a string
            It indicates the functionality to be performed
        *args:
            the mandatory arguments of the selected functionality
        **kwargs:
            the optional arguments of the selected functionality

    Returns:
        the output of the selected functionality

    """
    functionality_selector = FacadeFactory()
    return functionality_selector.select_functionality(functionality, *args, **kwargs)


def facade_helper(functionality=None, verbose=False):
    """
    Returns the available facade functions if no input is provided; if the name of a functionality is specified,
    the available parameters options for it are returned.

    Parameters:
        functionality: a string or None (optional, default: None)
            If provided, it indicates the functionality for which the helper is invoked. If it is None,
            the helper for the facade is returned.

        verbose: a boolean (optional, default: False)
            If provided, it indicates whether to print the available options and the default parameters are
            printed on the screen.

    Returns:
        A dictionary with the available options and default parameters.
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
