import inspect
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pandas as pd

from linchemin import settings
from linchemin.cgu.graph_transformations.exceptions import TranslationError
from linchemin.cgu.route_sanity_check import (
    get_available_route_sanity_checks,
    route_checker,
)
from linchemin.cgu.syngraph import MonopartiteReacSynGraph
from linchemin.cgu.syngraph_operations import (
    GraphTypeError,
    extract_reactions_from_syngraph,
)
from linchemin.cgu.translate import (
    get_available_data_models,
    get_input_formats,
    get_output_formats,
    translator,
)
from linchemin.cheminfo.atom_mapping import (
    MappingOutput,
    get_available_mappers,
    perform_atom_mapping,
    pipeline_atom_mapping,
)
from linchemin.interfaces.utils_interfaces import get_ged_dict, get_parallelization_dict
from linchemin.rem.clustering import (
    ClusteringError,
    clusterer,
    get_available_clustering,
    get_clustered_routes_metrics,
)
from linchemin.rem.graph_distance import GraphDistanceError, compute_distance_matrix
from linchemin.rem.route_descriptors import (
    DescriptorError,
    descriptor_calculator,
    get_available_descriptors,
    get_configuration,
)
from linchemin.utilities import console_logger

"""
Module containing high level functionalities/"user stories" to work in stream;
it provides a simplified interface for the user.
"""

logger = console_logger(__name__)


class FacadeError(Exception):
    """Base class for exceptions leading to unsuccessful execution of facade functionalities"""

    pass


class UnavailableFunctionality(FacadeError):
    """Raised if the selected functionality is not among the available ones"""

    pass


class MissingParameterError(FacadeError):
    """Raised if a mandatory parameter for the functionality is missing"""

    pass


class Facade(ABC):
    """Definition of the abstract class for high level functionalities' facade."""

    name: str
    info: str

    @abstractmethod
    def perform_functionality(self, routes: list) -> Tuple[Any, Dict]:
        pass

    @classmethod
    def get_available_options(cls) -> dict:
        return {
            "routes": {
                "name_or_flags": ["-routes"],
                "default": None,
                "required": True,
                "type": list,
                "choices": None,
                "dest": "routes",
            }
        }

    @classmethod
    def print_available_options(cls) -> dict:
        data = cls.get_available_options()
        for d, info in data.items():
            print("argument:", d)
            print("     info: ", info["help"])
            print("     default: ", info["default"])
            print("     available options: ", info["choices"])
        return data


class FacadeFactory:
    """Factory class for facade functionalities"""

    _functionalities = {}

    @classmethod
    def register_facade(cls, facade_class: Type[Facade]):
        """
        Decorator method to register a Facade implementation.
        """
        if hasattr(facade_class, "name") and hasattr(facade_class, "info"):
            name = facade_class.name
            info = facade_class.info
            if name not in cls._functionalities:
                cls._functionalities[name] = {"value": facade_class, "info": info}
        return facade_class

    @classmethod
    def select_functionality(cls, facade_name: str):
        """Takes a string indicating a functionality and its arguments and performs the functionality"""
        if facade_name not in cls._functionalities:
            logger.error(
                f"'{facade_name}' is not a valid functionality."
                f"Available functionalities are: {list(cls._functionalities.keys())}"
            )
            raise UnavailableFunctionality

        return cls._functionalities[facade_name]["value"]

    @classmethod
    def get_helper(cls, facade_name: str) -> dict:
        """Takes a string indicating a functionality and returns the available options"""
        facade_class = cls.select_functionality(facade_name)
        return facade_class.get_available_options()

    @classmethod
    def get_helper_verbose(cls, facade_name: str) -> dict:
        """Takes a string indicating a functionality and prints the available options"""
        facade_class = cls.select_functionality(facade_name)
        facade_class.print_available_options()
        return facade_class.get_available_options()

    @classmethod
    def list_functionalities(cls) -> list:
        """To list the registered Facade names."""
        return list(cls._functionalities.keys())

    @classmethod
    def list_functionalities_w_info(cls) -> dict:
        """To list the registered Facade names."""
        return {
            f: additional_info["info"]
            for f, additional_info in cls._functionalities.items()
        }


@FacadeFactory.register_facade
class TranslateFacade(Facade):
    """Subclass of Facade for translating list of routes between formats."""

    name = "translate"
    info = "To translate a list of routes from a format to another one."

    def __init__(
        self,
        input_format: str,
        output_format: str = settings.FACADE.data_format,
        out_data_model: str = settings.FACADE.out_data_model,
        parallelization: bool = settings.FACADE.parallelization,
        n_cpu: int = settings.FACADE.n_cpu,
    ):
        """
        Parameters:
        ------------
        input_format: str
            The format of the input file
        output_format: str
            The desired output format (format as for the translator factory) (default syngraph)
        out_data_model: str
            The desired output data model (default bipartite)
        parallelization: bool
            Whether parallelization should be used (default False)
        n_cpu: int
            The number of cpus to be used in the parallel calculation (default 4)
        """
        self.input_format = input_format
        self.out_format = output_format
        self.out_data_model = out_data_model
        self.parallelization = parallelization
        self.n_cpu = n_cpu

    def perform_functionality(self, routes: List) -> Tuple[Any, Dict]:
        """
        Takes a list of routes in the specified input format and converts it into the desired format (default: SynGraph).
        Returns the converted routes and some metadata.

        Parameters:
        ------------
        routes: List
            The list of routes to be translated

        Returns:
        ---------
        tuple
            output: The list of translated routes

            meta: a dictionary storing information about the original file and the CASP tool that produced the routes
        """
        exceptions: List = []
        out_routes: List = []

        try:
            if self.parallelization:
                converted_routes = self._run_parallel_functionality(routes=routes)

            else:
                converted_routes = self._run_serial_functionality(routes=routes)

            out_routes.extend([r for r in converted_routes if r is not None])

        except TranslationError as te:
            exceptions.append(te)

        except Exception as e:
            exceptions.append(e)

        invalid_routes = len(routes) - len(out_routes)

        meta = {
            "nr_routes_in_input_list": len(routes),
            "input_format": routes,
            "nr_output_routes": len(out_routes),
            "invalid_routes": invalid_routes,
        }

        return out_routes, meta

    def _run_parallel_functionality(self, routes: List) -> List:
        """To run the translation using parallelization"""
        pool = mp.Pool(self.n_cpu)
        converted_routes = pool.starmap(
            translator,
            [
                (self.input_format, route, self.out_format, self.out_data_model)
                for route in routes
            ],
        )
        return converted_routes

    def _run_serial_functionality(self, routes: List) -> List:
        """To run the translation serially"""
        return [
            translator(self.input_format, route, self.out_format, self.out_data_model)
            for route in routes
        ]

    @classmethod
    def get_available_options(cls) -> dict:
        """
        Returns the available options for output formats as a dictionary.
        """
        options = super().get_available_options()
        options["routes"]["help"] = "A list of routes to be translated"
        options.update(get_parallelization_dict())
        options.update(
            {
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
            }
        )
        return options

    @classmethod
    def print_available_options(cls):
        """
        Prints the available options for output formats.
        """
        print('"Translate" options and default:')
        return super().print_available_options()


@FacadeFactory.register_facade
class RoutesDescriptorsFacade(Facade):
    """Subclass of Facade for computing routes metrics."""

    name = "routes_descriptors"
    info = "To compute metrics of a list of SynGraph objects"

    def __init__(self, descriptors: Optional[List[str]] = settings.FACADE.descriptors):
        """
        Parameters:
        ------------
        descriptors: Optional[Union[List, None]]
            The list of strings indicating the desired descriptors to be computed
            (default None -> all the available descriptors)
        """
        available_descriptors = get_available_descriptors()
        if descriptors is None:
            self.descriptors = available_descriptors
        else:
            self.descriptors = descriptors

    def perform_functionality(self, routes: List) -> Tuple[Any, Dict]:
        """
        Computes the desired descriptors (default: all the available descriptors) for the routes in the provided list.

        Parameters:
        ------------
        routes: List[Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]]
            The list of SynGraph instances

        Returns:
        --------
        tuple
            output: a pandas dataframe

            meta: a dictionary storing information about the computed descriptors
        """
        output = pd.DataFrame()

        exceptions = []
        checked_routes = [r for r in routes if r is not None]
        invalid_routes = len(routes) - len(checked_routes)
        output["route_id"] = [route.uid for route in checked_routes]
        configuration: List = []

        for d in self.descriptors:
            try:
                output[d] = [
                    descriptor_calculator(route, d) for route in checked_routes
                ]
                configuration.append(get_configuration(d))
            except DescriptorError as ke:
                exceptions.append(ke)

            except Exception as e:
                exceptions.append(e)

        output.attrs["configuration"] = configuration
        meta = {
            "descriptors": self.descriptors,
            "invalid_routes": invalid_routes,
            "errors": exceptions,
        }
        return output, meta

    @classmethod
    def get_available_options(cls) -> dict:
        """
        Returns the available options for route descriptors as a dictionary.
        """
        options = super().get_available_options()
        options["routes"][
            "help"
        ] = "List of SynGraph objects for which the selected descriptors should be calculated"
        options.update(
            {
                "descriptors": {
                    "name_or_flags": ["-descriptors"],
                    "default": settings.FACADE["routes_descriptors"]["value"],
                    "required": False,
                    "type": List[str],
                    "choices": get_available_descriptors(),
                    "help": "List of descriptors to be calculated",
                    "dest": "descriptors",
                },
            }
        )
        return options

    @classmethod
    def print_available_options(cls):
        """
        Prints the available options for routes descriptors.
        """
        print("Routes descriptors options and default:")
        return super().print_available_options()


@FacadeFactory.register_facade
class GedFacade(Facade):
    """Subclass of Facade for computing the distance matrix of a list of routes with GED algorithms."""

    name = "distance_matrix"
    info = "To compute the distance matrix of a list of SynGraph objects via a Graph Edit Distance algorithm"

    def __init__(
        self,
        ged_method: str = settings.FACADE["ged_method"],
        ged_params: Union[dict, None] = settings.FACADE["ged_params"],
        parallelization: bool = settings.FACADE["parallelization"],
        n_cpu: int = settings.FACADE["n_cpu"],
    ):
        """
        Parameters:
        ------------
        ged_method: Optional[str]
            The GED algorithm to be used (default 'nx_optimized_ged')
        ged_params: Optional[Union[dict, None]]
            The optional parameters for chemical similarity and fingerprints
            (default None ->the default parameters are used)
        parallelization: Optional[bool]
            Whether parallelization should be used (default False)
        n_cpu: Optional[int]
            The number of cpus to be used in the parallel calculation (default 8)

        """
        self.ged_method = ged_method
        self.ged_params = ged_params
        self.parallelization = parallelization
        self.n_cpu = n_cpu

    def perform_functionality(
        self,
        routes: List,
    ) -> tuple:
        """
        Computes the distance matrix for the routes in the provided list.

        Parameters:
        ------------
        routes: List[Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]]
            The list of SynGraph instances for which should be computed; it is recommended to use the monopartite
            representation for performance reasons

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
                ged_method=self.ged_method,
                ged_params=self.ged_params,
                parallelization=self.parallelization,
                n_cpu=self.n_cpu,
            )
            meta = {
                "ged_algorithm": self.ged_method,
                "ged_params": self.ged_params,
                "graph_type": "monopartite"
                if isinstance(routes[0], MonopartiteReacSynGraph)
                else "bipartite",
                "invalid_routes": len(routes) - len(checked_routes),
                "errors": exceptions,
            }

        except GraphDistanceError as ke:
            exceptions.append(ke)
            meta = {
                "ged_algorithm": self.ged_method,
                "ged_params": self.ged_params,
                "graph_type": "monopartite"
                if isinstance(routes[0], MonopartiteReacSynGraph)
                else "bipartite",
                "invalid_routes": len(routes) - len(checked_routes),
                "errors": exceptions,
            }
            dist_matrix = pd.DataFrame()
        return dist_matrix, meta

    @classmethod
    def get_available_options(cls) -> dict:
        """
        Returns the available options for GED calculations.
        """
        options = super().get_available_options()
        options["routes"][
            "help"
        ] = "List of SynGraph objects for which the distance matrix should be calculated"
        options.update(get_parallelization_dict())
        options.update(get_ged_dict())
        return options

    @classmethod
    def print_available_options(cls):
        """
        Prints the available options for GED calculations as a dictionary.
        """
        print("GED options and default:")
        return super().print_available_options()


@FacadeFactory.register_facade
class ClusteringFacade(Facade):
    """Facade Factory to give access to the functionalities"""

    name = "clustering"
    info = "To cluster a list of SynGraph objects"

    def __init__(
        self,
        clustering_method: Union[str, None] = settings.FACADE["clustering_method"],
        ged_method: str = settings.FACADE["ged_method"],
        ged_params: Union[dict, None] = settings.FACADE["ged_params"],
        save_dist_matrix: bool = settings.FACADE["save_dist_matrix"],
        compute_metrics: bool = settings.FACADE["compute_metrics"],
        parallelization: bool = settings.FACADE["parallelization"],
        n_cpu: int = settings.FACADE["n_cpu"],
        **kwargs,
    ):
        """
        Parameters:
        -----------
        clustering_method: Optional[Union[str, None]]
            The clustering algorithm to be used. If None is given,
            agglomerative_cluster is used when there are less than 15 routes, otherwise hdbscan is used (default None)
        ged_method: Optional[str]
            The GED algorithm to be used (default 'nx_optimized_ged')
        ged_params: Union[dict, None]
            The optional parameters for chemical similarity and fingerprints
            (default None ->the default parameters are used)
        save_dist_matrix: Optional[bool]
            Whether the distance matrix should be saved and returned as output
        compute_metrics: Optional[bool]
            Whether the average metrics for each cluster should be computed
        parallelization: Optional[bool]
            Whether parallelization should be used (default False)
        n_cpu: Optional[int]
            The number of cpus to be used in the parallel calculation (default 8)
        kwargs: the type of linkage can be indicated when using the agglomerative_cluster; the minimum size of the
                clusters can be indicated when using hdbscan

        """
        self.clustering_method = clustering_method
        self.ged_method = ged_method
        self.ged_params = ged_params
        self.save_dist_matrix = save_dist_matrix
        self.compute_metrics = compute_metrics
        self.additional_args = kwargs
        self.parallelization = parallelization
        self.n_cpu = n_cpu

    def perform_functionality(self, routes: List) -> Tuple[Any, Dict]:
        """
        Performs clustering of the routes in the provided list.

        Parameters:
        -----------
        routes: List
            The input list of SynGraph

        Returns:
        ----------
        tuple
            results: a tuple with: clustering, score, (dist_matrix), corresponding to the output of the clustering,
                                 its silhouette score (and the distance matrix as a pandas dataframe if save_dist_matrix=True)

            meta: a dictionary storing information about the original file and the CASP tool that produced the routes,
                the type of graph (mono or bipartite), information regarding the clustering, information regarding
                the ged calculations and the parameters for chemical similarity and fingerprints
        """

        if self.clustering_method is None:
            self.clustering_method = (
                "hdbscan" if len(routes) > 15 else "agglomerative_cluster"
            )

        exceptions: List = []
        checked_routes = [r for r in routes if r is not None]
        metrics = pd.DataFrame()
        try:
            results = clusterer(
                checked_routes,
                ged_method=self.ged_method,
                clustering_method=self.clustering_method,
                ged_params=self.ged_params,
                save_dist_matrix=self.save_dist_matrix,
                parallelization=self.parallelization,
                n_cpu=self.n_cpu,
                **self.additional_args,
            )
            meta = {
                "graph_type": "monopartite"
                if isinstance(routes[0], MonopartiteReacSynGraph)
                else "bipartite",
                "clustering_algorithm": self.clustering_method,
                "clustering_params": self.additional_args,
                "ged_algorithm": self.ged_method,
                "ged_parameters": self.ged_params,
                "invalid_routes": len(routes) - len(checked_routes),
                "errors": exceptions,
            }

            if self.compute_metrics:
                metrics = get_clustered_routes_metrics(routes, results[0])

        except ClusteringError as sre:
            exceptions.append(sre)
            meta = {
                "graph_type": "monopartite"
                if isinstance(checked_routes[0], MonopartiteReacSynGraph)
                else "bipartite",
                "clustering_algorithm": self.clustering_method,
                "clustering_params": self.additional_args,
                "ged_algorithm": self.ged_method,
                "ged_parameters": self.ged_params,
                "invalid_routes": len(routes) - len(checked_routes),
                "errors": exceptions,
            }
            results = None

        return ((results, metrics), meta) if self.compute_metrics else (results, meta)

    @classmethod
    def get_available_options(cls) -> dict:
        """
        Returns the available options for clustering calculations.
        """
        options = super().get_available_options()
        options["routes"]["help"] = "List of SynGraph objects to be clustered"
        options.update(get_parallelization_dict())
        options.update(get_ged_dict())
        return {
            "clustering_method": {
                "name_or_flags": ["-clustering_method"],
                "default": settings.FACADE["clustering_method"],
                "required": False,
                "type": str,
                "choices": get_available_clustering(),
                "help": "Method to be used to calculate the GED",
                "dest": "ged_method",
            },
            "save_dist_matrix": {
                "name_or_flags": ["-save_dist_matrix"],
                "default": settings.FACADE["save_dist_matrix"],
                "required": False,
                "type": bool,
                "choices": [True, False],
                "help": "Whether the distance matrix should be saved and returned as output",
                "dest": "save_dist_matrix",
            },
            "compute_metrics": {
                "name_or_flags": ["-compute_metrics"],
                "default": settings.FACADE["compute_metrics"],
                "required": False,
                "type": bool,
                "choices": [True, False],
                "help": "Whether metrics aggregated by clusters should be computed",
                "dest": "compute_metrics",
            },
        }

    @classmethod
    def print_available_options(cls):
        """
        Prints the available options for clustering calculations as a dictionary.
        """
        print("Clustering options and default:")
        return super().print_available_options()


@FacadeFactory.register_facade
class ReactionExtractionFacade(Facade):
    """Subclass of Facade for extracting unique reaction strings from a list of routes."""

    name = "extract_reactions_strings"
    info = (
        "To extract a list of unique reaction strings from a list of SynGraph objects"
    )

    def perform_functionality(
        self,
        routes: List,
    ) -> Tuple[List, Dict]:
        """
        To extract a list of reaction strings

        Parameters:
        ---------------
        routes: List
            The list of SynGraph instances

        Returns:
        ---------
        Tuple[List, Dict]:
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

        except GraphTypeError as ke:
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

    @classmethod
    def get_available_options(cls) -> dict:
        return super().get_available_options()

    @classmethod
    def print_available_options(cls):
        print("Reaction strings extraction options and default:")
        return super().print_available_options()


@FacadeFactory.register_facade
class AtomMappingFacade(Facade):
    """Subclass of Facade for mapping the chemical equations in a list of routes."""

    name = "atom_mapping"
    info = "To get a list of mapped SynGraph objects"

    def __init__(
        self,
        mapper: Union[str, None] = settings.FACADE["mapper"],
    ):
        self.mapper = mapper

    def perform_functionality(
        self,
        routes: List,
    ) -> Tuple[List, Dict]:
        """
        To generate a list of SynGraph objects with mapped chemical equations

        Parameters:
        ------------
        routes: List
            The list of SynGraph instances

        Returns:
        ---------
        Tuple[List, dict]:
            output: The list of mapped SynGraph objects of the same type as the input objects
            meta: The dictionary storing information about the run
        """
        out_syngraphs: List = []
        syngraph_type = type(routes[0])
        tot_success_rate: Union[float, int] = 0
        exceptions = []
        for route in routes:
            name = route.name
            route_id = route.uid
            reaction_list = extract_reactions_from_syngraph(route)
            mapping_out = self._map_reaction_strings(reaction_list)
            try:
                mapped_route = syngraph_type(mapping_out.mapped_reactions)
                mapped_route.name = name
                out_syngraphs.append(mapped_route)
            except Exception as e:
                exceptions.append({"route_uid": route_id, "exception": e})

            tot_success_rate += mapping_out.success_rate

        tot_success_rate = float(tot_success_rate / len(routes))

        meta = {
            "mapper": self.mapper,
            "mapping_success_rate": tot_success_rate,
            "exception": exceptions,
            "nr_invalid_routes": len(routes) - len(out_syngraphs),
        }

        return out_syngraphs, meta

    def _map_reaction_strings(self, reaction_list: List) -> MappingOutput:
        """To perform the mapping of the reaction strings"""
        if self.mapper is None:
            # the full pipeline is used
            mapping_out = pipeline_atom_mapping(reaction_list)
        else:
            # the selected mapper is used
            mapping_out = perform_atom_mapping(reaction_list, self.mapper)
        if mapping_out.success_rate != 1:
            # if not all the reactions are mapped, a warning is raised and
            # the output graph is built using all the mapped and the unmapped reactions (so that it is complete)
            mapping_out.mapped_reactions.extend(mapping_out.unmapped_reactions)
        return mapping_out

    @classmethod
    def get_available_options(cls) -> dict:
        options = super().get_available_options()
        options["routes"]["help"] = ("List of SynGraph to be mapped",)
        options.update(
            {
                "mapper": {
                    "name_or_flags": ["-mapper"],
                    "default": settings.FACADE["mapper"],
                    "required": False,
                    "type": str,
                    "choices": get_available_mappers(),
                    "help": "Which mapper should be used",
                    "dest": "mapper",
                }
            }
        )
        return options

    @classmethod
    def print_available_options(cls):
        print("Atom mapping of routes reactions options and default:")
        return super().print_available_options()


@FacadeFactory.register_facade
class RouteSanityCheckFacade(Facade):
    """Subclass of Facade for performing sanity checks on a list of routes."""

    name = "routes_sanity_checks"
    info = "To perform sanity checks on a list of routes"

    def __init__(
        self,
        checks: Union[List[str], None] = settings.FACADE["checks"],
    ):
        """
        Parameters:
        ------------
        check: Optional[Union[List[str], None]]
            The list of sanity checks to be performed; if it is not provided, all the available
            sanity check are applied (default None)
        """
        self.checks = checks

    def perform_functionality(
        self,
        routes: List,
    ) -> Tuple[List, Dict]:
        """
        Returns a list of routes in which possible issues are removed

        Parameters:
        ------------
        routes: List
            The list of SynGraph instances

        Returns:
        ---------
        Tuple[List, dict]:
            output: The list of checked SynGraph objects

            meta: The dictionary storing information about the run
        """
        if self.checks is None:
            self.checks = get_available_route_sanity_checks()
        checked_routes: List = []
        exceptions = []
        valid_routes = [r for r in routes if r is not None]
        invalid_routes = len(routes) - len(valid_routes)

        for r in valid_routes:
            checked_route = r
            for check in self.checks:
                try:
                    checked_route = route_checker(checked_route, check, fix_issue=True)
                except Exception as ke:
                    exceptions.append(ke)
            checked_routes.append(checked_route)

        meta = {
            "checks": self.checks,
            "invalid_routes": invalid_routes,
            "errors": exceptions,
        }
        return checked_routes, meta

    @classmethod
    def get_available_options(cls):
        """
        Returns the available options for route sanity checks as a dictionary.
        """
        options = super().get_available_options()
        options["routes"][
            "help"
        ] = "List of SynGraphs for which the selected sanity checks should be performed"
        options.update(
            {
                "checks": {
                    "name_or_flags": ["-checks"],
                    "default": settings.FACADE["checks"],
                    "required": False,
                    "type": List[str],
                    "choices": get_available_route_sanity_checks(),
                    "help": "List of sanity checks to be performed",
                    "dest": "checks",
                },
            }
        )
        return options

    @classmethod
    def print_available_options(cls):
        """
        Prints the available options for routes sanity checks.
        """
        print("Routes sanity checks options and default:")
        return super().print_available_options()


def facade(functionality: str, routes: List, **kwargs) -> Tuple[Any, Dict]:
    """
    To perform one of the main functionality of the package.

    Parameters:
    -------------
    functionality: str
        The name of the functionality to be performed
    routes:
        The list of routes on which the functionality should be performed
    **kwargs:
        the arguments of the selected functionality

    Returns:
    --------
    the output of the selected functionality

    """
    facade_class = FacadeFactory.select_functionality(functionality)
    facade_object = facade_class(**kwargs)
    return facade_object.perform_functionality(routes)


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
        av_functionalities = FacadeFactory.list_functionalities_w_info()
        if verbose:
            print("Available functionalities are:")
            for f, info in av_functionalities.items():
                print(" ", f, ":", info)
        return av_functionalities

    helper_selector = FacadeFactory()
    if verbose:
        return helper_selector.get_helper_verbose(functionality)
    return helper_selector.get_helper(functionality)
