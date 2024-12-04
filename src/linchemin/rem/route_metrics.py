from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Type, Union, Optional, List, Dict

import linchemin.cheminfo.functions as cif
from linchemin.cgu.convert import converter
from linchemin.cgu.syngraph import (
    BipartiteSynGraph,
    MonopartiteMolSynGraph,
    MonopartiteReacSynGraph,
)
from linchemin.cgu.syngraph_operations import find_all_paths
from linchemin.cheminfo.constructors import MoleculeConstructor
from linchemin.cheminfo.models import ChemicalEquation, Molecule
from linchemin.rem.rem_utils.classification_schema import ClassificationSchema
from linchemin.rem.rem_utils.distance_strategy import distance_function_calculator
from linchemin.rem.step_descriptors import step_descriptor_calculator
from linchemin.utilities import console_logger

"""Module containing functions and classes for computing
route metrics that depend on external information"""
logger = console_logger(__name__)


class BaseMetricError(Exception):
    """Base class for errors related to route metric calculation"""


class NotFullyMappedRouteError(BaseMetricError):
    """Raised if the input route is not fully mapped"""


class MissingDataError(BaseMetricError):
    """Raised if some mandatory data is missing"""


class InvalidComponentTypeError(BaseMetricError):
    """Raised if the selected component type is not among the available types"""


class UnavailableMetricError(BaseMetricError):
    """Raised if the requested metric is not among the available ones"""


class UnavailableMoleculeFormat(BaseMetricError):
    """Raised if the requested molecular format is not among the available ones"""


class RouteComponentType(Enum):
    MOLECULES = "molecules"
    CHEMICAL_EQUATIONS = "chemical_equations"


class RouteComponents:
    molecule_func_map = {
        "smarts": cif.compute_mol_smarts,
        "mol_blockV3K": cif.compute_mol_blockV3k,
        "mol_blockV2K": cif.compute_mol_block,
    }
    structural_format: str
    uid_structure_map: dict

    def __init__(self, component_type: Union[RouteComponentType, str]):
        self._validate_component_type(component_type=component_type)
        self.component_type = component_type

    @staticmethod
    def _validate_component_type(
        component_type: Union[RouteComponentType, str]
    ) -> RouteComponentType:
        if isinstance(component_type, RouteComponentType):
            return component_type
        elif isinstance(component_type, str):
            try:
                return RouteComponentType(component_type)
            except ValueError:
                raise InvalidComponentTypeError(
                    f"Invalid component_type: {component_type}. "
                    f"Must be one of {[e.value for e in RouteComponentType]}"
                )
        else:
            raise InvalidComponentTypeError(
                f"component_type must be RouteComponentType or "
                f"str, not {type(component_type)}"
            )

    @classmethod
    def validate_format(cls, selected_format: str) -> None:
        if selected_format not in cls.molecule_func_map:
            logger.error(f"{selected_format} is not available.")
            raise UnavailableMoleculeFormat

    @classmethod
    def get_mol_to_string_function(cls, fmt: str):
        cls.validate_format(fmt)
        return cls.molecule_func_map[fmt]


@dataclass
class MetricOutput:
    """Dataclass to store the output and metadata of a route metric calculation"""

    metric_value: float = field(default=0.0)
    """ The value of the route metrics"""
    raw_data: dict = field(default_factory=dict)
    """ Raw data of the route metric calculation
    (metric non-normalized value, normalization term"""
    additional_info: dict = field(default_factory=dict)


class RouteMetric(ABC):
    name: str

    def __init__(
        self,
        route: Union[
            BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph
        ],
    ):
        """
        To initialize a route metric

        Parameters:
        -----------
        route: Union[MonopartiteReacSynGraph, MonopartiteMolSynGraph, BipartiteSynGraph]
            The input SynGraph for which the metrics should be computed
        """
        self.route = self._check_route_format(route)

    @staticmethod
    @abstractmethod
    def _check_route_format(
        route: Union[
            BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph
        ],
    ) -> Union[BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph]:
        """To check that the route format is suitable for the metric"""
        pass

    @abstractmethod
    def get_route_components_for_metric(
        self, structural_format: str
    ) -> RouteComponents:
        """To get a map of the uid of the components of a route (molecules or chemical equation)
        and their structural representation. Useful for perform searches to get the data
        necessary for the metric calculation"""  # noqa: E501
        pass

    @abstractmethod
    def compute_metric(
        self,
        data: dict,
        categories: Optional[
            Optional[List[Dict[str, Union[str, Dict[str, float], float]]]]
        ],
    ) -> MetricOutput:
        """
        To compute a route metric dependent on external information

        Parameters:
        -----------
        data: dict
            The external information necessary for computing the route metric
        categories: Optional[dict]
            The categorical or numerical classes used to assign a score to categories of entries

        Returns:
        --------
        MetricOutput
            An object containing the value of the metric, as well as metadata and raw data
        """  # noqa: E501
        pass

    @staticmethod
    def _get_molecules_map(
        molecule_list: List[Molecule], structural_format: str
    ) -> dict:
        if structural_format == "smiles":
            return {mol.uid: mol.smiles for mol in molecule_list}
        rdmol_to_string = RouteComponents.get_mol_to_string_function(structural_format)
        return {mol.uid: rdmol_to_string(mol.rdmol) for mol in molecule_list}


class MetricsFactory:
    """
    Factory to access the route metrics
    Attributes:
    -------------
    _metrics: a dictionary
        It maps the string representing the names of the
        available metrics to the correct RouteMetrics subclass
    """

    _metrics = {}

    @classmethod
    def register_metrics(cls, metric: Type[RouteMetric]):
        """
        Decorator method to register a Facade implementation.
        """
        if hasattr(metric, "name"):
            name = metric.name
            if name not in cls._metrics:
                cls._metrics[name] = metric
        return metric

    @classmethod
    def list_metrics(cls):
        """
        To list the names of all available metrics.

        Returns:
        ---------
        metrics: list
            The names and information of the available metrics.
        """
        return cls._metrics

    @classmethod
    def get_metric_instance(cls, name: str) -> Type[RouteMetric]:
        """
        Get an instance of the specified RouteMetric.

        Parameters:
        ------------
        name: str
            The name of the metric

        Returns:
        ---------
        RouteMetric: An instance of the specified metric.

        Raises:
        --------
        KeyError: If the specified metric is not registered.
        """
        metric = cls._metrics.get(name.lower())
        if metric is None:
            logger.error(f"Metric '{name}' not found")
            raise UnavailableMetricError
        return metric


@MetricsFactory.register_metrics
class ReactionPrecedent(RouteMetric):
    name = "reaction_precedent"
    """
    Subclass to compute a metric based on the precedents of the route steps
    The external data is expected to have the following format:
    {'precedents': {ChemicalEquation.uid: bool}}
    """  # noqa: E501

    def get_route_components_for_metric(
        self, structural_format: str
    ) -> RouteComponents:
        components = RouteComponents(
            component_type=RouteComponentType.CHEMICAL_EQUATIONS
        )
        components.structural_format = structural_format
        chemical_equations = self.route.get_unique_nodes()
        components.uid_structure_map = {
            ce.uid: cif.rdrxn_to_string(ce.rdrxn, out_fmt=structural_format)
            for ce in chemical_equations
        }
        return components

    def compute_metric(
        self,
        data: dict,
        categories=None,
    ) -> MetricOutput:
        output = MetricOutput()
        precedents = data["precedents"]
        n_chemical_equations = len(self.route.graph)
        w_precedents = 0
        for uid, b in precedents.items():
            if b is True:
                w_precedents += 1
        output.metric_value = w_precedents / n_chemical_equations
        return output

    @staticmethod
    def _check_route_format(
        route: Union[
            BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph
        ],
    ) -> MonopartiteReacSynGraph:
        """To ensure that the route is in the expected format"""
        if isinstance(route, MonopartiteReacSynGraph):
            return route
        return converter(route, "monopartite_reactions")


@MetricsFactory.register_metrics
class StartingMaterialsAmount(RouteMetric):
    name = "starting_materials_amount"
    """
    Subclass to compute the amount of starting materials needed to produce a certain amount of target
    The external data is expected to have the following format:
    {'target_amount': n (gr),
     'yield': {ChemicalEquation.uid: yield (0 < y < 1)}}
    """  # noqa: E501

    def get_route_components_for_metric(
        self, structural_format: str
    ) -> RouteComponents:
        components = RouteComponents(component_type=RouteComponentType.MOLECULES)
        components.structural_format = structural_format
        leaves = self.route.get_molecule_leaves()
        components.uid_structure_map = self._get_molecules_map(
            leaves, structural_format
        )
        return components

    def compute_metric(
        self,
        data: dict,
        categories=None,
    ) -> MetricOutput:
        """To compute the amount of starting materials
        needed to produce a certain amount of target"""
        output = MetricOutput()
        root = self.route.get_molecule_roots()[0]
        leaves = self.route.get_molecule_leaves()
        target_amount_mol = self.get_target_amount_in_mol(root, data["target_amount"])
        all_paths = find_all_paths(self.route)
        yields = data["yield"]
        quantities = {"starting_materials": {}, "intermediates": {}}
        for path in all_paths:
            quantities = self.get_path_quantities(
                path, yields, root, leaves, target_amount_mol, quantities
            )

        output.raw_data["quantities"] = quantities
        output.raw_data["target_amount"] = {
            "grams": data["target_amount"],
            "moles": round(target_amount_mol, 3),
        }
        return output

    def get_path_quantities(
        self,
        path: list,
        yields: dict,
        root: Molecule,
        leaves: list,
        target_amount_mol: float,
        quantities: dict,
    ) -> dict:
        """To get the quantities of the reactants in the input path"""
        while path:
            step = path.pop(-1)
            step_yield = yields[step.uid]
            reactants = step.get_reactants()
            product = step.get_products()[0]
            if product == root:
                prod_amount = target_amount_mol
            else:
                prod_amount = quantities["intermediates"][product.smiles]["moles"]
            for reactant in reactants:
                amount_in_moles, amount_in_grams = self.get_reactant_quantity(
                    reactant, step, step_yield, prod_amount
                )
                smiles = reactant.smiles
                if reactant in leaves:
                    quantities["starting_materials"].update(
                        {
                            smiles: {
                                "moles": round(amount_in_moles, 3),
                                "grams": amount_in_grams,
                            }
                        }
                    )

                else:
                    quantities["intermediates"].update(
                        {
                            smiles: {
                                "moles": round(amount_in_moles, 3),
                                "grams": amount_in_grams,
                            }
                        }
                    )
        return quantities

    @staticmethod
    def get_reactant_quantity(
        reactant: Molecule,
        step: ChemicalEquation,
        step_yield: float,
        prod_amount: float,
    ) -> tuple:
        """To compute the needed amount of the input reactant"""
        reactant_mw = cif.compute_molecular_weigth(reactant.rdmol)
        stoich = step.stoichiometry_coefficients["reactants"][reactant.uid]
        amount_in_moles = prod_amount / stoich / step_yield
        return round(amount_in_moles, 3), round(amount_in_moles * reactant_mw, 3)

    @staticmethod
    def _check_route_format(
        route: Union[
            BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph
        ],
    ) -> MonopartiteReacSynGraph:
        """To ensure that the route is in the expected format"""
        if isinstance(route, MonopartiteReacSynGraph):
            return route
        return converter(route, "monopartite_reactions")

    @staticmethod
    def get_target_amount_in_mol(target: Molecule, target_amount_gr: float) -> float:
        """To get the required amount of target in moles"""
        molecular_weight = cif.compute_molecular_weigth(target.rdmol)
        return target_amount_gr / molecular_weight


@MetricsFactory.register_metrics
class StartingMaterialsAvailability(RouteMetric):
    """
    Subclass to compute the starting materials availability metric.
    The external data is expected to have the following format:
    {'starting_material_uid': bool}
    """

    name = "starting_materials_availability"

    def get_route_components_for_metric(
        self, structural_format: str
    ) -> RouteComponents:
        components = RouteComponents(component_type=RouteComponentType.MOLECULES)
        components.structural_format = structural_format
        leaves = self.route.get_leaves()
        components.uid_structure_map = self._get_molecules_map(
            molecule_list=leaves, structural_format=structural_format
        )
        return components

    def compute_metric(
        self,
        data: dict,
        categories=None,
    ) -> MetricOutput:
        """To compute the basic reactant availability metric"""

        starting_materials = self.route.get_leaves()
        output = MetricOutput()
        n_starting_materials = len(starting_materials)
        available = 0.0
        for uid, b in data.items():
            if b is True:
                available += 1
        output.metric_value = available / n_starting_materials
        return output

    @staticmethod
    def _check_route_format(
        route: Union[
            BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph
        ],
    ) -> BipartiteSynGraph:
        """To ensure that the route is in the expected format"""
        if isinstance(route, BipartiteSynGraph):
            return route
        return converter(route, "bipartite")


@MetricsFactory.register_metrics
class ReactantAvailability(RouteMetric):
    """
    Subclass to compute the reactant availability metric.
    The external data is expected to have the following format:
    {'starting material uid': 'string indicating its availability'}
    """

    name = "reactant_availability"

    def get_route_components_for_metric(
        self, structural_format: str
    ) -> RouteComponents:
        components = RouteComponents(component_type=RouteComponentType.MOLECULES)
        components.structural_format = structural_format
        leaves = self.route.get_leaves()
        components.uid_structure_map = self._get_molecules_map(
            molecule_list=leaves, structural_format=structural_format
        )
        return components

    def compute_metric(
        self,
        data: dict,
        categories: List[Dict[str, Union[str, Dict[str, float], float]]],
    ) -> MetricOutput:
        """To compute the reactant availability metric"""
        if categories is None:
            logger.info("No categories provided.")
            raise MissingDataError

        starting_materials = self.route.get_leaves()
        data = self.compute_scores(data, categories)
        output = MetricOutput()
        distance_function = "simple"
        score = 0
        normalization = 0
        root = self.route.get_roots()[0]
        for sm in starting_materials:
            distance_term = distance_function_calculator(
                distance_function, self.route, sm, root
            )
            normalization += distance_term
            score += distance_term * data[sm.uid]
        output.metric_value = round(score / normalization, 2)
        output.raw_data = {
            "not_normalized_metric": score,
            "normalization_term": normalization,
            "distance_function": distance_function,
        }
        return output

    @staticmethod
    def _check_route_format(
        route: Union[
            BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph
        ],
    ) -> BipartiteSynGraph:
        """To ensure that the route is in the expected format"""
        if isinstance(route, BipartiteSynGraph):
            return route
        return converter(route, "bipartite")

    @staticmethod
    def compute_scores(
        external_info: dict,
        categories: List[Dict[str, Union[str, Dict[str, float], float]]],
    ) -> dict:
        """To compute the score for each entry based on the provided categories"""
        schema = ClassificationSchema(categories)
        new_data = {}
        for uid, provider in external_info.items():
            provider_score = schema.compute_score(provider)
            new_data[uid] = provider_score
        return new_data


@MetricsFactory.register_metrics
class YieldMetric(RouteMetric):
    """
    Subclass to compute the yield metric.
    The external data is expected to have the following format:
    {'reaction_uid': yield (float between 0 and 1)}
    """

    name = "yield"

    def get_route_components_for_metric(
        self, structural_format: str
    ) -> RouteComponents:
        components = RouteComponents(
            component_type=RouteComponentType.CHEMICAL_EQUATIONS
        )
        components.structural_format = structural_format
        chemical_equations = self.route.get_unique_nodes()
        components.uid_structure_map = {
            ce.uid: cif.rdrxn_to_string(ce.rdrxn, out_fmt=structural_format)
            for ce in chemical_equations
        }
        return components

    def compute_metric(
        self,
        external_data: dict,
        categories=None,
    ) -> MetricOutput:
        """To compute the yield metric"""
        output = MetricOutput()
        distance_function = "longest_sequence"
        score = 0.0
        normalization = 0.0
        reaction_root = self.route.get_roots()[0]
        for step in self.route.get_unique_nodes():
            distance_term = distance_function_calculator(
                distance_function, self.route, step, reaction_root
            )
            normalization += distance_term
            score += distance_term * external_data[step.uid]
        output.metric_value = round(score / normalization, 2)
        output.raw_data = {
            "not_normalized_metric": score,
            "normalization_term": normalization,
            "distance_function": distance_function,
        }
        return output

    @staticmethod
    def _check_route_format(
        route: Union[
            BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph
        ],
    ) -> MonopartiteReacSynGraph:
        """To ensure that the route is in the expected format"""
        if isinstance(route, MonopartiteReacSynGraph):
            return route
        return converter(route, "monopartite_reactions")


@MetricsFactory.register_metrics
class RenewableCarbonMetric(RouteMetric):
    """
    Subclass to compute the renewable carbon metric: % of atoms
    in the target molecule coming from starting materials in
    the specified list.
    The external data is expected to have the following format:
    {molecule_uid: bool} The bool indicates whether the molecule is a starting material with renewable carbon
    """  # noqa: E501

    name = "renewable_carbon"

    def get_route_components_for_metric(
        self, structural_format: str
    ) -> RouteComponents:
        components = RouteComponents(component_type=RouteComponentType.MOLECULES)
        components.structural_format = structural_format
        mpm_route = converter(self.route, "monopartite_molecules")
        all_molecules = mpm_route.get_unique_nodes()
        components.uid_structure_map = self._get_molecules_map(
            molecule_list=list(all_molecules), structural_format=structural_format
        )
        return components

    def compute_metric(
        self,
        external_data: dict,
        categories=None,
    ) -> MetricOutput:
        if not self.is_mapped_route(self.route):
            self._handle_unmapped_route_error()

        if all(value is False for value in external_data.values()):
            self._handle_no_building_blocks_found()

        target_c_atoms = self._get_target_carbon_atoms()
        starting_materials = self._get_starting_materials(external_data)

        contributing_atoms = self._calculate_contributing_atoms(
            starting_materials, target_c_atoms
        )
        output = self._calculate_metric_output(contributing_atoms, target_c_atoms)

        return output

    @staticmethod
    def _handle_unmapped_route_error():
        """To raise an error if the provided route has unmapped chemical equations"""
        logger.error(
            "Unmapped reactions were found."
            "A fully mapped route is needed to compute this metrics"
        )
        raise NotFullyMappedRouteError

    @staticmethod
    def _get_building_block_molecules(external_data: dict) -> set:
        """To build Molecule objects corresponding to the provided building blocks"""
        mol_constructor = MoleculeConstructor()
        return {
            mol_constructor.build_from_molecule_string(s, "smiles")
            for s in external_data["building_blocks"]
        }

    def _get_target_carbon_atoms(self) -> list:
        """To get the list of atom ids corresponding
        to the Carbon atoms in the target"""
        target = self.route.get_molecule_roots()[0]
        return [
            atom.GetIdx()
            for atom in target.rdmol_mapped.GetAtoms()
            if atom.GetSymbol() == "C"
        ]

    def _get_starting_materials(self, external_data: dict) -> set:
        """To get the set of starting materials
        which are building blocks with renewable carbon"""
        leaves = [m.uid for m in self.route.get_molecule_leaves()]
        return {
            uid
            for uid, is_bb in external_data.items()
            if is_bb is True and uid in leaves
        }

    def _calculate_contributing_atoms(
        self, starting_materials: set, target_c_atoms: list
    ) -> dict:
        """To get the atoms from the starting materials which appear in the target"""

        steps_w_bb = [
            ce
            for ce in self.route.get_unique_nodes()
            if any(r.uid in starting_materials for r in ce.get_reactants())
        ]
        contributing_atoms = {}
        for step in steps_w_bb:
            out = step_descriptor_calculator(
                "step_effectiveness", self.route, step.smiles
            )
            sm_uids = {
                sm_uid
                for sm_uid in starting_materials
                if sm_uid in out.additional_info["contributing_atoms"].keys()
            }
            for uid in sm_uids:
                contributing_atoms[uid] = []
                atom_info_list = out.additional_info["contributing_atoms"][uid]
                for atom_info in atom_info_list:
                    if atom_info["target_atom"] in target_c_atoms:
                        contributing_atoms[uid].append(atom_info)
        return contributing_atoms

    @staticmethod
    def _calculate_metric_output(
        contributing_atoms: dict, target_c_atoms: list
    ) -> MetricOutput:
        """To populate the output object"""
        output = MetricOutput()
        n_target_c_atoms = len(target_c_atoms)
        n_target_c_atoms_from_bb = sum(
            len(atom_info_list) for atom_info_list in contributing_atoms.values()
        )
        output.metric_value = round(n_target_c_atoms_from_bb / n_target_c_atoms, 2)
        output.additional_info["contributing_atoms"] = contributing_atoms
        return output

    @staticmethod
    def _handle_no_building_blocks_found() -> MetricOutput:
        """To populate the output object if none of the
        specified building block appears in the route"""
        msg = "None of the starting material is a renewable carbon building block"
        logger.warning(msg)
        output = MetricOutput()
        output.metric_value = 0.0
        output.raw_data = msg
        return output

    @staticmethod
    def _check_route_format(
        route: Union[
            BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph
        ],
    ) -> MonopartiteReacSynGraph:
        """To ensure that the route is in the expected format"""
        if isinstance(route, MonopartiteReacSynGraph):
            return route
        return converter(route, "monopartite_reactions")

    @staticmethod
    def is_mapped_route(route: MonopartiteReacSynGraph) -> bool:
        return all(ce.disconnection is not None for ce in route.get_unique_nodes())


def route_metric_calculator(
    metric_name: str,
    route: Union[BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph],
    external_data: dict,
    categories: Optional[List[Dict[str, Union[str, Dict[str, float], float]]]] = None,
) -> MetricOutput:
    """
    To compute a route metric

    Parameters:
    -----------
    metric_name: str
        The name of the metric to be computed
    route: Union[BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph]
        The route for which the metric should be computed
    external_data: dict
        The external data needed for computing the route metric
    categories: Optional[List[Dict[str, Union[str, Dict[str, float], float]]]] = None
        Categorical or numerical criterion to assign scores to entries

    Returns:
    ----------
    output: MetricOutput
        Its attributes contain (i) the metric value (ii) a dictionary containing the raw data
        (not-normalized metric,normalization term)

    """  # noqa: E501
    if isinstance(
        route, (BipartiteSynGraph, MonopartiteMolSynGraph, MonopartiteReacSynGraph)
    ):
        metric = MetricsFactory.get_metric_instance(metric_name)
        metric_instance = metric(route=route)
        return metric_instance.compute_metric(external_data, categories)
    logger.error(
        f"The input route must be a SynGraph object. "
        f"{type(route)} cannot be processed."
    )
    raise TypeError
