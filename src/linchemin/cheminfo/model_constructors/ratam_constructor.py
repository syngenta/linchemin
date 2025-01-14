from collections import Counter, defaultdict
from typing import Dict, List, Set, Union

from linchemin import utilities
from linchemin.cheminfo.models import (
    AtomTransformation,
    Molecule,
    Ratam,
    ReactionComponents,
)

logger = utilities.console_logger(__name__)


class BadMapping(Exception):
    pass


class RatamBuilder:
    """Class implementing the constructor of the Ratam class"""

    def __init__(
        self, reaction_components: ReactionComponents, desired_product: Molecule
    ):
        self.reaction_components = reaction_components
        self.desired_product = desired_product
        self.ratam = Ratam()

    def build(self) -> Ratam:
        """To build the Ratam object"""
        self._create_initial_map_info()
        self._reassign_roles()
        self._create_atom_transformations()
        self._analyze_unmapped_atoms()
        return self.ratam

    def _create_initial_map_info(self):
        """To create the initial role information"""
        self.ratam.full_map_info = {
            "reactants": {},
            "reagents": {},
            "products": self._get_molecule_maps(self.reaction_components.products),
        }

        # Create a list of dictionaries for precursors
        self.precursors = []
        for mol in self.reaction_components.reactants:
            self.precursors.append(self._get_molecule_maps([mol]))

        for mol in self.reaction_components.reagents:
            self.precursors.append(self._get_molecule_maps([mol]))

    @staticmethod
    def _get_molecule_maps(
        molecules: List[Molecule],
    ) -> Dict[str, List[Dict[int, int]]]:
        """To get the map of atom id and map number of a given molecule"""
        molecule_maps = defaultdict(list)
        for mol in molecules:
            mapping = {
                a.GetIdx(): a.GetAtomMapNum() for a in mol.rdmol_mapped.GetAtoms()
            }
            molecule_maps[mol.uid].append(mapping)
        return dict(molecule_maps)

    def _reassign_roles(self):
        """To re-assign roles to the precursors based on the mapping information"""
        desired_product_map_nums = set()
        for maps in self.ratam.full_map_info["products"][self.desired_product.uid]:
            desired_product_map_nums.update(maps.values())
        desired_product_map_nums -= {0, -1}
        for precursor in self.precursors:
            for uid, maps in precursor.items():
                if any(
                    any(map_num in desired_product_map_nums for map_num in m.values())
                    for m in maps
                ):
                    if uid in self.ratam.full_map_info["reactants"]:
                        self.ratam.full_map_info["reactants"][uid].extend(maps)
                    else:
                        self.ratam.full_map_info["reactants"][uid] = maps
                else:
                    if uid in self.ratam.full_map_info["reagents"]:
                        self.ratam.full_map_info["reagents"][uid].extend(maps)
                    else:
                        self.ratam.full_map_info["reagents"][uid] = maps
        self._mapping_sanity_check()

    def _mapping_sanity_check(self) -> None:
        """
        Perform a sanity check on the atom mapping.

        This method checks if any mapping number (except 0 and -1) is used more than once,
        which would indicate an invalid mapping.

        Raises:
            BadMapping: If an invalid mapping is detected.
        """
        map_nums = self._collect_map_numbers()
        duplicate_maps = self._find_duplicate_maps(map_nums)

        if duplicate_maps:
            duplicate_str = ", ".join(map(str, duplicate_maps))
            error_msg = f"Invalid mapping! The following map numbers are used more than once: {duplicate_str}"
            logger.error(error_msg)
            raise BadMapping(error_msg)

    def _collect_map_numbers(self) -> List[int]:
        """
        Collect all mapping numbers from the reactants.

        Returns:
            List[int]: A list of all mapping numbers.
        """
        map_nums = []
        for map_list in self.ratam.full_map_info["reactants"].values():
            for mapping in map_list:
                map_nums.extend(mapping.values())
        return map_nums

    @staticmethod
    def _find_duplicate_maps(map_nums: List[int]) -> List[int]:
        """
        Find mapping numbers that are used more than once, excluding 0 and -1.

        Args:
            map_nums (List[int]): List of all mapping numbers.

        Returns:
            List[int]: List of mapping numbers that are used more than once.
        """
        counter = Counter(map_nums)
        return [
            num for num, count in counter.items() if count > 1 and num not in {0, -1}
        ]

    def _create_atom_transformations(self):
        """To create a set of AtomTransformations"""
        self.ratam.atom_transformations = set()
        for p_uid, p_maps in self.ratam.full_map_info["products"].items():
            for p_map in p_maps:
                for r_uid, r_maps in self.ratam.full_map_info["reactants"].items():
                    for r_map in r_maps:
                        self._add_transformations(p_uid, p_map, r_uid, r_map)

    def _add_transformations(
        self, p_uid: str, p_map: Dict[int, int], r_uid: str, r_map: Dict[int, int]
    ):
        matching_map_nums = set(p_map.values()) & set(r_map.values()) - {0, -1}
        for map_num in matching_map_nums:
            p_atoms = [aid for aid, map_val in p_map.items() if map_val == map_num]
            r_atoms = [aid for aid, map_val in r_map.items() if map_val == map_num]
            self.ratam.atom_transformations.update(
                AtomTransformation(p_uid, r_uid, p_aid, r_aid, map_num)
                for p_aid in p_atoms
                for r_aid in r_atoms
            )

    def _analyze_unmapped_atoms(self):
        self.ratam.reactants_unmapped_atoms_info = self._get_unmapped_atoms_info(
            self.ratam.full_map_info["reactants"],
            {
                at
                for at in self.ratam.atom_transformations
                if at.reactant_uid in self.ratam.full_map_info["reactants"]
            },
        )

        dp_atom_transformations = {
            at
            for at in self.ratam.atom_transformations
            if at.product_uid == self.desired_product.uid
        }
        self.ratam.desired_product_unmapped_atoms_info = self._get_unmapped_atoms_info(
            {
                self.desired_product.uid: self.ratam.full_map_info["products"][
                    self.desired_product.uid
                ]
            },
            dp_atom_transformations,
        )

    @staticmethod
    def _get_unmapped_atoms_info(
        molecule_maps: Dict[Union[str, int], List[Dict[int, int]]],
        atom_transformations: Set[AtomTransformation],
    ) -> Dict[str, Union[Dict[Union[str, int], List[Set[int]]], float]]:
        info: Dict[str, Union[Dict[Union[str, int], List[Set[int]]], float]] = {
            "unmapped_atoms": {}
        }
        total_atoms = total_unmapped = 0

        for uid, map_list in molecule_maps.items():
            transformed_atoms = {
                at.map_num
                for at in atom_transformations
                if at.reactant_uid == uid or at.product_uid == uid
            }
            for mapping in map_list:
                unmapped = {
                    aid
                    for aid, map_num in mapping.items()
                    if map_num not in transformed_atoms
                }
                if unmapped:
                    if uid not in info["unmapped_atoms"]:
                        info["unmapped_atoms"][uid] = []
                    info["unmapped_atoms"][uid].append(unmapped)
                total_atoms += len(mapping)
                total_unmapped += len(unmapped)

        info["fraction"] = (
            round(total_unmapped / total_atoms, 2) if total_atoms > 0 else 0
        )
        return info


def create_ratam(
    reaction_components: ReactionComponents, desired_product: Molecule
) -> Ratam:
    """To create a Ratam object"""
    builder = RatamBuilder(reaction_components, desired_product)
    return builder.build()
