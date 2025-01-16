import copy
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import linchemin.cheminfo.functions as cif
from linchemin import utilities
from linchemin.cheminfo.chemical_hashes import calculate_disconnection_hash_map
from linchemin.cheminfo.models import ChemicalEquation, Disconnection, Molecule


class BondInfo(NamedTuple):
    product_atoms: Tuple[int, int]
    product_bond: int


@dataclass
class ReactiveCenter:
    """A service class storing data about a chemical reaction reactive center"""

    reactive_atoms: Dict[int, cif.Chem.Atom] = field(default_factory=dict)
    hydrogenated_atoms: Set[Tuple[int, int]] = field(default_factory=set)
    new_bonds: Set[BondInfo] = field(default_factory=set)
    changed_bonds: Set[BondInfo] = field(default_factory=set)


class ReactiveCenterAnalyzer:
    """An analyzer to extract data relevant for a ReactiveCenter"""

    @staticmethod
    def get_reacting_atoms_map_numbers(
        rdrxn: cif.rdChemReactions.ChemicalReaction,
    ) -> Dict[int, cif.Atom]:
        """To get a map of atom mapping number and the corresponding reactant atoms"""
        if not rdrxn.IsInitialized():
            rdrxn.Initialize()

        reacting_atoms_map = {}

        # Iterate over reactants and their corresponding reacting atoms
        for reactant, reactant_reacting_atoms in zip(
            rdrxn.GetReactants(), rdrxn.GetReactingAtoms()
        ):
            # For each reacting atom in the current reactant
            for atom_idx in reactant_reacting_atoms:
                # Get the atom object
                atom = reactant.GetAtomWithIdx(atom_idx)

                # Get the atom map number
                map_num = atom.GetAtomMapNum()

                # Only include atoms with valid map numbers (not 0 or -1)
                if map_num not in [0, -1]:
                    reacting_atoms_map[map_num] = atom

        return reacting_atoms_map

    @staticmethod
    def check_hydrogenation(
        product_atom: cif.Atom, reactant_atom: cif.Atom
    ) -> Optional[Tuple[int, int]]:
        """To collect information about atoms hydrogenation/dehydrogenation"""
        delta_h = product_atom.GetTotalNumHs() - reactant_atom.GetTotalNumHs()
        if delta_h != 0:
            return product_atom.GetIdx(), delta_h
        return None

    @staticmethod
    def get_bond_info(
        r_atom: cif.Atom,
        reactant: cif.Mol,
        p_atom: cif.Atom,
        product: cif.Mol,
        processed_bonds: Set[Tuple[int, int]],
    ) -> Tuple[Set[BondInfo], Set[BondInfo]]:
        """To collect information about new and changed bonds"""
        new_bonds = set()
        changed_bonds = set()
        r_atom_nbrs = ReactiveCenterAnalyzer.get_mapped_neighbors(r_atom)
        p_atom_nbrs = ReactiveCenterAnalyzer.get_mapped_neighbors(p_atom)

        for tpl in p_atom_nbrs:
            if tpl in processed_bonds:
                continue  # Skip already processed bonds

            processed_bonds.add(tpl)  # Mark this bond as processed
            p_bond = product.GetBondBetweenAtoms(*p_atom_nbrs[tpl])
            p_bond_id = p_bond.GetIdx()

            if tpl not in r_atom_nbrs:
                new_bonds.add(
                    BondInfo(product_atoms=p_atom_nbrs[tpl], product_bond=p_bond_id)
                )
            else:
                r_bond = reactant.GetBondBetweenAtoms(*r_atom_nbrs[tpl])
                if r_bond.GetBondType() != p_bond.GetBondType():
                    changed_bonds.add(
                        BondInfo(product_atoms=p_atom_nbrs[tpl], product_bond=p_bond_id)
                    )

        return new_bonds, changed_bonds

    @staticmethod
    def get_mapped_neighbors(atom: cif.Atom) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """To get the mapped atoms neighbors of the input atom"""
        neighbors = {}
        amap = atom.GetAtomMapNum()
        if amap in [0, -1]:
            return neighbors
        for bond in atom.GetBonds():
            neighbor = next(
                a
                for a in [bond.GetEndAtom(), bond.GetBeginAtom()]
                if a.GetAtomMapNum() != amap
            )
            n_map = neighbor.GetAtomMapNum()
            key = (min(amap, n_map), max(amap, n_map))
            neighbors[key] = (
                (atom.GetIdx(), neighbor.GetIdx())
                if amap < n_map
                else (neighbor.GetIdx(), atom.GetIdx())
            )
        return neighbors


class DisconnectionBuilder:
    """Builder of Disconnection objects"""

    def __init__(self, chemical_equation: ChemicalEquation, desired_product: Molecule):
        self.chemical_equation = chemical_equation
        self.desired_product = desired_product
        self.disconnection = Disconnection(
            identity_property=desired_product.identity_property
        )
        self.reactive_center = ReactiveCenter()
        self.analyzer = ReactiveCenterAnalyzer()

    def build(self) -> Disconnection:
        self._identify_reactive_center()
        self._populate_disconnection()
        return self.disconnection

    def _identify_reactive_center(self) -> None:
        self.reactive_center.reactive_atoms = (
            self.analyzer.get_reacting_atoms_map_numbers(self.chemical_equation.rdrxn)
        )
        ats_desired_product = (
            self._get_reactive_atom_transformations_for_desired_product()
        )
        self.product_reactive_atoms = sorted(
            [at.prod_atom_id for at in ats_desired_product]
        )

        processed_bonds = set()  # New set to keep track of processed bonds

        for atom_transformation in ats_desired_product:
            self._process_atom_transformation(atom_transformation, processed_bonds)

    def _process_atom_transformation(
        self, atom_transformation, processed_bonds: Set[Tuple[int, int]]
    ) -> None:
        reactant = self._get_reactant_for_atom_transformation(atom_transformation)
        reactant_atom, product_atom = self._get_reactant_and_product_atoms(
            atom_transformation
        )

        self._check_and_add_hydrogenation(product_atom, reactant_atom)
        self._check_and_add_bonds(
            reactant_atom, reactant, product_atom, processed_bonds
        )

    def _check_and_add_hydrogenation(
        self, product_atom: cif.Atom, reactant_atom: cif.Atom
    ) -> None:
        hydrogenation = self.analyzer.check_hydrogenation(product_atom, reactant_atom)
        if hydrogenation:
            self.reactive_center.hydrogenated_atoms.add(hydrogenation)

    def _check_and_add_bonds(
        self,
        reactant_atom: cif.Atom,
        reactant: Molecule,
        product_atom: cif.Atom,
        processed_bonds: Set[Tuple[int, int]],
    ) -> None:
        new_bonds, changed_bonds = self.analyzer.get_bond_info(
            reactant_atom,
            reactant.rdmol_mapped,
            product_atom,
            self.desired_product.rdmol_mapped,
            processed_bonds,
        )
        self.reactive_center.new_bonds.update(new_bonds)
        self.reactive_center.changed_bonds.update(changed_bonds)

    def _get_reactive_atom_transformations_for_desired_product(self) -> List:
        return [
            at
            for at in self.chemical_equation.mapping.atom_transformations
            if at.product_uid == self.desired_product.uid
            and at.map_num in self.reactive_center.reactive_atoms
        ]

    def _get_reactant_for_atom_transformation(self, atom_transformation) -> cif.Mol:
        return next(
            mol
            for h, mol in self.chemical_equation.catalog.items()
            if h == atom_transformation.reactant_uid
        )

    def _get_reactant_and_product_atoms(self, atom_transformation):
        reactant_atom = next(
            atom
            for map_nr, atom in self.reactive_center.reactive_atoms.items()
            if map_nr == atom_transformation.map_num
        )
        product_atom = next(
            atom
            for atom in self.desired_product.rdmol_mapped.GetAtoms()
            if atom.GetIdx() == atom_transformation.prod_atom_id
        )
        return reactant_atom, product_atom

    def _populate_disconnection(self):
        self.disconnection.molecule = copy.deepcopy(self.desired_product)
        self.disconnection.rdmol = copy.deepcopy(self.desired_product.rdmol_mapped)
        self.disconnection.reacting_atoms = self.product_reactive_atoms
        self.disconnection.hydrogenated_atoms = sorted(
            self.reactive_center.hydrogenated_atoms, key=lambda x: x[0]
        )
        self.disconnection.new_bonds = sorted(
            [tuple(sorted(bi.product_atoms)) for bi in self.reactive_center.new_bonds]
        )
        self.disconnection.modified_bonds = sorted(
            [
                tuple(sorted(bi.product_atoms))
                for bi in self.reactive_center.changed_bonds
            ]
        )
        self.disconnection.hash_map = calculate_disconnection_hash_map(
            self.disconnection, self.desired_product.identity_property
        )
        self.disconnection.identity_property = self.disconnection.hash_map.get(
            "disconnection_summary"
        )
        self.disconnection.uid = utilities.create_hash(
            self.disconnection.identity_property
        )


def create_disconnection(
    chemical_equation: ChemicalEquation, desired_product: Molecule
) -> Disconnection:
    builder = DisconnectionBuilder(
        chemical_equation=chemical_equation, desired_product=desired_product
    )
    return builder.build()
