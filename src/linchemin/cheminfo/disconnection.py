# Standard library imports
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

# Third party imports
from rdchiral import template_extractor

# Local imports
import linchemin.cheminfo.functions as cif
import linchemin.utilities as utilities
from linchemin.cheminfo.molecule import Molecule, MoleculeConstructor

AtomInfo = namedtuple(
    "AtomInfo", ("mapnum", "reactant", "reactant_atom", "product", "product_atom")
)
BondInfo = namedtuple(
    "BondInfo", ("product", "product_atoms", "product_bond", "status")
)


@dataclass(eq=True, order=False, frozen=False)
class RxnAtomInfo:
    """Class for keeping track of atoms involved in a reaction."""

    mapnum: int
    reactant: int
    reactant_atom: int
    product: int
    product_atom: int


@dataclass(eq=True, order=False, frozen=False)
class RxnBondInfo:
    """Class for keeping track of bonds that between reactants and products of a reaction are new or changed."""

    product: int
    product_atoms: tuple
    product_bond: int
    status: str


@dataclass(eq=True, order=False, frozen=False)
class RXNProductChanges:
    """Class for keeping track of the changes in a reaction product"""

    reacting_atoms: list[int]
    new_bonds: list[int]
    modified_bonds: list[int]


class Disconnection:
    def __init__(
        self,
        molecule: Molecule,
        rdmol: cif.Mol,
        rdmol_fragmented: cif.Mol,
        reacting_atoms: list[int],
        new_bonds: list[int],
        modified_bonds: list[int],
    ):
        self.molecule = molecule
        self.rdmol = molecule.rdmol
        # self.rdmol = rdmol
        self.rdmol_fragmented = rdmol_fragmented
        self.reacting_atoms = reacting_atoms
        self.new_bonds = new_bonds
        self.modified_bonds = modified_bonds

        self.hash_map = self.calculate_hash_values()
        self.identity_property = self.hash_map.get("disconnection_summary")
        self.uid = utilities.create_hash(self.identity_property)

    def __hash__(self) -> int:
        return self.uid

    def __eq__(self, other):
        return type(self) == type(other) and self.__hash__() == other.__hash__()

    def __str__(self) -> str:
        return f"{self.identity_property}"

    def to_dict(self) -> dict:
        return {
            "type": "Disconnection",
            "uid": self.uid,
            "hash_map": self.hash_map,
        }

    def calculate_hash_values(self):
        idp = self.molecule.identity_property
        changes = "__".join(
            [
                "_".join(map(str, self.reacting_atoms)),
                "_".join(map(str, self.new_bonds)),
                "_".join(map(str, self.modified_bonds)),
            ]
        )

        disconnection_summary = "|".join([idp, changes])

        return {"disconnection_summary": disconnection_summary}


class DisconnectionConstructor:
    def __init__(self, identity_property_name: str):
        self.identity_property_name = identity_property_name

    def build_from_rdrxn(
        self, rdrxn: cif.rdChemReactions.ChemicalReaction, desired_product_idx: int = 0
    ) -> Union[Disconnection, None]:
        if (
            len(rdrxn.GetReactants()) == 0 or len(rdrxn.GetProducts()) == 0
        ):  # TODO replace with a diagnosis at the CE level
            return None

        rxn_reactive_center = RXNReactiveCenter(rdrxn=rdrxn)

        if len(rxn_reactive_center.rxn_bond_info_list) == 0:
            return None

        product_rdmol = cif.Chem.Mol(rdrxn.GetProductTemplate(desired_product_idx))
        product_changes = rxn_reactive_center.get_product_changes(
            product_idx=desired_product_idx
        )

        molecule_constructor = MoleculeConstructor(
            identity_property_name=self.identity_property_name
        )
        product_molecule = molecule_constructor.build_from_rdmol(rdmol=product_rdmol)

        reacting_atoms, new_bonds, modified_bonds = self.re_map(
            product_changes, rdmol_old=product_rdmol, rdmol_new=product_molecule.rdmol
        )

        """
        rxn_atom_info_list = rxn_reactive_center.rxn_atom_info_list
        rxn_bond_info_list = rxn_reactive_center.rxn_bond_info_list
        print('rxn_atom_info_list:', rxn_atom_info_list)
        print('rxn_bond_info_list:', rxn_bond_info_list)
        print('reacting_atoms:', product_changes.reacting_atoms)
        print('new_bonds:', product_changes.new_bonds)
        print('modified_bonds:', product_changes.modified_bonds)
        """
        rdmol_fragmented = self.get_fragments(
            rdmol=product_rdmol, new_bonds=new_bonds, fragmentation_method=2
        )

        return Disconnection(
            molecule=product_molecule,
            rdmol=product_rdmol,
            rdmol_fragmented=rdmol_fragmented,
            reacting_atoms=reacting_atoms,
            new_bonds=new_bonds,
            modified_bonds=modified_bonds,
        )

    @staticmethod
    def re_map(product_changes, rdmol_old, rdmol_new):
        """
        rdmol_old and  rdmol_new differ by atom and bond ordering
        this function maps some atoms and bonds from the rdmol_old to rdmol_new

        """
        reacting_atoms_ = product_changes.reacting_atoms
        new_bonds_ = product_changes.new_bonds
        modified_bonds_ = product_changes.modified_bonds

        match = rdmol_new.GetSubstructMatch(rdmol_old)
        reacting_atoms = sorted([match[x] for x in reacting_atoms_])

        new_bonds = sorted(
            [
                rdmol_new.GetBondBetweenAtoms(
                    match[bond.GetBeginAtomIdx()], match[bond.GetEndAtomIdx()]
                ).GetIdx()
                for bond in rdmol_old.GetBonds()
                if bond.GetIdx() in new_bonds_
            ]
        )

        modified_bonds = sorted(
            [
                rdmol_new.GetBondBetweenAtoms(
                    match[bond.GetBeginAtomIdx()], match[bond.GetEndAtomIdx()]
                ).GetIdx()
                for bond in rdmol_old.GetBonds()
                if bond.GetIdx() in modified_bonds_
            ]
        )

        return reacting_atoms, new_bonds, modified_bonds

    def build_from_reaction_string(
        self, reaction_string: str, inp_fmt: str
    ) -> Union[Disconnection, None]:
        rdrxn = cif.rdrxn_from_string(input_string=reaction_string, inp_fmt=inp_fmt)
        rdrxn.Initialize()
        rdrxn.Validate()
        cif.Chem.rdChemReactions.PreprocessReaction(rdrxn)
        return self.build_from_rdrxn(rdrxn=rdrxn)

    def get_fragments(
        self, rdmol: cif.Mol, new_bonds: list[int], fragmentation_method: int = 1
    ):
        """
        inspired by
            https://github.com/rdkit/rdkit/issues/2081

        """

        def get_fragments_method_1(rdmol, bonds) -> cif.Mol:
            rdmol_fragmented = cif.rdkit.Chem.FragmentOnBonds(
                rdmol,
                bonds,
                addDummies=True,
                # dummyLabels=[(0, 0)]
            )
            return rdmol_fragmented

        def get_fragments_method_2(rdmol: cif.Mol, bonds) -> cif.Mol:
            mh = cif.Chem.RWMol(cif.Chem.AddHs(rdmol))
            cif.Chem.Kekulize(mh, clearAromaticFlags=True)

            for bond in [mh.GetBondWithIdx(idx) for idx in bonds]:
                a = bond.GetEndAtomIdx()
                b = bond.GetBeginAtomIdx()
                mh.RemoveBond(a, b)

                nAts = mh.GetNumAtoms()
                mh.AddAtom(cif.Chem.Atom(0))
                mh.AddBond(a, nAts, cif.Chem.BondType.SINGLE)
                mh.AddAtom(cif.Chem.Atom(0))
                mh.AddBond(b, nAts + 1, cif.Chem.BondType.SINGLE)
                nAts += 2
            cif.Chem.SanitizeMol(mh)
            cif.rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_NONE
            cif.Chem.RemoveHs(mh)

            return mh

        fragmentation_method_factory = {
            1: get_fragments_method_1,
            2: get_fragments_method_2,
        }

        frag_func = fragmentation_method_factory[fragmentation_method]
        from rdkit.Chem import AllChem

        AllChem.Compute2DCoords(rdmol)
        rdmol_fragmented = frag_func(rdmol=rdmol, bonds=new_bonds)
        return rdmol_fragmented


class RXNReactiveCenter:
    """
    Class to identify the atoms and bonds that change in a reaction
    inspired from
    https://greglandrum.github.io/rdkit-blog/tutorial/reactions/2021/11/26/highlighting-changed-bonds-in-reactions.html
    """

    def __init__(self, rdrxn: cif.rdChemReactions.ChemicalReaction):
        rdrxn.Initialize()
        (
            self.rxn_atom_info_list,
            self.rxn_bond_info_list,
        ) = self.find_modifications_in_products(rdrxn)

    def get_product_changes(self, product_idx: int = 0):
        reacting_atoms = sorted(
            [
                ai.product_atom
                for ai in self.rxn_atom_info_list
                if ai.product == product_idx
            ]
        )
        new_bonds = sorted(
            [
                bi.product_bond
                for bi in self.rxn_bond_info_list
                if bi.product == product_idx and bi.status == "new"
            ]
        )
        modified_bonds = sorted(
            [
                bi.product_bond
                for bi in self.rxn_bond_info_list
                if bi.product == product_idx and bi.status == "changed"
            ]
        )
        return RXNProductChanges(reacting_atoms, new_bonds, modified_bonds)

    @staticmethod
    def map_reacting_atoms_to_products(
        rdrxn: cif.rdChemReactions.ChemicalReaction, reactingAtoms
    ):
        """figures out which atoms in the products each mapped atom in the reactants maps to"""
        res = []
        for ridx, reacting in enumerate(reactingAtoms):
            reactant = rdrxn.GetReactantTemplate(ridx)
            for raidx in reacting:
                mapnum = reactant.GetAtomWithIdx(raidx).GetAtomMapNum()
                foundit = False
                for pidx, product in enumerate(rdrxn.GetProducts()):
                    for paidx, patom in enumerate(product.GetAtoms()):
                        if patom.GetAtomMapNum() == mapnum:
                            res.append(AtomInfo(mapnum, ridx, raidx, pidx, paidx))
                            foundit = True
                            break
                        if foundit:
                            break
        return res

    @staticmethod
    def get_mapped_neighbors(atom: cif.Atom):
        """test all mapped neighbors of a mapped atom"""
        res = {}
        amap = atom.GetAtomMapNum()
        if not amap:
            return res
        for nbr in atom.GetNeighbors():
            nmap = nbr.GetAtomMapNum()
            if nmap:
                if amap > nmap:
                    res[(nmap, amap)] = (atom.GetIdx(), nbr.GetIdx())
                else:
                    res[(amap, nmap)] = (nbr.GetIdx(), atom.GetIdx())
        return res

    def find_modifications_in_products(
        self, rxn
    ) -> tuple[list[RxnAtomInfo], list[RxnBondInfo]]:
        """returns a 2-tuple with the modified atoms and bonds from the reaction"""
        reactingAtoms = rxn.GetReactingAtoms()
        amap = self.map_reacting_atoms_to_products(rxn, reactingAtoms)
        res = []
        seen = set()
        # this is all driven from the list of reacting atoms:
        for _, ridx, raidx, pidx, paidx in amap:
            reactant = rxn.GetReactantTemplate(ridx)
            ratom = reactant.GetAtomWithIdx(raidx)
            product = rxn.GetProductTemplate(pidx)
            patom = product.GetAtomWithIdx(paidx)

            rnbrs = self.get_mapped_neighbors(ratom)
            pnbrs = self.get_mapped_neighbors(patom)
            for tpl in pnbrs:
                pbond = product.GetBondBetweenAtoms(*pnbrs[tpl])
                if (pidx, pbond.GetIdx()) in seen:
                    continue
                seen.add((pidx, pbond.GetIdx()))
                if tpl not in rnbrs:
                    # new bond in product
                    res.append(BondInfo(pidx, pnbrs[tpl], pbond.GetIdx(), "new"))
                else:
                    # present in both reactants and products, check to see if it changed
                    rbond = reactant.GetBondBetweenAtoms(*rnbrs[tpl])
                    if rbond.GetBondType() != pbond.GetBondType():
                        res.append(
                            BondInfo(pidx, pnbrs[tpl], pbond.GetIdx(), "changed")
                        )
        rxn_atom_info_list = [RxnAtomInfo(**item._asdict()) for item in amap]
        rxn_bond_info_list = [RxnBondInfo(**item._asdict()) for item in res]
        return rxn_atom_info_list, rxn_bond_info_list


if __name__ == "__main__":
    print("disconnection main")
