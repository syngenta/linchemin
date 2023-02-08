# Standard library imports
from typing import Dict, List, Tuple

# Local imports
import linchemin.cheminfo.functions as cif
import linchemin.utilities as utilities
from linchemin.cheminfo.disconnection import Disconnection, DisconnectionConstructor
from linchemin.cheminfo.molecule import Molecule, MoleculeConstructor
from linchemin.cheminfo.template import Template, TemplateConstructor

# Third party imports


class ChemicalEquation:
    """Class holding information of chemical reactions.

    Attributes:

        catalog: a dictionary containing the instances of the Molecule objects involved in the reaction

        role_map: a dictionary mapping the roles of each Molecule object involved in 'reactants', 'reagents' and
                  'products'

        stoichiometry_coefficients: a dictionary mapping the stoichiometry coefficients of each Molecule object
                                    involved in the reaction

        uid: the hash key identifying the ChemicalEquation instance

        rdrxn: the rdkit ChemicalReaction object associated with the ChemicalEquation instance

        smiles: the smiles string associated with the ChemicalEquation instance

    """

    def __init__(
        self,
        catalog: dict[int, Molecule],
        role_map: dict,
        stoichiometry_coefficients: dict,
    ):
        self.catalog = catalog

        self.role_map = role_map
        self.stoichiometry_coefficients = stoichiometry_coefficients

        ################
        self.roles = role_map  # TODO: remove me and chase down in cgu
        self.molecules = catalog  # TODO: remove me and chase down in cgu
        ################

        self.hash_map = self.calculate_hash_values()
        self.uid = self.hash_map.get("r_r_p")  # TODO: review
        self.rdrxn = self.build_rdrxn()
        self.smiles = cif.rdrxn_to_string(
            rdrxn=self.rdrxn, out_fmt="smiles", use_atom_mapping=True
        )

        # add disconnection
        dc = DisconnectionConstructor(identity_property_name="smiles")
        self.disconnection = dc.build_from_rdrxn(
            rdrxn=self.build_rdrxn(use_atom_mapping=True)
        )

        # add template
        tc = TemplateConstructor()
        # self.template = tc.build_from_rdrxn(rdrxn=self.build_rdrxn(use_atom_mapping=True))
        self.template = tc.build_from_reaction_string(
            reaction_string=self.smiles, inp_fmt="smiles"
        )

    def __hash__(self) -> int:
        return self.uid

    def __eq__(self, other):
        return type(self) == type(other) and self.__hash__() == other.__hash__()

    def __str__(self) -> str:
        return f"{self.smiles}"

    def to_dict(self) -> dict:
        """To return a dictionary with all the attributes of the ChemicalEquation instance"""
        return {
            "type": "ChemicalEquation",
            "uid": self.uid,
            "smiles": self.smiles,
            "hash_map": self.hash_map,
            "roles": self.role_map,
            "stoichiometry_coefficients": self.stoichiometry_coefficients,
            "template": self.template.to_dict() if self.template else None,
            "disconnection": self.disconnection.to_dict()
            if self.disconnection
            else None,
        }

    def calculate_hash_values(self):
        """To calculate the hash key identifying the ChemicalEquation instance"""
        mol_list_map = {
            role: [self.catalog.get(uid) for uid in uid_list]
            for role, uid_list in self.role_map.items()
        }

        # get the identity_property for each molecule
        idp_list_map = {
            role: [m.identity_property for m in molecule_list]
            for role, molecule_list in mol_list_map.items()
        }

        # for each role concatenate the properties
        idp_str_map = {role: ".".join(sorted(v)) for role, v in idp_list_map.items()}

        # add some more strings
        idp_str_map["r_p"] = ">>".join(
            [idp_str_map.get("reactants"), idp_str_map.get("products")]
        )
        idp_str_map["r_r_p"] = ">".join(
            [
                idp_str_map.get("reactants"),
                idp_str_map.get("reagents"),
                idp_str_map.get("products"),
            ]
        )
        idp_str_map["u_r_p"] = ">>".join(
            sorted([idp_str_map.get("reactants"), idp_str_map.get("products")])
        )
        idp_str_map["u_r_r_p"] = ">>".join(
            sorted(
                [
                    idp_str_map.get("reactants"),
                    idp_str_map.get("reagents"),
                    idp_str_map.get("products"),
                ]
            )
        )
        return {role: utilities.create_hash(v) for role, v in idp_str_map.items()}

    def build_reaction_smiles(self) -> str:
        """To build the reactio smiles from the smiles of the involved Molecule instances"""
        return ">".join(
            [
                ".".join([self.catalog.get(uid).smiles for uid in self.role_map[role]])
                for role in ["reactants", "reagents", "products"]
            ]
        )

    def build_rdrxn(self, use_atom_mapping=True):
        """To build the rdkit ChemicalReaction object associated with the ChemicalEquation instance"""
        return cif.build_rdrxn(
            catalog=self.catalog,
            role_map=self.role_map,
            stoichiometry_coefficients=self.stoichiometry_coefficients,
            use_smiles=False,
            use_atom_mapping=use_atom_mapping,
        )

    ######################


class ChemicalEquationConstructor:
    """Class implementing the constructor of the ChemicalEquation class

    Attributes:
        identity_property_name: a string indicating which kind of input string determines the identity
                                of the object (e.g. 'smiles')

    """

    def __init__(self, identity_property_name: str):
        self.identity_property_name = identity_property_name

    def read_reaction(
        self, reaction_string: str, inp_fmt: str
    ) -> tuple[cif.rdChemReactions.ChemicalReaction, utilities.OutcomeMetadata]:
        """To start the building of a ChemicalEquation instance from a reaction string"""
        try:
            rdrxn = cif.rdrxn_from_string(input_string=reaction_string, inp_fmt=inp_fmt)
            sanitize_msg = cif.rdChemReactions.SanitizeRxn(rdrxn, catchErrors=True)

            outcome = utilities.OutcomeMetadata(
                name="read_reaction",
                is_successful=True,
                log={"sanitize_msg": sanitize_msg},
            )
        except Exception as e:
            outcome = utilities.OutcomeMetadata(
                name="read_reaction", is_successful=False, log={"exception": e}
            )
            rdrxn = None

        return rdrxn, outcome

    def unpack_rdrxn(self, rdrxn: cif.rdChemReactions.ChemicalReaction):
        """To compute ChemicalEquation attributes from the associated rdkit ChemicalReaction object"""
        constructor = MoleculeConstructor(
            identity_property_name=self.identity_property_name
        )
        catalog, role_map, stoichiometry_coefficients = cif.unpack_rdrxn(
            rdrxn=rdrxn,
            identity_property_name=self.identity_property_name,
            constructor=constructor,
        )
        return catalog, stoichiometry_coefficients, role_map

    def build_from_rdrxn(
        self, rdrxn: cif.rdChemReactions.ChemicalReaction
    ) -> ChemicalEquation:
        """To build a ChemicalEquation instance from an rdkit ChemicalReaction object"""
        catalog, stoichiometry_coefficients, role_map = self.unpack_rdrxn(rdrxn=rdrxn)

        return ChemicalEquation(
            catalog=catalog,
            role_map=role_map,
            stoichiometry_coefficients=stoichiometry_coefficients,
        )

    def build_from_reaction_string(
        self, reaction_string: str, inp_fmt: str
    ) -> ChemicalEquation:
        """To build a ChemicalEquation instance from a reaction string"""
        rdrxn, read_outcome = self.read_reaction(
            reaction_string=reaction_string, inp_fmt=inp_fmt
        )

        return self.build_from_rdrxn(rdrxn=rdrxn)


if __name__ == "__main__":
    print("main")
