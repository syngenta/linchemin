# Sstandard library imports
import pprint
from typing import Dict, List, Tuple, Union

# Third party imports
from rdchiral import template_extractor

# Local imports
import linchemin.cheminfo.functions as cif
import linchemin.utilities as utilities


class Pattern:
    def __init__(self, rdmol: cif.Mol):
        rdmol_mapped = rdmol
        self.identity_property_name = "smarts"
        rdmol_unmapped = cif.remove_rdmol_atom_mapping(rdmol=rdmol_mapped)
        rdmol_unmapped_canonical = cif.canonicalize_rdmol_lite(
            rdmol=rdmol_unmapped, is_pattern=True
        )
        rdmol_mapped_canonical = cif.canonicalize_rdmol_lite(
            rdmol=rdmol_mapped, is_pattern=True
        )
        self.rdmol = rdmol_unmapped_canonical
        self.rdmol_mapped = rdmol_mapped_canonical
        self.smarts = cif.compute_mol_smarts(rdmol=self.rdmol)
        self.hash_map = self.calculate_hash_values()

        self.identity_property = self.hash_map.get(self.identity_property_name)
        self.uid = utilities.create_hash(
            self.identity_property
        )  # the hashed identity property

    def __hash__(self) -> int:
        return self.uid

    def __eq__(self, other) -> bool:
        return type(self) == type(other) and self.__hash__() == other.__hash__()

    def __str__(self) -> str:
        return f"{self.smarts}"

    def to_dict(self) -> dict:
        return {
            "type": "Molecule",
            "uid": self.uid,
            "smarts": self.smarts,
            "hash_map": self.hash_map,
        }

    def calculate_hash_values(self):
        return {"smarts": self.smarts}


class PatternConstructor:
    def __init__(self, identity_property_name: str = "smarts"):
        self.identity_property_name = identity_property_name

    def build_from_molecule_string(
        self,
        molecule_string: str,
        inp_fmt: str,
    ) -> Pattern:
        rdmol_input = cif.rdmol_from_string(
            input_string=molecule_string, inp_fmt=inp_fmt
        )
        return Pattern(rdmol=rdmol_input)

    def build_from_rdmol(self, rdmol: cif.Mol) -> Pattern:
        return Pattern(rdmol=rdmol)


class Template:
    def __init__(
        self,
        pattern_catalog: dict[int, Pattern],
        role_map: dict,
        stoichiometry_coefficients: dict,
        rdchiral_data: dict,
    ):
        self.pattern_catalog = pattern_catalog
        self.role_map = role_map
        self.stoichiometry_coefficients = stoichiometry_coefficients
        self.rdchiral_data = rdchiral_data

        self.hash_map = self.calculate_hash_values()
        self.uid = self.hash_map.get("r_p")  # TODO: review
        self.rdrxn = self.build_rdrxn()
        self.smarts = cif.rdrxn_to_string(rdrxn=self.rdrxn, out_fmt="smarts")

    def __hash__(self) -> int:
        return self.uid

    def __eq__(self, other):
        return type(self) == type(other) and self.__hash__() == other.__hash__()

    def __str__(self) -> str:
        return f"{self.smarts}"

    def to_dict(self) -> dict:
        return {
            "type": "Template",
            "uid": self.uid,
            "smarts": self.smarts,
            "hash_map": self.hash_map,
            "role_map": self.role_map,
            "rdchiral_data": self.rdchiral_data,
            "stoichiometry_coefficients": self.stoichiometry_coefficients,
        }

    def calculate_hash_values(self):
        pattern_list_map = {
            role: [self.pattern_catalog.get(uid) for uid in uid_list]
            for role, uid_list in self.role_map.items()
        }

        # get the identity_property for each molecule
        idp_list_map = {
            role: [m.identity_property for m in pattern_list]
            for role, pattern_list in pattern_list_map.items()
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
        return ">".join(
            [
                ".".join([self.role_map.get(uid).smiles for uid in self.role_map[role]])
                for role in ["reactants", "reagents", "products"]
            ]
        )

    def build_rdrxn(self, use_atom_mapping=True):
        return cif.build_rdrxn(
            catalog=self.pattern_catalog,
            role_map=self.role_map,
            stoichiometry_coefficients=self.stoichiometry_coefficients,
            use_smiles=False,
            use_atom_mapping=use_atom_mapping,
        )


def rdchiral_extract_template(
    reaction_string: str, inp_fmt: str, reaction_id: int = None
):
    if inp_fmt != "smiles":
        raise NotImplementedError
    mapped_smiles_split = reaction_string.split(">")
    rdchiral_input = {
        "_id": reaction_id,
        "reactants": mapped_smiles_split[0],
        "agents": mapped_smiles_split[1],
        "products": mapped_smiles_split[2],
    }
    return template_extractor.extract_from_reaction(reaction=rdchiral_input)


class TemplateConstructor:
    def __init__(self, identity_property_name: str = "smarts"):
        self.identity_property_name = identity_property_name

    def read_reaction(
        self, reaction_string: str, inp_fmt: str
    ) -> tuple[cif.rdChemReactions.ChemicalReaction, utilities.OutcomeMetadata]:
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
        constructor = PatternConstructor(
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
    ) -> Union[Template, None]:
        reaction_string = cif.rdChemReactions.ReactionToSmarts(rdrxn)
        return self.build_from_reaction_string(
            reaction_string=reaction_string, inp_fmt="smarts"
        )

    def build_from_reaction_string(
        self, reaction_string: str, inp_fmt: str
    ) -> Union[Template, None]:
        rdchiral_output = rdchiral_extract_template(
            reaction_string=reaction_string, inp_fmt=inp_fmt, reaction_id=None
        )
        return self.build_from_rdchiral_output(rdchiral_output=rdchiral_output)

    def build_from_rdchiral_output(
        self, rdchiral_output: dict
    ) -> Union[Template, None]:
        if not rdchiral_output:
            return None
        if not rdchiral_output.get("reaction_smarts"):
            return None
        reaction_rwd_smarts = rdchiral_output.get("reaction_smarts")
        reaction_fwd_smarts = ">".join(reaction_rwd_smarts.split(">")[::-1])
        rdrxn, read_outcome = self.read_reaction(
            reaction_string=reaction_fwd_smarts, inp_fmt="smarts"
        )

        pattern_catalog, stoichiometry_coefficients, role_map = self.unpack_rdrxn(
            rdrxn=rdrxn
        )

        rdchiral_data = rdchiral_output.copy()
        rdchiral_data.pop("reaction_id")
        rdchiral_data.pop("reaction_smarts")
        rdchiral_data["reaction_fwd_smarts"] = reaction_fwd_smarts
        rdchiral_data["reaction_rwd_smarts"] = reaction_rwd_smarts

        return Template(
            pattern_catalog=pattern_catalog,
            role_map=role_map,
            stoichiometry_coefficients=stoichiometry_coefficients,
            rdchiral_data=rdchiral_data,
        )


if __name__ == "__main__":
    print("DODDOODO")
    tc = TemplateConstructor(identity_property_name="smarts")
    mapped_smiles = "[CH3:1][CH2:2][NH:3][CH3:4]>>[CH3:1][CH2:2][NH2+1:3][CH3:4]"
    # mapped_smiles = '[#6:1]-[N+:2]#[N:3]=[N-:4]>>[#6:1]-[N+0:2]=[N+1:3]=[N-:4]'
    mapper_smiles = "CC#N.[CH3:1][C:2](=[O:3])Cl.[CH3:4][NH2:5]>>[CH3:1][C:2](=[O:3])[NH:5][CH3:4].O "
    mapped_smiles = "[CH3:1][O:2][c:3]1[cH:4][cH:5][c:6]([c:9]([cH:12]1)[O:10][CH3:11])[C:7](=[O:8])Cl.[NH4+:13]>[OH-]>[CH3:1][O:2][c:3]1[cH:4][cH:5][c:6]([c:9]([cH:12]1)[O:10][CH3:11])[C:7](=[O:8])[NH2:13]"
    # mapped_smiles = '>>[CH3:1][NH:2][C:3]([CH3:4])=[O:5]'

    print(10 * "#")
    mapped_smiles = (
        "[CH3:1][NH2:2].O[C:3]([CH3:4])=[O:5]>O>[CH3:1][NH:2][C:3]([CH3:4])=[O:5]"
    )
    template = tc.build_from_reaction_string(
        reaction_string=mapped_smiles, inp_fmt="smiles"
    )
    print(mapped_smiles)
    pprint.pprint(template.to_dict())
    print(10 * "#")
    mapped_smiles = (
        "[CH3:1][NH:2][C:3]([CH3:4])=[O:5]>O>[CH3:1][NH2:2].O[C:3]([CH3:4])=[O:5]"
    )
    template = tc.build_from_reaction_string(
        reaction_string=mapped_smiles, inp_fmt="smiles"
    )
    print(mapped_smiles)
    pprint.pprint(template.to_dict())
