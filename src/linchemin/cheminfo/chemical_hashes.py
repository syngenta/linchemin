from abc import ABC, abstractmethod
from typing import List, Type, Union

import linchemin.cheminfo.functions as cif
import linchemin.utilities as utilities
from linchemin.cheminfo.models import Disconnection
from linchemin.utilities import console_logger


class UnavailableMolIdentifier(KeyError):
    """To be raised if the selected name for the molecular property name is not among the available ones"""

    pass


class UnavailableReactionIdentifier(KeyError):
    """To be raised if the selected name for the reaction identifier is not among the available ones"""

    pass


logger = console_logger(__name__)


# Molecular Identifiers to be used as hash keys
class MolecularIdenfier(ABC):
    """Abstract class Molecular Identifiers"""

    @abstractmethod
    def get_identifier(self, rdmol: cif.Mol) -> str:
        pass


class MolIdentifierFactory:
    """Factory to give access to the Molecular Identifier Generators"""

    _identifiers = {}

    @classmethod
    def register_mol_identifier(cls, name: str):
        """
        Decorator for registering a new molecular identifier.

        Parameters:
        ------------
        name: str
            The name of the identifier to be used as a key in the registry


        Returns:
        ---------
        function: The decorator function.
        """

        def decorator(mol_identifier: Type[MolecularIdenfier]):
            cls._identifiers[name.lower()] = mol_identifier

            return mol_identifier

        return decorator

    @classmethod
    def select_identifier(cls, name: str) -> MolecularIdenfier:
        """
        To get an instance of the specified MolecularIdenfier.

        Parameters:
        ------------
        name: str
            The name of the identifier.

        Returns:
        ---------
        MolecularIdenfier: An instance of the specified molecular identifier.

        Raises:
        -------
        UnavailableMolIdentifier: If the specified identifier is not registered.
        """
        identifier = cls._identifiers.get(name.lower())
        if identifier is None:
            logger.error(f"Identifier '{name}' not found")
            raise UnavailableMolIdentifier
        return identifier()

    @classmethod
    def list_mol_identifiers(cls) -> list:
        """
        To list the names of all available molecular identifiers.

        Returns:
        ---------
        idetifiers: list
            The names of the available molecular identifiers.
        """
        return list(cls._identifiers.keys())


@MolIdentifierFactory.register_mol_identifier("inchi_key")
class InchiKey(MolecularIdenfier):
    """To compute inchKey"""

    def get_identifier(self, rdmol: cif.Mol) -> str:
        return cif.Chem.inchi.MolToInchiKey(rdmol)


@MolIdentifierFactory.register_mol_identifier("inchikey_ket_15T")
class InchiKeyKET15(MolecularIdenfier):
    """To compute InchiKeyKET15T"""

    def get_identifier(self, rdmol: cif.Mol) -> str:
        return cif.Chem.MolToInchiKey(rdmol, options="-KET -15T")


@MolIdentifierFactory.register_mol_identifier("inchi")
class Inchi(MolecularIdenfier):
    """To compute inch and inchKey"""

    def get_identifier(self, rdmol: cif.Mol) -> str:
        return cif.Chem.MolToInchi(rdmol)


@MolIdentifierFactory.register_mol_identifier("inchi_ket_15T")
class InchiKET15(MolecularIdenfier):
    """To compute InchiKET15T"""

    def get_identifier(self, rdmol: cif.Mol) -> str:
        return cif.Chem.MolToInchi(rdmol, options="-KET -15T")


@MolIdentifierFactory.register_mol_identifier("noiso_smiles")
class NoisoSmiles(MolecularIdenfier):
    """To compute Noiso smiles"""

    def get_identifier(self, rdmol: cif.Mol) -> str:
        return cif.Chem.MolToSmiles(rdmol, isomericSmiles=False)


@MolIdentifierFactory.register_mol_identifier("cx_smiles")
class CxSmiles(MolecularIdenfier):
    """To compute CxSmiles"""

    def get_identifier(self, rdmol: cif.Mol) -> str:
        return cif.Chem.MolToCXSmiles(rdmol)


def calculate_molecular_hash_map(
    all_available_identifiers: list, rdmol: cif.Mol, hash_list: Union[set, None] = None
) -> dict:
    """To compute the hash_map dictionary containing molecular properties/representations names and the
    corresponding hash values"""
    molhashf = cif.HashFunction.names
    if hash_list is None:
        hash_list = all_available_identifiers
    hash_map = {}

    if rdkit_hashes := [h for h in hash_list if h in molhashf]:
        hash_map.update(
            {k: cif.MolHash(rdmol, v) for k, v in molhashf.items() if k in rdkit_hashes}
        )
    if "smiles" in hash_list:
        hash_map.update(
            {
                "smiles": cif.MolHash(rdmol, v)
                for k, v in molhashf.items()
                if k == "CanonicalSmiles"
            }
        )

    if other_hashes := [h for h in hash_list if h not in rdkit_hashes]:
        factory = MolIdentifierFactory
        for h in other_hashes:
            if h.lower() not in all_available_identifiers:
                logger.warning(f"{h} is not supported as molecular identifier")
            elif h != "smiles":
                hash_map[h] = factory.select_identifier(h).get_identifier(rdmol)
    """

    hash_map['ExtendedMurcko_AG'] = smiles_to_anonymus_graph(hash_map['ExtendedMurcko'])
    hash_map['ExtendedMurcko_EG'] = smiles_to_element_graph(hash_map['ExtendedMurcko'])
    hash_map['MurckoScaffold_AG'] = smiles_to_anonymus_graph(hash_map['MurckoScaffold'])
    hash_map['MurckoScaffold_EG'] = smiles_to_element_graph(hash_map['MurckoScaffold'])
    """
    return hash_map


def get_all_molecular_identifiers() -> List:
    """To list all the available molecular identifiers"""
    return (
        list(cif.HashFunction.names.keys())
        + MolIdentifierFactory.list_mol_identifiers()
        + ["smiles"]
    )


def is_valid_molecular_identifier(molecular_identifier: str) -> bool:
    """To check if the input molecule identifier is among the valid ones"""
    return molecular_identifier in get_all_molecular_identifiers()


def validate_molecular_identifier(molecular_identifier: str) -> None:
    """To validate the input molecule identifier"""
    if not is_valid_molecular_identifier(molecular_identifier):
        logger.error(f"{molecular_identifier} is not a valid molecular identifier.")
        raise UnavailableMolIdentifier


# ChemicalEquation hash calculations
class ReactionIdentifier(ABC):
    name: str

    @staticmethod
    @abstractmethod
    def get_reaction_identifier(molecules_identity_property_map: dict) -> str:
        pass


class ReactionIdentifiersFactory:
    """Factory to give access to the ReactionRepresentation objects"""

    _reaction_identifiers = {}

    @classmethod
    def register_reaction_identifier(cls, identifier: Type[ReactionIdentifier]):
        """
        Decorator method to register a ReactionIdentifier implementation.
        Adds the 'name' attribute of the class to the registry.
        """
        if (
            hasattr(identifier, "name")
            and identifier.name not in cls._reaction_identifiers
        ):
            cls._reaction_identifiers[identifier.name.lower()] = identifier
        return identifier

    @classmethod
    def select_reaction_identifier(cls, name: str) -> ReactionIdentifier:
        """
        To get an instance of the specified ReactionIdentifier.

        Parameters:
        ------------
        name: str
            The name of the reaction identifier.

        Returns:
        ---------
        ReactionIdentifier: An instance of the specified reaction representation.

        Raises:
        -------
        UnavailableReactionIdentifier: If the specified identifier is not registered.
        """
        representation = cls._reaction_identifiers.get(name.lower())
        if representation is None:
            logger.error(f"Reaction property '{name}' not found")
            raise UnavailableReactionIdentifier
        return representation()

    @classmethod
    def list_reaction_identifiers(cls) -> list:
        """
        To list the names of all available reaction identifiers
        Returns:
        ---------
        list
            The names of the available reaction identifiers
        """
        return list(cls._reaction_identifiers.keys())


@ReactionIdentifiersFactory.register_reaction_identifier
class ReactantProduct(ReactionIdentifier):
    """Class corresponding to a representation of a reaction which includes reactants and products"""

    name = "r_p"

    @staticmethod
    def get_reaction_identifier(molecules_identity_property_map: dict) -> str:
        return ">>".join(
            [
                molecules_identity_property_map.get("reactants"),
                molecules_identity_property_map.get("products"),
            ]
        )


@ReactionIdentifiersFactory.register_reaction_identifier
class ReactantReagentProduct(ReactionIdentifier):
    """Class corresponding to a representation of a reaction which includes reactants,reagents and products"""

    name = "r_r_p"

    @staticmethod
    def get_reaction_identifier(molecules_identity_property_map: dict) -> str:
        return ">".join(
            [
                molecules_identity_property_map.get("reactants"),
                molecules_identity_property_map.get("reagents"),
                molecules_identity_property_map.get("products"),
            ]
        )


@ReactionIdentifiersFactory.register_reaction_identifier
class UnorderedReactantProduct(ReactionIdentifier):
    """Class corresponding to a representation of a reaction which includes reactants
    and products sorted alphanumerically"""

    name = "u_r_p"

    @staticmethod
    def get_reaction_identifier(molecules_identity_property_map: dict) -> str:
        return ">>".join(
            sorted(
                [
                    molecules_identity_property_map.get("reactants"),
                    molecules_identity_property_map.get("products"),
                ]
            )
        )


@ReactionIdentifiersFactory.register_reaction_identifier
class UnorderedReactantReagentProduct(ReactionIdentifier):
    """Class corresponding to a representation of a reaction which includes reactants, reagents
    and products sorted alphanumerically"""

    name = "u_r_r_p"

    @staticmethod
    def get_reaction_identifier(molecules_identity_property_map: dict) -> str:
        return ">>".join(
            sorted(
                [
                    molecules_identity_property_map.get("reactants"),
                    molecules_identity_property_map.get("reagents"),
                    molecules_identity_property_map.get("products"),
                ]
            )
        )


def calculate_reaction_like_hash_map(catalog: dict, role_map: dict) -> dict:
    """To calculate the hash keys for reaction-like objects"""
    mol_list_map = {
        role: [catalog.get(uid) for uid in uid_list]
        for role, uid_list in role_map.items()
    }
    # get the identity_property for each molecule
    idp_list_map = {
        role: [m.identity_property for m in molecule_list]
        for role, molecule_list in mol_list_map.items()
    }
    # for each role concatenate the properties
    idp_str_map = {role: ".".join(sorted(v)) for role, v in idp_list_map.items()}

    # get all the available reaction representations
    for identifier in ReactionIdentifiersFactory.list_reaction_identifiers():
        reaction_identifier = ReactionIdentifiersFactory.select_reaction_identifier(
            identifier
        )
        idp_str_map[
            reaction_identifier.name
        ] = reaction_identifier.get_reaction_identifier(idp_str_map)
    return {role: utilities.create_hash(v) for role, v in idp_str_map.items()}


def get_all_reaction_identifiers():
    return ReactionIdentifiersFactory.list_reaction_identifiers()


def is_valid_reaction_identifier(reaction_identifier: str) -> bool:
    """To check if the input reaction identifier is among the valid ones"""
    return reaction_identifier in get_all_reaction_identifiers()


def validate_reaction_identifier(reaction_identifier: str) -> None:
    """To validate the input reaction identifier"""
    if not is_valid_reaction_identifier(reaction_identifier):
        logger.error(
            f"{reaction_identifier} is not a valid reaction identity property."
        )
        raise UnavailableReactionIdentifier


def calculate_disconnection_hash_map(disconnection: Disconnection) -> dict:
    idp = disconnection.molecule.identity_property

    changes_map = {
        "reacting_atoms": disconnection.reacting_atoms,
        "hydrogenated_atoms": disconnection.hydrogenated_atoms,
        "new_bonds": disconnection.new_bonds,
        "mod_bonds": disconnection.modified_bonds,
    }

    """
    | separates properties and is followed by the name and a:
    """
    changes_str = "|".join(
        [f'{k}:{",".join(map(str, v))}' for k, v in changes_map.items()]
    )

    disconnection_summary = "|".join([idp, changes_str])

    return {"disconnection_summary": disconnection_summary}


def calculate_pattern_hash_map(smarts) -> dict:
    return {"smarts": smarts}
