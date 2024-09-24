from abc import ABC, abstractmethod
from typing import List, Optional, Type, Union

import linchemin.IO.io as lio
from linchemin.cgu.syngraph import (
    BipartiteSynGraph,
    MonopartiteMolSynGraph,
    MonopartiteReacSynGraph,
)
from linchemin.interfaces.facade import facade
from linchemin.utilities import console_logger

"""Module containing classes and functions to write SynGraph to various types of files"""

logger = console_logger(__name__)


class UnavailableWriter(KeyError):
    """Raised when an unavailable writer is selected"""


# Writers factory
class SyngraphWriter(ABC):
    """Abstract class for the SyngraphWriter taking care of writing the routes in different file formats"""

    file_format: str

    @abstractmethod
    def write_file(
        self, syngraphs: list, out_data_model: str, file_name: Optional[str]
    ) -> None:
        pass


class SyngraphWriterFactory:
    _file_formats = {}

    @classmethod
    def register_writer(cls, writer: Type[SyngraphWriter]):
        """
        Decorator method to register a SyngraphWriter implementation.
        Maps the 'file_format' attribute of the class to the class itself in the registry.
        """  # noqa: E501
        # Ensure that the writer class has a 'name' attribute,
        # and it's not already registered
        if (
            hasattr(writer, "file_format")
            and writer.file_format not in cls._file_formats
        ):
            cls._file_formats[writer.file_format.lower()] = writer
        return writer

    @classmethod
    def select_writer(cls, output_format: str) -> Type[SyngraphWriter]:
        if output_format not in cls._file_formats:
            logger.error(
                f"{output_format} is not a valid format. "
                f"Available formats are {list(cls._file_formats.keys())}"
            )
            raise UnavailableWriter
        return cls._file_formats[output_format]

    @classmethod
    def list_writers(cls):
        return list(cls._file_formats.keys())


@SyngraphWriterFactory.register_writer
class JsonWriter(SyngraphWriter):
    """Writer to generate a Json file of the routes"""

    file_format = "json"

    def write_file(
        self, syngraphs: list, out_data_model: str, file_name: Optional[str]
    ):
        routes, _ = facade(
            functionality="translate",
            routes=syngraphs,
            input_format="syngraph",
            output_format="noc",
            out_data_model=out_data_model,
        )
        if file_name is None:
            file_name = syngraphs[0].name
        file_name = "".join([file_name, ".", self.file_format])
        lio.write_json(routes, file_name)


@SyngraphWriterFactory.register_writer
class CsvWriter(SyngraphWriter):
    """Writer to generate a csv file of the routes"""

    file_format = "csv"

    def write_file(
        self, syngraphs: list, out_data_model: str, file_name: Optional[str]
    ):
        routes, _ = facade(
            functionality="translate",
            routes=syngraphs,
            input_format="syngraph",
            output_format="noc",
            out_data_model=out_data_model,
        )
        if file_name is None:
            file_name = syngraphs[0].name
        file_name = "".join([file_name, ".", self.file_format])
        lio.dict_list_to_csv(routes, file_name)


@SyngraphWriterFactory.register_writer
class PngWriter(SyngraphWriter):
    """Writer to generate png files of the routes"""

    file_format = "png"

    def write_file(
        self, syngraphs: list, out_data_model: str, file_name: Optional[str]
    ):
        facade(
            functionality="translate",
            routes=syngraphs,
            input_format="syngraph",
            output_format="pydot_visualization",
            out_data_model=out_data_model,
        )


@SyngraphWriterFactory.register_writer
class GraphMLWriter(SyngraphWriter):
    """Writer to generate graphml files of the routes"""

    file_format = "graphml"

    def write_file(
        self, syngraphs: list, out_data_model: str, file_name: Optional[str]
    ):
        nx_routes, _ = facade(
            functionality="translate",
            routes=syngraphs,
            input_format="syngraph",
            output_format="networkx",
            out_data_model=out_data_model,
        )
        for nx_route in nx_routes:
            for n, data in nx_route.nodes.items():
                r_id = data["name"]
                del nx_route.nodes[n]["properties"]

            for n1, n2, d in nx_route.edges(data=True):
                del d["label"]
            fname = ".".join([r_id, "graphml"])
            lio.write_nx_to_graphml(nx_route, fname)


def write_syngraph(
    syngraphs: List[
        Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
    ],
    out_data_model: str,
    output_format: str,
    file_name: Optional[str] = None,
) -> None:
    """
    To write a list of SynGraph instances to a file

    Parameters:
    -------------
    syngraphs: List[Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]]
        The input list of SynGraph objects
    out_data_model: str
        The data model to be used for the output graphs
    output_format: str
        The format of the output file
    file_name: Optional[str]
        The name of the output file (default None)
    """

    writer = SyngraphWriterFactory.select_writer(output_format=output_format)
    return writer().write_file(
        syngraphs=syngraphs, out_data_model=out_data_model, file_name=file_name
    )
