import csv
import json
import pickle
from pathlib import Path
from typing import List, Union

import networkx as nx
import pandas as pd


def get_file_path(
    file_path: Union[Path, str], mode: str, overwrite: bool = True
) -> Path:
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if file_path.is_dir():
        raise IsADirectoryError
    if mode in {"r", "rb"} and not file_path.exists():
        raise FileNotFoundError
    if mode in {"w", "wb"} and file_path.exists() and not overwrite:
        raise FileExistsError
    return file_path


def write_json(data, file_path: Union[Path, str], indent=4, overwrite=True):
    file_path = get_file_path(file_path=file_path, mode="w", overwrite=overwrite)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=indent)


def read_json(file_path: Union[Path, str]):
    file_path = get_file_path(file_path, "r")
    with open(file_path) as f:
        data = json.load(f)
    return data


def dict_list_to_csv(
    data: List[dict],
    file_path: Union[Path, str],
    delimiter: str = ",",
    newline: str = "",
    encoding: str = "utf-8",
    overwrite=True,
):
    file_path = get_file_path(file_path=file_path, mode="w", overwrite=overwrite)
    keys = list(data[0].keys())
    with open(file_path, "w", newline=newline, encoding=encoding) as output_file:
        dict_writer = csv.DictWriter(output_file, keys, delimiter=delimiter)
        dict_writer.writeheader()
        dict_writer.writerows(data)
    return


def csv_to_dict_list(file_path: Union[Path, str], delimiter: str = ","):
    file_path = get_file_path(file_path=file_path, mode="r")
    with open(file_path) as file:
        content = csv.DictReader(file, delimiter=delimiter)
        return [dict(row) for row in content]


def write_pickle(data, file_path: Union[Path, str], overwrite=True):
    file_path = get_file_path(file_path=file_path, mode="w", overwrite=overwrite)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def read_pickle(file_path: Union[Path, str]):
    file_path = get_file_path(file_path=file_path, mode="r")
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def dataframe_to_csv(df: pd.DataFrame, file_path: Union[Path, str], overwrite=True):
    file_path = get_file_path(file_path=file_path, mode="w", overwrite=overwrite)
    df.to_csv(file_path, index=False)


def write_rdkit_depict(data: bin, file_path: Union[Path, str]):
    file_path = get_file_path(file_path=file_path, mode="w")
    with open(file_path, "wb+") as file:
        file.write(data)


def write_nx_to_graphml(
    nx_graph: nx.classes.digraph.DiGraph, file_path: Union[Path, str], overwrite=True
):
    file_path = get_file_path(file_path=file_path, mode="w", overwrite=overwrite)
    nx.write_graphml(nx_graph, file_path)
