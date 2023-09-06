import unittest.mock

import pytest

from linchemin.interfaces.facade import facade
from linchemin.interfaces.workflows import get_workflow_options, process_routes


@unittest.mock.patch("linchemin.IO.io.write_json")
def test_workflow_basic(mock_os, az_path, ibm2_path):
    az_file = str(az_path)
    ibm2_file = str(ibm2_path)
    input_dict = {az_file: "az"}
    out = process_routes(input_dict)
    assert out

    routes, meta = facade(
        "translate", "syngraph", out.routes_list, "noc", out_data_model="bipartite"
    )
    mock_os.assert_called_with(routes, "routes.json")

    input_dict_multicasp = {az_file: "az", ibm2_file: "ibmrxn"}
    out = process_routes(input_dict_multicasp)
    assert out

    routes, meta = facade(
        "translate", "syngraph", out.routes_list, "noc", out_data_model="bipartite"
    )
    mock_os.assert_called_with(routes, "routes.json")

    # error raised for invalid casp
    with pytest.raises(KeyError) as ke:
        input_dict2 = {az_path: "wrong_casp"}
        process_routes(input_dict2, output_format="json")
    assert "KeyError" in str(ke.type)

    # error raised for invalid output format
    with pytest.raises(KeyError) as ke:
        input_dict = {az_file: "az"}
        process_routes(input_dict, output_format="jpg")
    assert "KeyError" in str(ke.type)

    # error raised for invalid functionality
    with pytest.raises(KeyError) as ke:
        process_routes(input_dict, output_format="csv", functionalities=["func"])
    assert "KeyError" in str(ke.type)


@unittest.mock.patch("linchemin.IO.io.dict_list_to_csv")
@unittest.mock.patch("linchemin.IO.io.dataframe_to_csv")
def test_workflow_metric(mock_dataframe, mock_csv, ibm2_path):
    path = str(ibm2_path)
    input_dict = {path: "ibmrxn"}
    out = process_routes(
        input_dict,
        output_format="csv",
        functionalities=["compute_descriptors"],
        descriptors=["branching_factor", "nr_steps"],
        parallelization=True,
        n_cpu=16,
    )
    routes, meta = facade(
        "translate", "syngraph", out.routes_list, "noc", out_data_model="bipartite"
    )
    mock_csv.assert_called_with(routes, "routes.csv")
    mock_dataframe.assert_called_with(out.descriptors, "descriptors.csv")


@unittest.mock.patch("linchemin.interfaces.workflows.write_syngraph")
def test_merging(mock_writer1, az_path):
    path = str(az_path)
    input_dict = {path: "az"}
    out = process_routes(input_dict, output_format="png", functionalities=["merging"])
    mock_writer1.assert_called_with(out.routes_list, "bipartite", "png", "routes")


@unittest.mock.patch("linchemin.IO.io.dict_list_to_csv")
@unittest.mock.patch("linchemin.IO.io.dataframe_to_csv")
def test_workflow_cluster_dist_matrix(mock_dataframe, mock_csv, mit_path):
    path = str(mit_path)
    input_dict = {path: "askcos"}
    out = process_routes(
        input_dict, output_format="csv", functionalities=["clustering"]
    )
    routes, meta = facade(
        "translate", "syngraph", out.routes_list, "noc", out_data_model="bipartite"
    )
    mock_csv.assert_called_with(routes, "routes.csv")
    mock_dataframe.assert_called_with(out.clustered_descriptors, "cluster_metrics.csv")


@unittest.mock.patch("linchemin.IO.io.write_nx_to_graphml")
def test_workflow_graphml(mock_graphml, mit_path):
    path = str(mit_path)
    input_dict = {path: "askcos"}
    process_routes(input_dict, output_format="graphml")
    mock_graphml.assert_called()


def test_helper(capfd):
    assert type(get_workflow_options()) == dict
    assert "functionalities" in get_workflow_options()
    get_workflow_options(verbose=True)
    out, err = capfd.readouterr()
    assert "input_dict" in out


@unittest.mock.patch("linchemin.IO.io.dict_list_to_csv")
@unittest.mock.patch("linchemin.IO.io.write_json")
def test_reaction_strings_extraction(mock_json, mock_csv, az_path):
    path = str(az_path)
    input_dict = {path: "az"}
    out = process_routes(
        input_dict, output_format="csv", functionalities=["extracting_reactions"]
    )
    routes, meta = facade(
        "translate", "syngraph", out.routes_list, "noc", out_data_model="bipartite"
    )
    mock_csv.assert_called_with(routes, "routes.csv")
    mock_json.assert_called_with(out.reaction_strings, "reaction_strings.json")


@unittest.mock.patch("linchemin.IO.io.dict_list_to_csv")
@unittest.mock.patch("linchemin.IO.io.write_json")
def test_graphml(mock_json, mock_csv, az_path):
    path = str(az_path)
    input_dict = {path: "az"}
    out = process_routes(
        input_dict, output_format="csv", functionalities=["extracting_reactions"]
    )
    routes, meta = facade(
        "translate", "syngraph", out.routes_list, "noc", out_data_model="bipartite"
    )
    mock_csv.assert_called_with(routes, "routes.csv")
    mock_json.assert_called_with(out.reaction_strings, "reaction_strings.json")


@unittest.mock.patch("linchemin.IO.io.write_json")
@unittest.mock.patch("linchemin.interfaces.workflows.MergingStep.perform_step")
@unittest.mock.patch(
    "linchemin.interfaces.workflows.ClusteringAndDistanceMatrixStep.perform_step"
)
@unittest.mock.patch("linchemin.interfaces.workflows.DescriptorsStep.perform_step")
@unittest.mock.patch("linchemin.interfaces.workflows.TranslationStep.perform_step")
def test_full_workflow(
    mock_translate, mock_descriptors, mock_cluster, mock_merging, mock_write, az_path
):
    path = str(az_path)
    input_dict = {path: "az"}
    process_routes(
        input_dict,
        out_data_model="monopartite_reactions",
        functionalities=["compute_descriptors", "clustering_and_d_matrix", "merging"],
    )
    mock_translate.assert_called()
    mock_descriptors.assert_called()
    mock_cluster.assert_called()
    mock_merging.assert_called()
    mock_write.assert_called()


@unittest.mock.patch("linchemin.IO.io.write_json")
@unittest.mock.patch("linchemin.interfaces.workflows.AtomMappingStep.perform_step")
def test_atom_mapping(mock_mapping, mock_write, az_path):
    path = str(az_path)
    input_dict = {path: "az"}
    process_routes(input_dict, out_data_model="monopartite_reactions", mapping=True)
    mock_mapping.assert_called()
    mock_write.assert_called()
