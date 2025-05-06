from unittest.mock import patch

import pytest

from linchemin.interfaces.workflows import (
    AtomMappingStep,
    ClusteringAndDistanceMatrixStep,
    ClusteringStep,
    DescriptorsStep,
    DistanceMatrixStep,
    Executor,
    ExtractingReactionsStep,
    Finisher,
    InvalidCasp,
    MergingStep,
    NoValidRoute,
    TranslationStep,
    WorkflowOutput,
    WorkflowStarter,
    get_workflow_options,
    process_routes,
)

routes_csv_filename = "routes.csv"


@patch("linchemin.interfaces.workflows.lio.read_json")
@patch("linchemin.interfaces.workflows.facade")
def test_translate_step(mock_facade, mock_read, az_path, ibm2_path, reaxys_path):
    step = TranslationStep()
    input_dict = {str(az_path): "az", str(reaxys_path): "reaxys"}
    params = {
        "input": input_dict,
        "out_data_model": "bipartite",
        "parallelization": False,
        "n_cpu": 4,
    }
    routes = [{}]
    mock_read.return_value = routes
    mock_facade.return_value = ("translated_routes", {"meta1": "value1"})
    out = WorkflowOutput()
    out = step.perform_step(params=params, output=out)
    mock_facade.assert_called_with(
        functionality="translate",
        input_format="reaxys",
        routes=routes,
        out_data_model=params["out_data_model"],
        parallelization=params["parallelization"],
        n_cpu=params["n_cpu"],
    )
    assert out.routes_list
    input_dict2 = {az_path: "wrong_casp"}
    params2 = {
        "input": input_dict2,
        "out_data_model": "bipartite",
        "parallelization": False,
        "n_cpu": 4,
    }
    # error raised for invalid casp
    with pytest.raises(InvalidCasp):
        step.perform_step(params=params2, output=out)

    # error raised for no valid routes
    mock_facade.return_value = ([], {})
    with pytest.raises(NoValidRoute):
        step.perform_step(params=params, output=out)


@patch("linchemin.IO.io.dataframe_to_csv")
@patch("linchemin.interfaces.workflows.facade")
def test_workflow_metric(mock_facade, mock_dataframe, mpr_syngraph_instance):
    step = DescriptorsStep()
    routes = [mpr_syngraph_instance] * 2
    mock_facade.return_value = ("output", {"meta1": "value1"})
    params = {"descriptors": ["nr_steps", "nr_branches"]}
    out = WorkflowOutput(routes_list=routes)
    out = step.perform_step(params=params, output=out)
    mock_facade.assert_called_with(
        functionality="routes_descriptors",
        routes=routes,
        descriptors=params["descriptors"],
    )
    mock_dataframe.assert_called()
    assert out.descriptors


@patch("linchemin.IO.io.write_json")
@patch("linchemin.interfaces.workflows.converter")
@patch("linchemin.interfaces.workflows.merge_syngraph")
def test_merging(mock_merge, mock_coverter, mock_writer, mpr_syngraph_instance):
    params = {"out_data_model": "monopartite_reactions", "output_format": "json"}
    routes = [mpr_syngraph_instance] * 2
    step = MergingStep()
    out = WorkflowOutput(routes_list=routes)
    out = step.perform_step(params=params, output=out)
    mock_merge.assert_called_once()
    mock_coverter.assert_called_once()
    mock_writer.assert_called()
    assert out.tree


@patch("linchemin.IO.io.dataframe_to_csv")
@patch("linchemin.interfaces.workflows.facade")
def test_workflow_cluster(mock_facade, mock_dataframe, mpr_syngraph_instance):
    params = {
        "clustering_method": "agglomerative_cluster",
        "ged_method": "nx_ged",
        "ged_params": None,
        "save_dist_matrix": True,
        "compute_metrics": True,
        "parallelization": False,
        "n_cpu": 4,
    }
    routes = [mpr_syngraph_instance] * 2
    mock_facade.return_value = (("clustering", "metrics"), {})
    step = ClusteringStep()
    out = WorkflowOutput(routes_list=routes)
    out = step.perform_step(params=params, output=out)
    mock_facade.assert_called_with(
        functionality="clustering",
        routes=routes,
        clustering_method=params["clustering_method"],
        ged_method=params["ged_method"],
        ged_params=params["ged_params"],
        compute_metrics=True,
        parallelization=params["parallelization"],
        n_cpu=params["n_cpu"],
    )
    mock_dataframe.assert_called()
    assert out.clustering


@patch("linchemin.IO.io.dataframe_to_csv")
@patch("linchemin.interfaces.workflows.facade")
def test_clustering_distance(mock_facade, mock_to_csv, mpr_syngraph_instance):
    routes = [mpr_syngraph_instance] * 2
    mock_facade.return_value = (
        (("clustering", "score", "matrix"), "metrics"),
        {"meta1": "value1"},
    )
    workflow_step = ClusteringAndDistanceMatrixStep()
    params = {
        "clustering_method": "agglomerative_cluster",
        "ged_method": "nx_ged",
        "ged_params": None,
        "save_dist_matrix": True,
        "compute_metrics": True,
        "parallelization": False,
        "n_cpu": 4,
    }
    out = WorkflowOutput(routes_list=routes)
    out = workflow_step.perform_step(params=params, output=out)
    mock_facade.assert_called_with(
        functionality="clustering",
        routes=routes,
        clustering_method="agglomerative_cluster",
        ged_method="nx_ged",
        ged_params=None,
        save_dist_matrix=True,
        compute_metrics=True,
        parallelization=False,
        n_cpu=4,
    )
    mock_to_csv.assert_called()
    assert out.clustering
    assert out.clustered_descriptors
    assert out.distance_matrix


@patch("linchemin.IO.io.dataframe_to_csv")
@patch("linchemin.interfaces.workflows.facade")
def test_distance_matrix(mock_facade, mock_to_csv, mpr_syngraph_instance):
    routes = [mpr_syngraph_instance] * 2
    mock_facade.return_value = (
        "matrix",
        {"meta1": "value1"},
    )
    workflow_step = DistanceMatrixStep()
    params = {
        "clustering_method": "agglomerative_cluster",
        "ged_method": "nx_ged",
        "ged_params": None,
        "save_dist_matrix": True,
        "compute_metrics": True,
        "parallelization": False,
        "n_cpu": 4,
    }
    out = WorkflowOutput(routes_list=routes)
    out = workflow_step.perform_step(params=params, output=out)
    mock_facade.assert_called_with(
        functionality="distance_matrix",
        routes=routes,
        ged_method=params["ged_method"],
        ged_params=params["ged_params"],
        parallelization=params["parallelization"],
        n_cpu=params["n_cpu"],
    )
    mock_to_csv.assert_called()

    assert out.distance_matrix


@patch("linchemin.IO.io.write_json")
@patch("linchemin.interfaces.workflows.facade")
def test_reaction_strings_extraction(
    mock_facade,
    mock_json,
    mpr_syngraph_instance,
):
    step = ExtractingReactionsStep()
    params = {}
    routes = [mpr_syngraph_instance]
    mock_facade.return_value = ("reactions", {})
    out = WorkflowOutput(routes_list=routes)
    out = step.perform_step(params=params, output=out)
    mock_facade.assert_called_with(
        functionality="extract_reactions_strings", routes=out.routes_list
    )
    mock_json.assert_called()
    assert out.reaction_strings


def test_helper(capfd):
    assert isinstance(get_workflow_options(), dict)
    assert "functionalities" in get_workflow_options()
    get_workflow_options(verbose=True)
    out, _ = capfd.readouterr()
    assert "input_dict" in out


@patch("linchemin.IO.io.write_json")
@patch("linchemin.interfaces.workflows.MergingStep.perform_step")
@patch("linchemin.interfaces.workflows.ClusteringAndDistanceMatrixStep.perform_step")
@patch("linchemin.interfaces.workflows.DescriptorsStep.perform_step")
@patch("linchemin.interfaces.workflows.TranslationStep.perform_step")
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


@patch("linchemin.interfaces.workflows.converter")
@patch("linchemin.interfaces.workflows.facade")
def test_atom_mapping(
    mock_facade, mock_converter, mpr_syngraph_instance, bp_syngraph_instance
):
    params = {"mapper": "rxnmapper", "out_data_model": "monopartite_molecules"}
    step = AtomMappingStep()
    routes = [mpr_syngraph_instance] * 2
    routes_out = [bp_syngraph_instance] * 2
    mock_facade.return_value = (routes_out, {})
    out = WorkflowOutput(routes_list=routes)
    out = step.perform_step(params=params, output=out)
    mock_facade.assert_called_with(
        functionality="atom_mapping",
        routes=routes,
        mapper=params["mapper"],
    )
    mock_converter.assert_called_with(routes_out[-1], params["out_data_model"])
    assert out.routes_list != routes


@patch("linchemin.interfaces.workflows.Executor.execute")
@patch("linchemin.interfaces.workflows.AtomMappingStep.perform_step")
@patch("linchemin.interfaces.workflows.TranslationStep.perform_step")
def test_workflow_starter(
    mock_translate, mock_mapping, mock_executor, az_path, reaxys_path
):
    input_dict = {str(az_path): "az", str(reaxys_path): "reaxys"}
    # mapping is not requested
    params = {
        "input": input_dict,
        "out_data_model": "bipartite",
        "parallelization": False,
        "n_cpu": 4,
        "mapping": False,
    }
    out = WorkflowOutput()
    starter = WorkflowStarter()
    starter.execute(params=params, output=out)
    mock_translate.assert_called()
    mock_executor.assert_called()

    # mapping is requested
    params2 = {
        "input": input_dict,
        "out_data_model": "bipartite",
        "parallelization": False,
        "n_cpu": 4,
        "mapping": True,
    }
    starter = WorkflowStarter()
    starter.execute(params=params2, output=out)
    mock_translate.assert_called()
    mock_mapping.assert_called()
    mock_executor.assert_called()

    # None is returned if no valid route is identified
    mock_translate.side_effect = NoValidRoute
    out = starter.execute(params=params, output=out)
    assert out is None


@patch("linchemin.interfaces.workflows.DescriptorsStep.perform_step")
@patch("linchemin.interfaces.workflows.MergingStep.perform_step")
@patch("linchemin.interfaces.workflows.Finisher.execute")
def test_workflow_executor(mock_finisher, mock_merging, mock_descriptors):
    params = {"functionalities": None}
    out = WorkflowOutput()
    executor = Executor()
    executor.execute(params=params, output=out)
    mock_finisher.assert_called()

    params2 = {"functionalities": ["compute_descriptors", "merging"]}
    executor.execute(params=params2, output=out)
    mock_descriptors.assert_called()
    mock_merging.assert_called()
    mock_finisher.assert_called()

    funcs = executor.get_workflow_functions()
    assert "distance_matrix" in funcs


@patch("linchemin.interfaces.workflows.write_syngraph")
def test_finisher(mock_writer, mpr_syngraph_instance):
    routes = [mpr_syngraph_instance] * 2
    out = WorkflowOutput(routes_list=routes)
    finisher = Finisher()
    params = {"out_data_model": "monopartite_reactions", "output_format": "json"}
    finisher.execute(params=params, output=out)
    mock_writer.assert_called_with(
        syngraphs=routes,
        out_data_model=params["out_data_model"],
        output_format=params["output_format"],
        file_name="routes",
    )


def test_get_workflow_options():
    d = get_workflow_options()
    assert "parallelization" in d
    assert "ged_method" in d
    assert "input_dict" in d
