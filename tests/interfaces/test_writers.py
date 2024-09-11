from unittest.mock import patch

import networkx as nx

from linchemin.interfaces.writers import (
    CsvWriter,
    GraphMLWriter,
    JsonWriter,
    PngWriter,
    SyngraphWriter,
    SyngraphWriterFactory,
    write_syngraph,
)


def test_writer_factory():
    class FakeWriter(SyngraphWriter):
        file_format = "fake_format"

        def write_file(
            self, syngraphs: list, out_data_model: str, file_name: str
        ) -> None:
            """Mocked abstract method"""
            pass

    SyngraphWriterFactory.register_writer(FakeWriter)
    assert "fake_format" in SyngraphWriterFactory.list_writers()

    retrieved_class = SyngraphWriterFactory.select_writer("fake_format")
    assert retrieved_class is FakeWriter


@patch("linchemin.interfaces.writers.lio.write_json")
@patch("linchemin.interfaces.writers.facade")
def test_json_writer(mock_facade, mock_lio, mpr_syngraph_instance):
    out_data_model = "bipartite"
    syngraphs = [mpr_syngraph_instance] * 2
    mock_facade.return_value = (syngraphs, {})
    writer = JsonWriter()
    writer.write_file(
        syngraphs=syngraphs, out_data_model=out_data_model, file_name="test_file"
    )
    mock_facade.assert_called_with(
        functionality="translate",
        routes=syngraphs,
        input_format="syngraph",
        output_format="noc",
        out_data_model=out_data_model,
    )
    mock_lio.assert_called_with(syngraphs, "test_file.json")


@patch("linchemin.interfaces.writers.lio.dict_list_to_csv")
@patch("linchemin.interfaces.writers.facade")
def test_csv_writer(mock_facade, mock_lio, mpr_syngraph_instance):
    out_data_model = "bipartite"
    syngraphs = [mpr_syngraph_instance] * 2
    mock_facade.return_value = (syngraphs, {})
    writer = CsvWriter()
    writer.write_file(
        syngraphs=syngraphs, out_data_model=out_data_model, file_name="test_file_csv"
    )
    mock_facade.assert_called_with(
        functionality="translate",
        routes=syngraphs,
        input_format="syngraph",
        output_format="noc",
        out_data_model=out_data_model,
    )
    mock_lio.assert_called_with(syngraphs, "test_file_csv.csv")


@patch("linchemin.interfaces.writers.facade")
def test_png_writer(mock_facade, mpr_syngraph_instance):
    out_data_model = "bipartite"
    syngraphs = [mpr_syngraph_instance] * 2
    mock_facade.return_value = (syngraphs, {})
    writer = PngWriter()
    writer.write_file(
        syngraphs=syngraphs, out_data_model=out_data_model, file_name="test_file"
    )
    mock_facade.assert_called_with(
        functionality="translate",
        routes=syngraphs,
        input_format="syngraph",
        output_format="pydot_visualization",
        out_data_model=out_data_model,
    )


@patch("linchemin.interfaces.writers.lio.write_nx_to_graphml")
@patch("linchemin.interfaces.writers.facade")
def test_graphml_writer(mock_facade, mock_lio, mpr_syngraph_instance):
    out_data_model = "bipartite"
    syngraphs = [mpr_syngraph_instance] * 2
    # Creating test networkx graph to be returned by the translation facade
    g1 = nx.DiGraph()
    g1.add_nodes_from([1, 2, 3], name="route_id_1", properties={"prop1": "value1"})
    g1.add_edges_from([(1, 2), (2, 3)], label="edge_label")
    g2 = nx.DiGraph()
    g2.add_nodes_from([4, 5, 6], name="route_id_2", properties={"prop2": "value1"})
    g2.add_edges_from([(4, 5), (5, 6)], label="edge_label")

    out_routes = [g1, g2]
    mock_facade.return_value = (out_routes, {})
    writer = GraphMLWriter()
    writer.write_file(
        syngraphs=syngraphs, out_data_model=out_data_model, file_name="test_file_g"
    )
    mock_facade.assert_called_with(
        functionality="translate",
        routes=syngraphs,
        input_format="syngraph",
        output_format="networkx",
        out_data_model=out_data_model,
    )
    mock_lio.assert_called()


@patch("linchemin.interfaces.writers.CsvWriter.write_file")
def test_write_syngraphs(mock_writer, bp_syngraph_instance):
    syngraphs = [bp_syngraph_instance] * 2
    write_syngraph(
        syngraphs=syngraphs,
        out_data_model="monopartite_reactions",
        output_format="csv",
        file_name="test_csv",
    )
    mock_writer.assert_called()
