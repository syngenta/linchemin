import pytest

from linchemin.cheminfo import depiction


def test_color_map_search_by_name():
    color_map = depiction.ColorMap()
    results = color_map.search_by_name(name="aliceblue")
    assert len(results) == 1
    assert results[0]._asdict() == {
        "name": "aliceblue",
        "red": 240,
        "green": 248,
        "blue": 255,
        "hex_string": "#F0F8FF",
        "color_blind": None,
    }
    results = color_map.search_by_name(name="aliceblueS")
    assert len(results) == 0


def test_color_map_search_by_rgb_tuple_identical():
    color_map = depiction.ColorMap()
    results = color_map.search_by_rgb(rgb=(240, 248, 255))
    assert len(results) == 1
    assert results[0]._asdict() == {
        "name": "aliceblue",
        "red": 240,
        "green": 248,
        "blue": 255,
        "hex_string": "#F0F8FF",
        "color_blind": None,
    }
    results = color_map.search_by_rgb(rgb=(240, 248, 256))
    assert len(results) == 0


def xtest_color_map_search_by_rgb_tuple_similar():
    color_map = depiction.ColorMap()
    result = color_map.closest(rgb=(240, 248, 255))
    print(result)

    assert result._asdict() == {
        "name": "aliceblue",
        "red": 240,
        "green": 248,
        "blue": 255,
        "hex_string": "#F0F8FF",
        "color_blind": None,
    }

    # results = color_map.closest(rgb=(240, 249, 255))
    # print(results)


def test_color_map_search_by_hex():
    color_map = depiction.ColorMap()
    results = color_map.search_by_hex_string(hex_string="#F0F8FF")
    assert len(results) == 1
    assert results[0]._asdict() == {
        "name": "aliceblue",
        "red": 240,
        "green": 248,
        "blue": 255,
        "hex_string": "#F0F8FF",
        "color_blind": None,
    }
    results = color_map.search_by_hex_string(hex_string="#F0F8FF_")
    assert len(results) == 0


def test_color_map_search_by_color_blind_attribute():
    color_map = depiction.ColorMap()
    results = color_map.search_by_color_blind(color_blind_type="okabe_ito")
    assert len(results) > 0
