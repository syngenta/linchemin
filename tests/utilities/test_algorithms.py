from linchemin import utilities


def test_hash_is_deterministic():
    text = "the house is red"
    computed_value = utilities.create_hash(text)
    expected_value = 102997808994593549738162937557500919986
    assert computed_value == expected_value


def test_camel_to_snake():
    assert utilities.camel_to_snake("JavaScript") == "java_script"
    assert utilities.camel_to_snake("Foo-Bar") == "foo-_bar"
    assert utilities.camel_to_snake("foo_bar") == "foo_bar"
    assert utilities.camel_to_snake("fooBAR") == "foo_b_a_r"
    assert utilities.camel_to_snake("Foo_bar") == "foo_bar"


def test_list_of_dict_groupby():
    print("\n\nMISSING: test_utilities.test_list_of_dict_groupby")  #  TODO
