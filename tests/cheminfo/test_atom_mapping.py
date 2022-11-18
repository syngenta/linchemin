from linchemin.cheminfo.atom_mapping import pipeline_atom_mapping, perform_atom_mapping, get_available_mappers
import pytest


def test_basic_factory(capfd):
    perform_atom_mapping('namerxn', [])
    out, err = capfd.readouterr()
    assert 'NameRxn' in out
    with pytest.raises(KeyError) as ke:
        perform_atom_mapping('unavailable_mapper', [])
    assert "KeyError" in str(ke.type)


def test_basic_pipeline(capfd):
    pipeline_atom_mapping()
    out, err = capfd.readouterr()
    assert 'NameRxn' in out
    assert 'Chematica' in out
    assert 'RxnMapper' in out


def test_helper_function():
    h = get_available_mappers()
    assert h
    assert 'namerxn' in h
