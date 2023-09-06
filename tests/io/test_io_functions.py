import pytest

from linchemin.IO import io as lio


def test_success_file_exists_while_writing_and_overwriting():
    pass


def test_success_file_not_exists_while_writing_and_overwriting():
    pass


def test_success_file_not_exists_while_writing_and_not_overwriting():
    pass


def test_fail_file_does_not_exists_while_reading():
    with pytest.raises(FileNotFoundError):
        lio.read_json(file_path="not_exists.json")


def test_fail_file_is_a_directory_while_reading():
    with pytest.raises(IsADirectoryError):
        lio.read_json(file_path="../")


def test_fail_file_exists_while_writing_not_overwriting():
    pass


def test_fail_file_is_a_directory_while_writing():
    pass
