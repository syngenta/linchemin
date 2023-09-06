import linchemin


def test_version():
    version = linchemin.__version__
    assert version is not None


if __name__ == "__main__":
    test_version()
