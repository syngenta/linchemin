# Contributing

## Contribution Terms and License

The code and documentation of LinChemIn is contained in this repository. To contribute
to this project or any of the elements of LinChemIn we recommend you start by reading this
contributing guide.

## Contributing to LinChemIn codebase

If you would like to contribute to the package, we recommend the following development setup.

1. Create a copy of the [repository](https://github.com/syngenta/linchemin) via the "_Fork_" button.

2. Clone the linchemin repository:

    ```sh
    git clone git@github.com:${GH_ACCOUNT_OR_ORG}/linchemin.git
    ```

3. Add remote linchemin repo as an "upstream" in your local repo, so you can check/update remote changes.

   ```sh
   git remote add upstream git@github.com:syngenta/linchemin.git
   ```

4. Create a dedicated branch:

    ```sh
    cd linchemin
    git checkout -b a-super-nice-feature-we-all-need
    ```

5. Create and activate a dedicated conda environment (any other virtual environment management would work):

    ```sh
    conda env create linchemin
    conda activate linchemin
    ```

6. Install linchemin in editable mode:

    ```sh
    pip install -e .[dev]
    ```

7. Implement your changes and once you are ready run the tests:

    ```sh
    # this can take quite long
    cd linchemin/tests
    python -m pytest
    ```

   And add style checks (be aware that running isort might change your files!):
   ```sh
    cd linchemin
    # sorting the imports
    python -m isort src/linchemin
    # checking flake8
    python -m flake8 --ignore E501 src/linchemin
    ```

8. Once the tests and checks passes, but most importantly you are happy with the implemented feature, commit your changes.

    ```sh
    # add the changes
    git add
    # commit them
    git commit -s -m "feat: implementing super nice feature." -m "A feature we all need."
    # check upstream changes
    git fetch upstream
    git rebase upstream/main
    # push changes to your fork
    git push -u origin a-super-nice-feature-we-all-need
    ```

9. From your fork, open a pull request via the "_Contribute_" button, the maintainers will be happy to review it.
