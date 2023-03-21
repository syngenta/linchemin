
[![PyPI - Version](https://img.shields.io/pypi/v/linchemin)](https://pypi.python.org/pypi/linchemin)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/linchemin)](https://pypi.python.org/pypi/linchemin)
[![Downloads monthly](https://static.pepy.tech/badge/linchemin/month)](https://pepy.tech/project/linchemin)
[![Downloads total](https://static.pepy.tech/badge/linchemin)](https://pepy.tech/project/linchemin)

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/syngenta/linchemin/test_suite.yml?branch=main)
[![Documentation Status](https://readthedocs.org/projects/linchemin/badge/?version=latest)](https://linchemin.readthedocs.io/en/latest/?badge=latest)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Contributions](https://img.shields.io/badge/contributions-welcome-blue)](https://github.com/syngenta/linchemin/blob/main/CONTRIBUTING.md)

[![Powered by RDKit](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAwsPGi+MyC9RAAAAQElEQVQI12NgQABGQUEBMENISUkRLKBsbGwEEhIyBgJFsICLC0iIUdnExcUZwnANQWfApKCK4doRBsKtQFgKAQC5Ww1JEHSEkAAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyMi0wMy0xMVQxNToyNjo0NyswMDowMDzr2J4AAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjItMDMtMTFUMTU6MjY6NDcrMDA6MDBNtmAiAAAAAElFTkSuQmCC)](https://www.rdkit.org/)

![LinChemIn logo](https://raw.githubusercontent.com/syngenta/linchemin/main/docs/source/static/linchemin_logo.png)

## LinChemIn (Linked Chemical Information)

The Linked Chemical Information (LinChemIn) package is a python toolkit that allows chemoinformatics operations on synthetic routes and reaction networks.

## Authors and Maintainers
[Marta Pasquini](mailto:marta.pasquini@syngenta.com)
[Marco Stenta](mailto:marco.stenta@syngenta.com)

## Resources
|resource type|link|
|---|---|
|project page|https://linchemin.github.io|
|repository|https://github.com/syngenta/linchemin   |
|documentation|http://linchemin.readthedocs.io|
|pypi page|https://pypi.org/project/linchemin|

## Install for usage
### 1) pypi
>pip install linchemin
>
### 2) clone & pip
>git clone https://github.com/syngenta/linchemin
>cd linchemin
>pip install .

### 3) pip from git
>pip install git+https://github.com/syngenta/linchemin

## Install for development
>git clone https://github.com/syngenta/linchemin
>cd linchemin
>pip install -e .[dev]

Testing is based on pytest


## Configuration

This package requires some configuration parameters to work,
including some secretes to store access credentials to database and services.
After installation, and before the first usage, the use should run the following command

>linchemin_configure

his command generates the user_home/linchemin directory and places into it two files:
1. settings.toml populated with defaults settings. The user can review and modify these values if necessary.
2. .secrets.toml containing the keys for the necessary secrets. The user must replace the placeholders with the correct values
For more details please refer to the Configuration section of the documentation