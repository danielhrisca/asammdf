# Contributing

## The basics

Your help is appreciated and welcome!

The _master_ branch is meant to hold the release code. At any time this should be
identical to the code available on PyPI.

PR's will be pushed on the _development_ branch if the actual package code is changed. When the time comes this branch
will be merged to the _master_ branch and a new release will be issued.

PR's that deal with documentation, and other adjacent files (README for example) can be pushed to the _master_ branch.

When submitting PR's please take into account:

- the project's goals
- PEP8 and the style guide below

## Developer Instructions

Clone the repository with submodules, make sure you have a C compiler installed, then install the development dependencies (we recommend using a virtual environment):

```bash
python -m venv .venv  # create virtual environment in .venv
source .venv/bin/activate  # activate virtual environment (POSIX)
.venv\Scripts\activate.bat  # activate virtual environment (Windows)
pip install --requirement requirements.txt  # install development dependencies (includes asammdf in editable mode)
```

Now you can start developing. If you are using VSCode, the virtual environment should be detected automatically. If not, open the command panel `Ctrl + Shift + P` and search for `Python: Select Interpreter`.

## Testing

You can use tox to run tests locally. Example for the unit tests with Python version 3.10:

```console
tox -e py310
```

Otherwise, you can just push and the tests will be run by GitHub Actions.

## Style guide

Just run [_black_](https://black.readthedocs.io) on modified files before sending the PR. There is no need to reinvent the wheel here!

**Tip**: install Git hooks using pre-commit `pre-commit install --install-hooks`
