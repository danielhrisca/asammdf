# The basics

Your help is appreciated and welcome!

The _master_ branch is meant to hold the release code. At any time this should be 
identical to the code available on PyPI. 

PR's will be pushed on the _development_ branch if the actual package code is changed. When the time comes this branch
will be merged to the _master_ branch and a new release will be issued.

PR's that deal with documentation, and other adjacent files (README for example) can be pushed to the _master_ branch.

When submitting PR's please take into account:
* the project's gloals
* PEP8 and the style guide below

# Testing
Travis CI is enabled for this project. It is really helpful for quality assurance
and Python 2 and 3 compatibility check. It is advised to run the tests on both
Python 3 and 2 before pushing the PR (Python 3.6 64 bit and Python 2.7 32 bit would be 
a very good combination to test).

# Style guide

Just run [*black*](https://black.readthedocs.io) on modified files before sending the PR. There is no need to reinvent the wheel here!

**Tip**: Add a pre-commit hook with `pip install pre-commit && pre-commit install`
