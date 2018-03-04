# The basics

Your help is appreciated and welcome!

The _master_ branch is meant to hold the release code. At any time is should be 
identical to the code available on PyPI. 

PR's will be pushed on the _development_ branch if the actual package code is changed. When the time comes this branch
will be merged to the _master_ branch and a new release will be issued.

PR's that deal with documentation, and other addiacent files (README for example) can be pushed to the _master_ branch.

When submitting PR's please take into account:
* the project's gloals
* PEP8 and the style guide bellow

# Testing
Travis CI is enabled for this project. It is really helpfull for quality assurance
and Python 2 and 3 compatibility check. It is advside to run the tests on both
Python 3 and 2 before pushing the PR (Python 3.6 64 bit and Python 2.7 32 bit would be 
a very good combination to test).

# Style guide

Please follow the following style guide in your new code:

* use numpy style docstrings [(see example)](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html)

* chained function calls 
    
    ```python
    name = new_ch['short_name'].decode('latin-1').strip(' \r\n\t\0')
    
    # can be written as
    
    name = (
        new_ch['short_name']
        .decode('latin-1')
        .strip(' \r\n\t\0')
    )
    ```
    
* sets, dicts, tuples or lists (use trailing comma for last element)
    
    ```python
    __all__ = [
        'Channel',
        'ChannelConversion',
        'ChannelDependency',
        'ChannelExtension',
        'ChannelGroup',
        'DataBlock',
        'DataGroup',
        'FileIdentificationBlock',
        'HeaderBlock',
        'ProgramBlock',
        'SampleReduction',
        'TextBlock',
        'TriggerBlock',
    ]
    
    INT_TYPES = {
        DATA_TYPE_UNSIGNED,
        DATA_TYPE_SIGNED,
        DATA_TYPE_UNSIGNED_INTEL,
        DATA_TYPE_UNSIGNED_MOTOROLA,
        DATA_TYPE_SIGNED_INTEL,
        DATA_TYPE_SIGNED_MOTOROLA,
    }
    
    mydict = {
        'key1': val1,
        'key2': val2,
        'key3': val3,
    }

    ```
    
* long strings

    ```python
    message = (
        'Multiple occurances for channel "{}". '
        'Using first occurance from data group {}. '
        'Provide both "group" and "index" arguments'
        ' to select another data group'
    )

    ```
    
* list comprehension

    ```python
    compacted_signals = [
        {'signal': sig}
        for sig in simple_signals
        if sig.samples.dtype.kind in 'ui'
    ]

    ```
    
* prefer "()" over line continuation "\\"
    
    
