"""
asammdf

"""

from pathlib import Path

from numpy import get_include
from setuptools import Extension, find_packages, setup

PROJECT_PATH = Path(__file__).parent


with (PROJECT_PATH / "requirements.txt").open() as f:
    install_requires = [l.strip() for l in f.readlines()]


def _get_version():
    with PROJECT_PATH.joinpath("src", "asammdf", "version.py").open() as f:
        line = next(line for line in f if line.startswith("__version__"))

    version = line.partition("=")[2].strip()[1:-1]

    return version


def _get_long_description():
    description = PROJECT_PATH.joinpath("README.md").read_text(encoding="utf-8")

    return description


def _get_ext_modules():
    modules = [
        Extension(
            "asammdf.blocks.cutils",
            ["src/asammdf/blocks/cutils.c"],
            include_dirs=[get_include()],
            extra_compile_args=["-std=c99"],
        )
    ]

    return modules


setup(
    name="asammdf",
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=_get_version(),
    description="ASAM MDF measurement data file parser",
    long_description=_get_long_description(),
    long_description_content_type=r"text/markdown",
    # The project's main homepage.
    url="https://github.com/danielhrisca/asammdf",
    # Author details
    author="Daniel Hrisca",
    author_email="daniel.hrisca@gmail.com",
    # Choose your license
    license="LGPLv3+",
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 5 - Production/Stable",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    # Supported python versions
    python_requires=">=3.8",
    # What does your project relate to?
    keywords="read reader edit editor parse parser asam mdf measurement",
    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages("src"),
    package_dir={"": "src"},
    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],
    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=install_requires,
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        "decode": ["faust-cchardet==2.1.19", "chardet"],
        "export": [
            "fastparquet",
            "h5py",
            "hdf5storage>=0.1.19",
            "python-snappy",
        ],
        "export_matlab_v5": "scipy",
        "gui": [
            "lxml>=4.9.2",
            "natsort",
            "psutil",
            "PySide6==6.6.0",
            "pyqtgraph==0.13.3",
            "pyqtlet2==0.9.3",
            "packaging",
            "QtPy==2.3.1",
        ],
        "encryption": ["cryptography", "keyring"],
        "symbolic_math": "sympy",
        "filesystem": "fsspec",
    },
    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={"asammdf.gui.ui": ["*.ui"]},
    include_package_data=True,
    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    #    data_files=[('my_data', ['data/data_file'])],
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={"console_scripts": ["asammdf = asammdf.gui.asammdfgui:main [gui,export,decode]"]},
    ext_modules=_get_ext_modules(),
)
