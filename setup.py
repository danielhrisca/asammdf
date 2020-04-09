"""
asammdf

"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages, Extension
from distutils.core import setup, Extension

from distutils.command import build_ext


def get_export_symbols(self, ext):
    parts = ext.name.split(".")
    print("parts", parts)
    if parts[-1] == "__init__":
        initfunc_name = "PyInit_" + parts[-2]
    else:
        initfunc_name = "PyInit_" + parts[-1]


build_ext.build_ext.get_export_symbols = get_export_symbols


def get_export_symbols(self, ext):
    parts = ext.name.split(".")
    print("parts", parts)
    if parts[-1] == "__init__":
        initfunc_name = "PyInit_" + parts[-2]
    else:
        initfunc_name = "PyInit_" + parts[-1]


build_ext.build_ext.get_export_symbols = get_export_symbols
# To use a consistent encoding
from codecs import open
from os import path, listdir, walk

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
long_description = open("README.md", "rb").read().decode("utf-8")

with open(path.join("asammdf", "version.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[-1].strip().strip('"')
            break

try:
    try:
        import numpy
    except ImportError:
        print(
            "numpy must be preinstalled to compile the asammdf C extension; falling back to pure python"
        )
    setup(
        name="asammdf",
        # Versions should comply with PEP440.  For a discussion on single-sourcing
        # the version across setup.py and the project code, see
        # https://packaging.python.org/en/latest/single_source_version.html
        version=version,
        description="ASAM MDF measurement data file parser",
        long_description=long_description,
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
            "Development Status :: 4 - Beta",
            # Indicate who your project is intended for
            "Intended Audience :: Developers",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            # Pick your license as you wish (should match "license" above)
            "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
            # Specify the Python versions you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both.
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
        ],
        # What does your project relate to?
        keywords="read reader edit editor parse parser asam mdf measurement",
        # You can just specify the packages manually here if your project is
        # simple. Or you can use find_packages().
        packages=find_packages(exclude=["contrib", "docs", "test"]),
        # Alternatively, if you want to distribute just a my_module.py, uncomment
        # this:
        #   py_modules=["my_module"],
        # List run-time dependencies here.  These will be installed by pip when
        # your project is installed. For an analysis of "install_requires" vs pip's
        # requirements files see:
        # https://packaging.python.org/en/latest/requirements.html
        install_requires=[
            "numpy>=1.16.1",
            "pandas",
            "numexpr",
            "wheel",
            "canmatrix",
            "natsort",
            "lxml",
            "cChardet==2.1.5",
            "lz4",
        ],
        # List additional groups of dependencies here (e.g. development
        # dependencies). You can install these using the following syntax,
        # for example:
        # $ pip install -e .[dev,test]
        extras_require={"gui": ["PyQt5>=5.13.1", "pyqtgraph>=0.10.0", "psutil"],},
        # If there are data files included in your packages that need to be
        # installed, specify them here.  If using Python 2.6 or less, then these
        # have to be included in MANIFEST.in as well.
        package_data={"asammdf": ["asammdf/gui/ui/*.ui"]},
        include_package_data=True,
        # Although 'package_data' is the preferred approach, in some case you may
        # need to place data files outside of your packages. See:
        # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
        # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
        #    data_files=[('my_data', ['data/data_file'])],
        # To provide executable scripts, use entry points in preference to the
        # "scripts" keyword. Entry points provide cross-platform support and allow
        # pip to create the appropriate form of executable for the target platform.
        entry_points={"console_scripts": ["asammdf=asammdf.gui.asammdfgui:main",],},
        ext_package="asammdf.blocks",
        ext_modules=[
            Extension(
                "cutils",
                ["asammdf/blocks/cutils.c"],
                include_dirs=[numpy.get_include()],
            )
        ],
    )

except:
    setup(
        name="asammdf",
        # Versions should comply with PEP440.  For a discussion on single-sourcing
        # the version across setup.py and the project code, see
        # https://packaging.python.org/en/latest/single_source_version.html
        version=version,
        description="ASAM MDF measurement data file parser",
        long_description=long_description,
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
            "Development Status :: 4 - Beta",
            # Indicate who your project is intended for
            "Intended Audience :: Developers",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            # Pick your license as you wish (should match "license" above)
            "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
            # Specify the Python versions you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both.
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
        ],
        # What does your project relate to?
        keywords="read reader edit editor parse parser asam mdf measurement",
        # You can just specify the packages manually here if your project is
        # simple. Or you can use find_packages().
        packages=find_packages(exclude=["contrib", "docs", "test"]),
        # Alternatively, if you want to distribute just a my_module.py, uncomment
        # this:
        #   py_modules=["my_module"],
        # List run-time dependencies here.  These will be installed by pip when
        # your project is installed. For an analysis of "install_requires" vs pip's
        # requirements files see:
        # https://packaging.python.org/en/latest/requirements.html
        install_requires=[
            "numpy>=1.16.1",
            "pandas",
            "numexpr",
            "wheel",
            "canmatrix",
            "natsort",
            "lxml",
            "cChardet==2.1.5",
            "lz4",
        ],
        # List additional groups of dependencies here (e.g. development
        # dependencies). You can install these using the following syntax,
        # for example:
        # $ pip install -e .[dev,test]
        extras_require={"gui": ["PyQt5>=5.13.1", "pyqtgraph>=0.10.0", "psutil"],},
        # If there are data files included in your packages that need to be
        # installed, specify them here.  If using Python 2.6 or less, then these
        # have to be included in MANIFEST.in as well.
        package_data={"asammdf": ["asammdf/gui/ui/*.ui"]},
        include_package_data=True,
        # Although 'package_data' is the preferred approach, in some case you may
        # need to place data files outside of your packages. See:
        # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
        # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
        #    data_files=[('my_data', ['data/data_file'])],
        # To provide executable scripts, use entry points in preference to the
        # "scripts" keyword. Entry points provide cross-platform support and allow
        # pip to create the appropriate form of executable for the target platform.
        entry_points={"console_scripts": ["asammdf=asammdf.gui.asammdfgui:main",],},
    )
