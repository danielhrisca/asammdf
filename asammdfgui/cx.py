import sys
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
    "packages": [
        "os",
        'numpy',
        'pyqtgraph',
    ],
    "excludes": [
        'gtk',
        'tkinter',
        'bcolz',
        'bokeh',
        'bs4',
        'alabaster',
        'babel',
        'cloudpickle',
        'blosc',
        'colorama',
        'cryptography',
        'Cython',
        'dask',
        'IPython',
        'jedi',
        'jinja2',
        'Levenshtein',
        'lxml',
        'netCDF4',
        'nose',
        'openpyxl',
        'PIL',
        'statsmodels',
        'tables',
        'sphinx',
        'sphinx_rtd_theme',
        'sqlalchemy',
        'tornado',
        'zmq',
        'botocore',
        'notebook',
        'cvxopt',
        'bottleneck',
        'cyordereddict',
        'cytoolz',
        'markupsafe',
        'psutil',
        'simplejson',
        'boto3',
        'docutils',
        'html5lib',
        'ipykernel',
        'ipyparallel',
        'ipython_genutils',
        'ipywidgets',
        'jmespath',
        'joblib',
        'jupyter_client',
        'jupyter_core',
        'mock',
        'msgpack',
        'pbr',
        'pkg_resources',
        'py',
        'pycparser',
        's3fs',
        's3transfer',
        'seaborn',
        'sortedcontainers',
        'tblib',
        'test',
        'toolz',
        'traitlets',
        'xarray',
        'xlrd',
        'zict',
    ],
}

# GUI applications require a different base on Windows (the default is for a
# console application).
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="asammdfgui",
    version ="3.4.1",
    description="GUI for asammdf",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            "gui.py",
            base=base,
            targetName="asammdf.exe",
        )
    ],
)

