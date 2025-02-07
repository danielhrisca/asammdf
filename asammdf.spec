# -*- mode: python -*-
import os
from pathlib import Path
import sys

from PyInstaller.utils.hooks import copy_metadata

from asammdf.gui.dialogs.dependencies_dlg import find_all_dependencies

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

asammdf_path = Path.cwd() / "src" / "asammdf" / "app" / "asammdfgui.py"

block_cipher = None
added_files = []

# get metadata for importlib.metadata (used by DependenciesDlg)
for dep in find_all_dependencies("asammdf"):
    added_files += copy_metadata(dep)

for root, dirs, files in os.walk(asammdf_path.parent / "ui"):
    for file in files:
        if file.lower().endswith(("ui", "png", "qrc")):
            added_files.append((os.path.join(root, file), os.path.join("asammdf", "gui", "ui")))

import pyqtlet2

pyqtlet_path = Path(pyqtlet2.__path__[0]).resolve()
site_packages = pyqtlet_path.parent
site_packages_str = os.path.join(str(site_packages), "")
for root, dirs, files in os.walk(pyqtlet_path):
    for file in files:
        if not file.lower().endswith(("py", "pyw")):
            file_name = os.path.join(root, file)
            dest = root.replace(site_packages_str, "")
            added_files.append((file_name, dest))

a = Analysis(
    [asammdf_path],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        "numpy.core._dtype_ctypes",
        "canmatrix.formats",
        "canmatrix.formats.dbc",
        "canmatrix.formats.arxml",
        "canmatrix.formats.ldf",
        "asammdf.blocks.cutils",
        "import pyqtgraph.canvas.CanvasTemplate_pyside6",
        "import pyqtgraph.canvas.TransformGuiTemplate_pyside6",
        "import pyqtgraph.console.template_pyside6",
        "import pyqtgraph.graphicsItems.PlotItem.plotConfigTemplate_pyside6",
        "import pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate_pyside6",
        "import pyqtgraph.GraphicsScene.exportDialogTemplate_pyside6",
        "import pyqtgraph.imageview.ImageViewTemplate_pyside6",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name="asammdfgui",
    exclude_binaries=True,
    bootloader_ignore_signals=False,
    debug=False,
    strip=False,
    console=False,
    icon="asammdf.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    Tree(asammdf_path.parent),
    upx=False,
    upx_exclude=[],
    name="asammdfgui",
)
