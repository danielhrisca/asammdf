# -*- mode: python -*-
import sys
import site
from pathlib import Path

if sys.platform.startswith('linux'):
    asammdf_path = Path('/home/appveyor/venv3.7/lib/python3.7/site-packages/asammdf/gui/asammdfgui.py')
elif sys.platform.startswith('win'):
    asammdf_path = Path(site.getsitepackages()[1]) / 'asammdf' / 'gui' / 'asammdfgui.py'
else:
    asammdf_path = Path('/Users/appveyor/venv3.7/lib/python3.7/site-packages/asammdf/gui/asammdfgui.py')

block_cipher = None
added_files = []

for root, dirs, files in os.walk(asammdf_path.parent / 'ui'):
    for file in files:
        if file.lower().endswith(('ui', 'png', 'qrc')):
            added_files.append((os.path.join(root, file), os.path.join('asammdf', 'gui', 'ui')))

a = Analysis([asammdf_path],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'numpy.core._dtype_ctypes',
        'numpy.random.common',
        'numexpr',
        'numpy.core._dtype_ctypes',
        'numpy.random.bounded_integers',
        'numpy.random.entropy',
        'numpy.random.common',
        'pandas._libs.tslibs.nattype',
        'pandas._libs.tslibs.np_datetime',
        'pandas._libs.tslibs.timedeltas',
        'pandas._libs.skiplist',
        'canmatrix.formats',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher
)
    
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)
    
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    Tree(asammdf_path.parent),
    name='asammdfgui',
    debug=True,
    strip=False,
    upx=True,
    console=True,
    icon='asammdf.ico',
)
