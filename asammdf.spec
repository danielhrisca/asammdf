# -*- mode: python -*-
import sys
import site
from pathlib import Path

if sys.platform.startswith('linux'):
    asammdf_path = Path('/home/appveyor/.local/lib/python3.6/site-packages/asammdf/gui/asammdfgui.py')
else:

    asammdf_path = Path(site.getsitepackages()[1]) / 'asammdf' / 'gui' / 'asammdfgui.py'

block_cipher = None

a = Analysis([asammdf_path],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'numpy.core._dtype_ctypes'
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
    Tree(asammdf, prefix='asammdf\\' if sys.platform.startswith('linux') else 'asammdf/'),
    name='asammdfgui',
    debug=True,
    strip=False,
    upx=True,
    console=True,
    icon='asammdf.ico',
)
