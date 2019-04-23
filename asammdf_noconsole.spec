# -*- mode: python -*-
import os
import sys
import site
from pathlib import Path

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # We are the main py2exe script, not a module
    HERE = os.path.dirname(os.path.abspath(sys.argv[0]))

sys.path.insert(0, HERE)

block_cipher = None

a = Analysis(['asammdf\\gui\\asammdfgui.py'],
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
    name='asammdfgui',
    debug=True,
    strip=False,
    upx=True,
    console=False,
    icon='asammdf.ico',
)
