# -*- mode: python -*-
import os
import sys
import site

sys.path.insert(0, '.')

asammdf = os.path.join(site.getsitepackages()[1], 'asammdf')
block_cipher = None

a = Analysis(['asammdf\\gui\\asammdfgui.py'],
    pathex=['..\\asammdf'],
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
    Tree(asammdf, prefix='asammdf\\'),
    name='py-asammdf',
    debug=True,
    strip=False,
    upx=True,
    console=True,
    icon='asammdf.png',
)
