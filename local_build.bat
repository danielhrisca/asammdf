set proxy="http://%~1:%~2@cias3basic.conti.de:8080"
set py="c:\Python36x64Daxil\python.exe"
SET pyinstaller="c:\Python36x64Daxil\Scripts\pyinstaller.exe"
set pypiserver="--trusted-host iasp026a.cw01.contiwan.com --index-url http://iasp026a.cw01.contiwan.com:9000 --extra-index-url https://pypi.python.org/simple"
%py% -m pip install --upgrade --proxy=%proxy% pip PyQt5
%py% -m pip install --upgrade --proxy=%proxy% https://github.com/ebroecker/canmatrix/archive/development.zip \\iasp026a\XFER\pypi\pyqtgraph-0.11.0.dev0+gf18c098.tar.gz \\iasp026a\XFER\pypi\mfile-0.9.3.tar.gz \\iasp026a\XFER\pypi\pyxcp-0.10.2.tar.gz \\iasp026a\XFER\pypi\PyInstaller-3.4-py2.py3-none-any.whl https://github.com/danielhrisca/asammdf/archive/development.zip
rem %py% -m pip install --proxy=%proxy% --trusted-host iasp026a.cw01.contiwan.com --index-url http://iasp026a.cw01.contiwan.com:9000 --extra-index-url https://pypi.python.org/simple -r requirements_local_build.txt

%pyinstaller% d:\DSUsers\uidn3651\02__PythonWorkspace\asammdf\asammdf\gui\asammdfgui.py --onefile
