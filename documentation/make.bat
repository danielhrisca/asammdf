@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=python -msphinx
)
set SOURCEDIR=.
set BUILDDIR=_build
set SPHINXPROJ=asammdf

set ALLSPHINXOPTS=-d %BUILDDIR%/doctrees %SPHINXOPTS% .
if NOT "%PAPER%" == "" (
    set ALLSPHINXOPTS=-D latex_paper_size=%PAPER% %ALLSPHINXOPTS%
)

if "%1" == "" goto help

if "%1" == "livehtml" (
    sphinx-autobuild  -B -s 1 -b html %ALLSPHINXOPTS% %BUILDDIR%/html
    if errorlevel 1 exit /b 1
    echo.
    echo.Testing of doctests in the sources finished, look at the ^
results in %BUILDDIR%/html/output.txt.
    goto end
)

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The Sphinx module was not found. Make sure you have Sphinx installed,
	echo.then set the SPHINXBUILD environment variable to point to the full
	echo.path of the 'sphinx-build' executable. Alternatively you may add the
	echo.Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

:end
popd
