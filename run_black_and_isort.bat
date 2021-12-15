black --config pyproject.toml asammdf\signal.py && ^
black --config pyproject.toml asammdf\mdf.py && ^
black --config pyproject.toml asammdf\blocks && ^
black --config pyproject.toml asammdf\gui\widgets && ^
black --config pyproject.toml asammdf\gui\dialogs && ^
black --config pyproject.toml asammdf\gui\asammdfgui.py && ^
black --config pyproject.toml asammdf\gui\utils.py && ^
black --config pyproject.toml asammdf.spec && ^

isort asammdf\signal.py && ^
isort asammdf\mdf.py && ^
isort asammdf\blocks && ^
isort asammdf\gui\widgets && ^
isort asammdf\gui\asammdfgui.py && ^
isort asammdf\gui\utils.py && ^
isort asammdf.spec && ^
isort asammdf\gui\dialogs