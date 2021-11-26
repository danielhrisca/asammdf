black --config pyproject.toml asammdf\signal.py && ^
black --config pyproject.toml asammdf\mdf.py && ^
black --config pyproject.toml asammdf\blocks && ^
black --config pyproject.toml asammdf\gui\widgets && ^
black --config pyproject.toml asammdf\gui\dialogs && ^
black --config pyproject.toml asammdf\gui\asammdfgui.py && ^

isort asammdf\signal.py && ^
isort asammdf\mdf.py && ^
isort asammdf\blocks && ^
isort asammdf\gui\widgets && ^
isort asammdf\gui\asammdfgui.py && ^
isort asammdf\gui\dialogs