pip install -U black ruff && ^
black --config pyproject.toml . && ^
black --config pyproject.toml asammdf.spec && ^
black --config pyproject.toml setup.py && ^
ruff check --fix ./src && ^
ruff check --fix ./test && ^
ruff check --fix ./setup.py
