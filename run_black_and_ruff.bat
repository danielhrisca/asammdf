pip install -U black ruff && ^
ruff check --fix ./src && ^
ruff check --fix ./test && ^
ruff check --fix ./setup.py && ^
black --config pyproject.toml . && ^
black --config pyproject.toml asammdf.spec && ^
black --config pyproject.toml setup.py
