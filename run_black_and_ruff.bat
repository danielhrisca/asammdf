pip install -U black ruff && ^
black --config pyproject.toml ./src && ^
black --config pyproject.toml ./test && ^
black --config pyproject.toml asammdf.spec && ^
black --config pyproject.toml setup.py && ^
ruff check --fix ./src && ^
ruff check --fix ./test && ^
ruff check --fix ./setup.py
