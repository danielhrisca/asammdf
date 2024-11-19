pip install --upgrade black~=24.1 ruff~=0.7.0 && ^
black --config pyproject.toml . asammdf.spec && ^
ruff check --fix
