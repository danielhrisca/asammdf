pip install -U black isort && ^
black --config pyproject.toml . && ^
black --config pyproject.toml asammdf.spec && ^
isort --settings-path pyproject.toml asammdf.spec && ^
isort --settings-path pyproject.toml .
