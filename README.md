### DTSC - BRAIN elia-hackaton solution

This repository contains the solution for the elia-hackaton 2025 challenge.

## Cloning the Repository

Install `poetry`:
```sh
pip install poetry
```

Clone the repository:
```sh
git clone https://github.com/DT-Service-Consulting/elia-hackaton.git
```

Access the folder and install the package with `poetry`:
```sh
cd elia-hackaton

poetry install
```

Activate the local environment:
```sh
source .venv/bin/activate
```

## Running the app

```sh
poetry run  streamlit run elia_hackaton/dashboard.py
```
## Running the tests

```sh
poetry run pytest --cov=elia_hackaton --cov-report=html
```

## Troubleshooting poetry

If you find an error such as this:
```sh
/bin/bash: line 1: poetry: command not found
```
it means that poetry is not in your PATH. Try the following:
```sh
# find poetry executable
find ~ -name poetry -type f 

# add the path to your .bashrc or .zshrc (macOS)
echo 'export PATH="$PATH:/path/to/poetry"' >> ~/.bashrc
```
