### DTSC - BRAIN elia-hackaton solution

This repository contains the solution for the elia-hackaton 2025 challenge.

## Cloning the Repository

Install poetry
```sh

curl -sSL https://install.python-poetry.org | python3 -
```

Clone the repository

```sh
git git clone https://github.com/DT-Service-Consulting/elia-hackaton.git
```

Access the folder and install the package with `poetry`
```sh
cd elia-hackaton

poetry install
```

Activate the local environment
```sh
source .venv/bin/activate
```

## Running the tests

```sh
poetry run pytest --cov=elia_hackaton --cov-report=html
```
