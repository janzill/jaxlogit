#!/usr/bin/env bash

set -eu

main() {

    ensure_pip

    setup_venv

    # Install required packages
    python -m pip install setuptools uv
    python -m uv pip install -e .[dev]

    echo
    echo "Activate venv with: . venv/bin/activate"
}

ensure_pip() {
    PY_EXE=python
    python --version || PY_EXE=python3
    ${PY_EXE} -m pip list > /dev/null || (sudo apt update && sudo apt install -y python3-pip)

    # otherwise we get a warning for every subsequent call to pip
    ${PY_EXE} -m pip install --upgrade pip
}

setup_venv() {
    # Figure out if we need python or python3 as our exe name
    PY_EXE=python
    python --version || PY_EXE=python3

    # Install virtualenv package, use it to create a venv and then activate it
    ${PY_EXE} -m pip install virtualenv
    [[ -d venv ]] || ${PY_EXE} -m virtualenv venv
    . venv/bin/activate
}

main "$@"
